use std::borrow::{Borrow, Cow, ToOwned};
use std::cell::RefCell;
use std::collections::hash_map;
use std::convert::TryFrom;
use std::rc::Rc;

use fxhash::FxHashMap;
use itertools::Itertools;
use typed_arena::Arena;
use utils::*;

use crate::codegen::{
    self, AllocationScheme, Dimension, Function, Instruction, ParamValKey,
};
use crate::ir::{self, dim, DimMap, InstId, Type};

use super::llir::{self, IntLiteral as _, Operand, Register};

// TODO(cleanup): refactor

/// Print the appropriate String for a given value on target
/// Values could be constants or variables.
pub trait NameGenerator {
    /// Provides a name for a variable of the given type.
    fn name(&mut self, t: Type) -> String;
}

/// Maps variables to names.
pub struct NameMap<'a> {
    interner: &'a Interner<str>,

    /// Provides fresh names.
    namegen: &'a mut dyn NameGenerator,
    /// Keeps track of the names of the variables used in the kernel
    variables: FxHashMap<ir::VarId, VariableNames<Register<'a>>>,
    /// Keeps track of the name of the values produced by instructions.
    insts: FxHashMap<InstId, VariableNames<Register<'a>>>,
    /// Keeps track of loop index names.
    indexes: FxHashMap<ir::DimId, Register<'a>>,
    /// Keeps track of parameter names in the code
    params: FxHashMap<ParamValKey<'a>, Register<'a>>,
    /// Keeps track of memory block address names.
    mem_blocks: FxHashMap<ir::MemId, Register<'a>>,
    /// Keeps track of the next fresh ID that can be assigned to a loop.
    num_loop: u32,
    /// Tracks the current index on expanded dimensions.
    current_indexes: FxHashMap<ir::DimId, usize>,
    /// Index variable for each unrolled dimension
    unrolled_indices: FxHashMap<ir::DimId, Rc<RefCell<i32>>>,
    /// Casted sizes.
    size_casts: FxHashMap<(&'a codegen::Size, ir::Type), Register<'a>>,
    /// Guard to use in front of instructions with side effects.
    side_effect_guard: Option<Register<'a>>,
    expr_to_operand: super::expr::ExprToOperand<'a>,
    /// Keeps track of induction variable names.
    induction_vars: FxHashMap<ir::IndVarId, Operand<'a>>,
    /// Keeps track of accesses variable names.
    ///
    /// Also includes predicate when applicable.
    accesses: FxHashMap<ir::AccessId, (Operand<'a>, Option<Register<'a>>)>,
}

/// An interner, used to convert owned objects into a long-lived borrowed version.
///
/// This is essentially a wrapper around an `Arena` which avoids leaking the `typed_arena`
/// dependency.
pub struct Interner<B: ToOwned + ?Sized> {
    arena: Arena<B::Owned>,
}

impl<B: ToOwned + ?Sized> Default for Interner<B> {
    fn default() -> Self {
        Interner {
            arena: Arena::new(),
        }
    }
}

impl<B: ToOwned + ?Sized> Interner<B> {
    /// Stores the owned object into the inerner's local storage and returns a reference to the
    /// stored object.
    pub fn intern(&self, owned: B::Owned) -> &B {
        (*self.arena.alloc(owned)).borrow()
    }
}

impl<'a> NameMap<'a> {
    /// Creates a new `NameMap`.
    pub fn new(
        interner: &'a Interner<str>,
        function: &'a Function<'a>,
        namegen: &'a mut dyn NameGenerator,
    ) -> Self {
        let mut mem_blocks = FxHashMap::default();
        // Setup parameters names.
        let params = function
            .device_code_args()
            .map(|val| {
                let var_t = val.t();
                let var_name = Register::new(interner.intern(namegen.name(var_t)), var_t);
                if let ParamValKey::GlobalMem(id) = val.key() {
                    mem_blocks.insert(id, var_name);
                }
                (val.key(), var_name)
            })
            .collect();
        // Name dimensions indexes.
        let mut indexes = FxHashMap::default();
        let mut unrolled_indices = FxHashMap::default();
        for dim in function.dimensions() {
            let name = interner.intern(namegen.name(Type::I(32)));
            if dim.kind() == crate::search_space::DimKind::UNROLL {
                let cell = Rc::new(RefCell::new(-43i32));
                for id in dim.dim_ids() {
                    unrolled_indices.insert(id, cell.clone());
                }
            } else {
                for id in dim.dim_ids() {
                    indexes.insert(id, Register::new(name, Type::I(32)));
                }
            }
        }
        // Name shared memory blocks. Global mem blocks are named by parameters.
        for mem_block in function.mem_blocks() {
            if mem_block.alloc_scheme() == AllocationScheme::Shared {
                let name = Register::new(
                    interner.intern(namegen.name(mem_block.ptr_type())),
                    mem_block.ptr_type(),
                );
                mem_blocks.insert(mem_block.id(), name);
            }
        }
        let mut variables: FxHashMap<_, VariableNames<_>> = FxHashMap::default();
        for var in function.variables() {
            let names = match var.alias() {
                None => VariableNames::new(var.instantiation_dims(), |_| {
                    Register::new(interner.intern(namegen.name(var.t())), var.t())
                }),
                Some(alias) => {
                    // `codegen::Function` lists variables that depend on an alias after the
                    // alias so we know the alias is already in `vars_names`.
                    variables[&alias.other_variable()].create_alias(alias.dim_mapping())
                }
            };
            variables.insert(var.id(), names);
        }

        let mut name_map = NameMap {
            interner,
            namegen,
            insts: FxHashMap::default(),
            variables,
            num_loop: 0,
            current_indexes: FxHashMap::default(),
            unrolled_indices,
            size_casts: FxHashMap::default(),
            indexes,
            params,
            mem_blocks,
            side_effect_guard: None,
            expr_to_operand: Default::default(),
            induction_vars: Default::default(),
            accesses: Default::default(),
        };

        // Setup induction variables.
        {
            let merged_dimensions: super::dimension::MergedDimensions<'_> =
                function.dimensions().collect();

            let mut builder = super::expr::ExprToOperandBuilder::new(
                function.space(),
                &merged_dimensions,
                &mut name_map,
            );

            let mut induction_vars = FxHashMap::default();
            for &(var_id, ref expr) in function.induction_vars() {
                let operand = builder.to_operand(expr);
                induction_vars.insert(var_id, operand);
            }

            let mut accesses = FxHashMap::default();
            for &(aid, ref expr, ref predicate) in function.accesses() {
                let operand = builder.to_operand(expr);
                let predicate = predicate.as_ref().map(|predicate| {
                    builder.to_operand(predicate).to_register().unwrap()
                });
                accesses.insert(aid, (operand, predicate));
            }

            let expr_to_operand = builder.finish();
            name_map.expr_to_operand = expr_to_operand;
            name_map.induction_vars = induction_vars;
            name_map.accesses = accesses;
        }

        // Setup the name of variables holding instruction results.
        for inst in function.cfg().instructions() {
            // If the instruction has a return variable, use its name instead.
            if let Some(var) = inst.result_variable() {
                name_map
                    .insts
                    .insert(inst.id(), name_map.variables[&var].clone());
            } else if let Some((inst_id, dim_map)) = inst.as_reduction() {
                name_map.decl_alias(inst, inst_id, dim_map);
            } else if inst.t().is_some() {
                name_map.decl_inst(inst);
            }
        }
        name_map
    }

    /// Generates a variable of the given `Type`.
    pub fn gen_name(&mut self, t: Type) -> Register<'a> {
        Register::new(self.interner.intern(self.namegen.name(t)), t)
    }

    /// Generates an ID for a loop.
    pub fn gen_label(&mut self, prefix: &str) -> llir::Label<'a> {
        let id = self.num_loop;
        self.num_loop += 1;
        llir::Label::new(self.interner.intern(format!("{}_{}", prefix, id)))
    }

    pub fn expr_to_operand(&self) -> &super::expr::ExprToOperand<'a> {
        &self.expr_to_operand
    }

    pub fn name_induction_var(&self, ind_var_id: ir::IndVarId) -> llir::Operand<'a> {
        self.induction_vars[&ind_var_id].clone()
    }

    pub fn name_access(
        &self,
        aid: ir::AccessId,
    ) -> (llir::Operand<'a>, Option<llir::Register<'a>>) {
        self.accesses[&aid].clone()
    }

    /// Asigns a name to an operand.
    pub fn name_op(
        &self,
        operand: &'a ir::Operand,
    ) -> (llir::Operand<'a>, Option<llir::Register<'a>>) {
        self.name_op_with_indexes(operand, Cow::Borrowed(&self.current_indexes))
    }

    /// Returns the name of the operand, for the given indexes on the given dimensions.
    fn name_op_with_indexes(
        &self,
        operand: &'a ir::Operand,
        indexes: Cow<FxHashMap<ir::DimId, usize>>,
    ) -> (llir::Operand<'a>, Option<llir::Register<'a>>) {
        match operand {
            ir::Operand::Int(val, len) => {
                (llir::Operand::IntLiteral(Cow::Borrowed(val), *len), None)
            }
            ir::Operand::Float(val, len) => {
                (llir::Operand::FloatLiteral(Cow::Borrowed(val), *len), None)
            }
            ir::Operand::Inst(id, _, dim_map, _)
            | ir::Operand::Reduce(id, _, dim_map, _) => {
                (self.name_mapped_inst(*id, indexes, dim_map).into(), None)
            }
            ir::Operand::Index(id) => {
                if let Some(&idx) = indexes.get(id) {
                    (i32::try_from(idx).unwrap().int_literal(), None)
                } else {
                    (self.indexes[id].into(), None)
                }
            }
            ir::Operand::Param(p) => {
                (self.name_param_val(ParamValKey::External(&*p)).into(), None)
            }
            ir::Operand::Addr(id) => (self.name_addr(*id).into(), None),
            ir::Operand::InductionVar(id, _) => (self.name_induction_var(*id), None),
            ir::Operand::Variable(val_id, _t) => {
                ((*self.variables[val_id].get_name(&indexes)).into(), None)
            }
            ir::Operand::ComputedAddress(aid, _t) => self.name_access(*aid),
        }
    }

    /// Returns the name of the instruction.
    pub fn name_inst(&self, inst: ir::InstId) -> Register<'a> {
        self.name_mapped_inst(
            inst,
            Cow::Borrowed(&self.current_indexes),
            &dim::Map::empty(),
        )
    }

    pub fn indexed_inst_name(
        &self,
        inst: ir::InstId,
        indexes: &[(ir::DimId, u32)],
    ) -> Register<'a> {
        let mut indexes_map = Cow::Borrowed(&self.current_indexes);
        if !indexes.is_empty() {
            indexes_map
                .to_mut()
                .extend(indexes.iter().map(|&(dim, idx)| (dim, idx as usize)));
        }
        self.name_mapped_inst(inst, indexes_map, &DimMap::empty())
    }

    pub fn indexed_op_name(
        &self,
        op: &'a ir::Operand,
        indexes: &[(ir::DimId, u32)],
    ) -> llir::Operand<'a> {
        let mut indexes_map = Cow::Borrowed(&self.current_indexes);
        if !indexes.is_empty() {
            indexes_map
                .to_mut()
                .extend(indexes.iter().map(|&(dim, idx)| (dim, idx as usize)));
        }
        let (operand, predicate) = self.name_op_with_indexes(op, indexes_map);
        // Can't do vectorized predicates
        assert!(predicate.is_none());
        operand
    }

    /// Returns the name of the instruction.
    fn name_mapped_inst(
        &self,
        id: InstId,
        mut indexes: Cow<FxHashMap<ir::DimId, usize>>,
        dim_map: &DimMap,
    ) -> Register<'a> {
        for &(lhs, rhs) in dim_map.iter() {
            indexes
                .to_mut()
                .remove(&rhs)
                .map(|idx| indexes.to_mut().insert(lhs, idx));
        }
        *self.insts[&id].get_name(&indexes)
    }

    /// Declares an instruction to the namegen.
    fn decl_inst(&mut self, inst: &Instruction) {
        // We temporarily rely on `VariableNames` to generate instruction names until we
        // remove the need to rename variables altogether.
        let t = inst.t().unwrap();
        let dims = inst
            .instantiation_dims()
            .iter()
            .map(|&(dim, size)| (dim, size as usize));
        let names = VariableNames::new(dims, |_| {
            Register::new(self.interner.intern(self.namegen.name(t)), t)
        });
        assert!(self.insts.insert(inst.id(), names).is_none());
    }

    /// Declares an instruction as an alias of another.
    fn decl_alias(&mut self, alias: &Instruction, base: InstId, dim_map: &DimMap) {
        // We temporarily rely on `VariableNames` to generate instruction names until we
        // remove the need to rename variables altogether.
        let mut mapping: FxHashMap<_, _> = dim_map.iter().cloned().collect();
        for &(dim, _) in alias.instantiation_dims() {
            mapping.insert(dim, dim);
        }
        let mut names = self.insts[&base].clone();
        let new_indexes = names
            .indexes
            .iter()
            .map(|idx| match idx {
                VarNameIndex::FromDim(dim) => mapping
                    .get(dim)
                    .map(|&dim| VarNameIndex::FromDim(dim))
                    .unwrap_or(VarNameIndex::Last),
                VarNameIndex::Last => VarNameIndex::Last,
            })
            .collect();
        names.indexes = new_indexes;
        assert!(self.insts.insert(alias.id(), names).is_none());
    }

    /// Returns the name of an index.
    pub fn name_index(&self, dim_id: ir::DimId) -> Register<'a> {
        self.indexes[&dim_id]
    }

    /// Returns the value of an index for use as an operand.
    ///
    /// The operand will be a register, except for unrolled dimensions where it will be the current
    /// value associated with the index.
    pub fn name_index_as_operand(&self, dim_id: ir::DimId) -> Operand<'a> {
        match self.unrolled_indices.get(&dim_id) {
            Some(cell) => Operand::new_index_cell(cell.clone()),
            None => self.name_index(dim_id).into_operand(),
        }
    }

    /// Set the current index of an unrolled dimension.
    pub fn set_current_index(&mut self, dim: &Dimension, idx: u32) {
        for id in dim.dim_ids() {
            self.current_indexes.insert(id, idx as usize);
            *self.unrolled_indices[&id].borrow_mut() = i32::try_from(idx).unwrap();
        }
    }

    /// Unset the current index of an unrolled dimension.
    pub fn unset_current_index(&mut self, dim: &Dimension) {
        for id in dim.dim_ids() {
            assert!(self.current_indexes.remove(&id).is_some());
            *self.unrolled_indices[&id].borrow_mut() = -42i32;
        }
    }

    /// Returns the name of a variable representing a parameter value.
    pub fn name_param_val<'c>(&'c self, param: ParamValKey<'c>) -> Register<'a> {
        *self
            .params
            .get(&param)
            .unwrap_or_else(|| panic!("cannot find {:?} entry", param))
    }

    /// Returns the name of the address of a memory block.
    pub fn name_addr(&self, id: ir::MemId) -> Register<'a> {
        self.mem_blocks[&id]
    }

    /// Declares a size cast. Returns the name of the variable only if a new variable was
    /// allocated.
    pub fn declare_size_cast(
        &mut self,
        size: &'a codegen::Size,
        t: ir::Type,
    ) -> Option<Register<'a>> {
        if size.dividend().is_empty() || t == Type::I(32) {
            return None;
        }
        match self.size_casts.entry((size, t)) {
            hash_map::Entry::Occupied(..) => None,
            hash_map::Entry::Vacant(entry) => Some(
                *entry
                    .insert(Register::new(self.interner.intern(self.namegen.name(t)), t)),
            ),
        }
    }

    /// Assigns a name to a size.
    pub fn name_size<'c>(
        &'c self,
        size: &'c codegen::Size,
        expected_t: ir::Type,
    ) -> llir::Operand<'a> {
        match (size.dividend(), expected_t) {
            (&[], _) => {
                assert_eq!(size.divisor(), 1);
                i32::try_from(size.factor())
                    .unwrap()
                    .typed_int_literal(expected_t)
                    .unwrap()
            }
            ([p], Type::I(32)) if size.factor() == 1 && size.divisor() == 1 => {
                self.name_param_val(ParamValKey::External(&**p)).into()
            }
            (_, Type::I(32)) => self.name_param_val(ParamValKey::Size(size)).into(),
            _ => (*self.size_casts.get(&(size, expected_t)).unwrap()).into(),
        }
    }

    pub fn name_div_magic<'c>(
        &'c self,
        size: &'c codegen::Size,
        expected_t: ir::Type,
    ) -> llir::Operand<'a> {
        match (size.dividend(), expected_t) {
            (&[], ir::Type::I(32)) => {
                assert_eq!(size.divisor(), 1);
                let i32_size = i32::try_from(size.factor()).unwrap();
                codegen::i32_div_magic(i32_size).int_literal()
            }
            (&[], _) => unimplemented!("constant div magic"),
            _ => self
                .name_param_val(ParamValKey::DivMagic(size, expected_t))
                .into(),
        }
    }

    pub fn name_div_shift<'c>(
        &'c self,
        size: &'c codegen::Size,
        expected_t: ir::Type,
    ) -> llir::Operand<'a> {
        match (size.dividend(), expected_t) {
            (&[], ir::Type::I(32)) => {
                assert_eq!(size.divisor(), 1);
                let i32_size = i32::try_from(size.factor()).unwrap();
                codegen::i32_div_shift(i32_size).int_literal()
            }
            (&[], _) => unimplemented!("constant div shift"),
            _ => self
                .name_param_val(ParamValKey::DivShift(size, expected_t))
                .into(),
        }
    }

    /// Returns the side-effect guard, if any is set.
    pub fn side_effect_guard(&self) -> Option<Register<'a>> {
        self.side_effect_guard
    }

    /// Sets the predicate to use in front of side-effect instruction.
    pub fn set_side_effect_guard(&mut self, guard: Option<Register<'a>>) {
        self.side_effect_guard = guard;
    }

    fn vectorize<F, T>(
        &self,
        vector_levels: &[Vec<Dimension>; 2],
        get_name: F,
    ) -> llir::ScalarOrVector<T>
    where
        F: Fn(&[(ir::DimId, u32)]) -> T,
    {
        assert!(vector_levels[0].is_empty());

        if vector_levels[1].is_empty() {
            return llir::ScalarOrVector::Scalar(get_name(&[]));
        }

        let sizes = vector_levels
            .iter()
            .flat_map(|level| level.iter().map(|d| d.size().as_int().unwrap()))
            .collect::<Vec<_>>();
        llir::ScalarOrVector::Vector(
            NDRange::new(&sizes)
                .map(|indices| {
                    get_name(
                        &vector_levels
                            .iter()
                            .flatten()
                            .zip_eq(indices)
                            .map(|(d, idx)| (d.id(), idx))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        )
    }

    /// Name an operand, vectorized on the given dimensions.
    pub fn vector_operand(
        &self,
        vector_levels: &[Vec<Dimension>; 2],
        op: &'a ir::Operand,
    ) -> llir::OpVec<'a> {
        self.vectorize(vector_levels, |indices| self.indexed_op_name(op, indices))
    }

    /// Names an instruction, vectorized on the given dimensions.
    pub fn vector_inst(
        &self,
        vector_levels: &[Vec<Dimension>; 2],
        inst: ir::InstId,
    ) -> llir::RegVec<'a> {
        self.vectorize(vector_levels, |indices| {
            self.indexed_inst_name(inst, indices)
        })
    }
}

/// Tracks the different names a variable takes when instantiated along dimensions.
///
/// When we do point-to-point communication between two loop nests through registers, we
/// must store the variables produced at each iteration at different places. Inversely,
/// when a variable is defined as the value of another they must alias. This structure
/// tracks a shared tensor of names and how to map dimension indexes to the tensor
/// indexes.
#[derive(Clone, Debug)]
struct VariableNames<T> {
    indexes: Vec<VarNameIndex>,
    names: std::rc::Rc<NDArray<T>>,
}

/// Indicates how to compute indexes in the n-dimensional arrays that store a value names.
#[derive(Clone, Copy, Debug)]
enum VarNameIndex {
    /// Use the current index of a dimension.
    FromDim(ir::DimId),
    /// Take the last index.
    Last,
}

impl<T> VariableNames<T> {
    /// Returns the name of the variable for the given dimension indexes.
    fn get_name(&self, dim_indexes: &FxHashMap<ir::DimId, usize>) -> &T {
        let indexes = self
            .indexes
            .iter()
            .zip_eq(&self.names.dims)
            .map(|(index, size)| match index {
                VarNameIndex::FromDim(dim) => {
                    dim_indexes.get(dim).cloned().unwrap_or(size - 1)
                }
                VarNameIndex::Last => size - 1,
            })
            .collect_vec();
        &self.names[&indexes]
    }

    /// Creates a new set of names for a variable instantiated along the given dimensions.
    fn new<IT, F>(dims: IT, fresh: F) -> Self
    where
        IT: IntoIterator<Item = (ir::DimId, usize)>,
        F: FnMut(usize) -> T,
    {
        let (indexes, dim_sizes): (Vec<_>, Vec<_>) = dims
            .into_iter()
            .map(|(dim, size)| (VarNameIndex::FromDim(dim), size))
            .unzip();
        let num_names = dim_sizes.iter().product::<usize>();
        let names = (0..num_names).map(fresh).collect();
        VariableNames {
            indexes,
            names: std::rc::Rc::new(NDArray::new(dim_sizes, names)),
        }
    }

    /// Creates a `VariableNames` aliasing with this one, but with the indexes mapped to
    /// different dimensions as specified by `dim_mapping`. We take the last value on
    /// dimensions mapped to `None`.
    fn create_alias(
        &self,
        dim_mapping: &FxHashMap<ir::DimId, Option<ir::DimId>>,
    ) -> Self {
        let indexes = self
            .indexes
            .iter()
            .map(|index| match index {
                VarNameIndex::Last => VarNameIndex::Last,
                VarNameIndex::FromDim(dim) => dim_mapping
                    .get(dim)
                    // Here, we have an `Option<Option<DimId>>`.
                    .map_or(VarNameIndex::FromDim(*dim), |maps_to| {
                        maps_to
                            .map(VarNameIndex::FromDim)
                            .unwrap_or(VarNameIndex::Last)
                    }),
            })
            .collect();
        VariableNames {
            indexes,
            names: self.names.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir;

    use super::*;

    /// A `NameGenerator` for use in tests.
    #[derive(Default)]
    pub struct FakeNameGenerator {
        next_name: usize,
    }

    impl NameGenerator for FakeNameGenerator {
        fn name(&mut self, _: ir::Type) -> String {
            let name = format!("%{}", self.next_name);
            self.next_name += 1;
            name
        }
    }

    /// Converts list of indexes into a `FxHashMap`.
    fn mk_index(idx: &[(ir::DimId, usize)]) -> FxHashMap<ir::DimId, usize> {
        idx.iter().cloned().collect()
    }

    /// Ensures variables names creation works correctly.
    #[test]
    fn variable_names() {
        let _ = ::env_logger::try_init();
        let dim0 = ir::DimId(0); // A dimension of size 3.
        let dim1 = ir::DimId(1); // A dimension of size 5.

        // Test without alias.
        let t = ir::Type::F(32);
        let mut namegen = FakeNameGenerator::default();
        let root_names =
            VariableNames::new(vec![(dim0, 3), (dim1, 5)], |_| namegen.name(t));
        let name_1_3 = root_names.get_name(&mk_index(&[(dim0, 1), (dim1, 3)]));
        let name_2_4 = root_names.get_name(&mk_index(&[(dim0, 2), (dim1, 4)]));
        assert!(name_1_3 != name_2_4);
        let name_2_0 = root_names.get_name(&mk_index(&[(dim0, 2), (dim1, 4)]));
        let name_2 = root_names.get_name(&mk_index(&[(dim0, 2)]));
        assert_eq!(name_2_0, name_2);

        // Test with an alias that maps a dimension.
        let dim3 = ir::DimId(0); // A dimension mapped to `dim0`.
        let mapping = [(dim0, Some(dim3))].iter().cloned().collect();
        let mapped_names = root_names.create_alias(&mapping);
        let mapped_name_1_3 = mapped_names.get_name(&mk_index(&[(dim3, 1), (dim1, 3)]));
        assert_eq!(name_1_3, mapped_name_1_3);

        // Test with an alias that takes the last value on a dimension
        let mapping = [(dim1, None)].iter().cloned().collect();
        let last_names = root_names.create_alias(&mapping);
        let last_name_2 = last_names.get_name(&mk_index(&[(dim0, 2)]));
        assert_eq!(last_name_2, name_2_4);
    }
}
