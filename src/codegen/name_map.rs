use std::borrow::{Borrow, Cow, ToOwned};
use std::collections::hash_map;
use std::convert::TryFrom;

use fxhash::FxHashMap;
use itertools::Itertools;
use typed_arena::Arena;
use utils::*;

use crate::codegen::{
    self, AllocationScheme, Dimension, Function, Instruction, ParamValKey,
};
use crate::ir::{self, dim, DimMap, InstId, Type};

use super::llir::{self, Register};

// TODO(cleanup): refactor

/// A value that can be named.
#[derive(Copy, Clone)]
pub enum Operand<'a> {
    InductionLevel(ir::IndVarId, ir::DimId),
    Operand(&'a ir::Operand),
}

/// Print the appropriate String for a given value on target
/// Values could be constants or variables.
pub trait NameGenerator {
    /// Provides a name for a variable of the given type.
    fn name(&mut self, t: Type) -> String;
    /// Generates a name for a parameter.
    fn name_param(&mut self, p: ParamValKey) -> String;
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
    /// Keeps track of parameter names, both in the code and in the arguments.
    params: FxHashMap<ParamValKey<'a>, (Register<'a>, &'a str)>,
    /// Keeps track of memory block address names.
    mem_blocks: FxHashMap<ir::MemId, Register<'a>>,
    /// Keeps track of the next fresh ID that can be assigned to a loop.
    num_loop: u32,
    /// Tracks the current index on expanded dimensions.
    current_indexes: FxHashMap<ir::DimId, usize>,
    /// Tracks the name of induction variables partial names.
    induction_vars: FxHashMap<ir::IndVarId, llir::Operand<'a>>,
    induction_levels: FxHashMap<(ir::IndVarId, ir::DimId), Register<'a>>,
    /// Casted sizes.
    size_casts: FxHashMap<(&'a codegen::Size, ir::Type), Register<'a>>,
    /// Guard to use in front of instructions with side effects.
    side_effect_guard: Option<Register<'a>>,
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
                let var_name =
                    Register::new(interner.intern(namegen.name(val.t())), val.t());
                let param_name = interner.intern(namegen.name_param(val.key()));
                if let ParamValKey::GlobalMem(id) = val.key() {
                    mem_blocks.insert(id, var_name);
                }
                (val.key(), (var_name, param_name))
            })
            .collect();
        // Name dimensions indexes.
        let mut indexes = FxHashMap::default();
        for dim in function.dimensions() {
            let name = interner.intern(namegen.name(Type::I(32)));
            for id in dim.dim_ids() {
                indexes.insert(id, Register::new(name, Type::I(32)));
            }
        }
        // Name induction levels.
        let mut induction_levels = FxHashMap::default();
        let mut induction_vars = FxHashMap::default();
        for level in function.induction_levels() {
            let name = interner.intern(namegen.name(level.t()));
            if let Some((dim, _)) = level.increment {
                induction_levels
                    .insert((level.ind_var, dim), Register::new(name, level.t()));
            } else {
                induction_vars
                    .insert(level.ind_var, Register::new(name, level.t()).into());
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
            size_casts: FxHashMap::default(),
            indexes,
            params,
            mem_blocks,
            induction_vars,
            induction_levels,
            side_effect_guard: None,
        };
        // Setup induction variables.
        for var in function.induction_vars() {
            match var.value.components().collect_vec()[..] {
                // In the first case, the value is computed elsewhere.
                [] => (),
                [value] => {
                    let name = name_map.name(value);
                    name_map.induction_vars.insert(var.id, name);
                }
                _ => panic!(
                    "inductions variables with more than two components must be
                            computed by the outermost induction level"
                ),
            };
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
    pub fn gen_loop_id(&mut self) -> u32 {
        let id = self.num_loop;
        self.num_loop += 1;
        id
    }

    pub fn name(&self, operand: Operand<'a>) -> llir::Operand<'a> {
        match operand {
            Operand::Operand(op) => self.name_op(op),
            Operand::InductionLevel(ind_var, level) => {
                self.name_induction_var(ind_var, Some(level))
            }
        }
    }

    /// Asigns a name to an operand.
    pub fn name_op(&self, operand: &'a ir::Operand) -> llir::Operand<'a> {
        self.name_op_with_indexes(operand, Cow::Borrowed(&self.current_indexes))
    }

    /// Returns the name of the operand, for the given indexes on the given dimensions.
    fn name_op_with_indexes(
        &self,
        operand: &'a ir::Operand,
        indexes: Cow<FxHashMap<ir::DimId, usize>>,
    ) -> llir::Operand<'a> {
        match operand {
            ir::Operand::Int(val, len) => {
                llir::Operand::IntLiteral(Cow::Borrowed(val), *len)
            }
            ir::Operand::Float(val, len) => {
                llir::Operand::FloatLiteral(Cow::Borrowed(val), *len)
            }
            ir::Operand::Inst(id, _, dim_map, _)
            | ir::Operand::Reduce(id, _, dim_map, _) => {
                self.name_mapped_inst(*id, indexes, dim_map).into()
            }
            ir::Operand::Index(id) => {
                if let Some(&idx) = indexes.get(id) {
                    llir::Operand::int(i32::try_from(idx).unwrap())
                } else {
                    self.indexes[id].into()
                }
            }
            ir::Operand::Param(p) => {
                self.name_param_val(ParamValKey::External(&*p)).into()
            }
            ir::Operand::Addr(id) => self.name_addr(*id).into(),
            ir::Operand::InductionVar(id, _) => self.name_induction_var(*id, None),
            ir::Operand::Variable(val_id, _t) => {
                self.variables[val_id].get_name(&indexes).into()
            }
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
        let mut indexes_map = self.current_indexes.clone();
        indexes_map.extend(indexes.iter().map(|&(dim, idx)| (dim, idx as usize)));
        self.name_op_with_indexes(op, Cow::Owned(indexes_map))
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

    /// Set the current index of an unrolled dimension.
    pub fn set_current_index(&mut self, dim: &Dimension, idx: u32) {
        for id in dim.dim_ids() {
            self.current_indexes.insert(id, idx as usize);
        }
    }

    /// Unset the current index of an unrolled dimension.
    pub fn unset_current_index(&mut self, dim: &Dimension) {
        for id in dim.dim_ids() {
            assert!(self.current_indexes.remove(&id).is_some());
        }
    }

    /// Returns the name of a variable representing a parameter.
    pub fn name_param<'c>(&'c self, param: ParamValKey<'c>) -> &'a str {
        &self.params.get(&param).unwrap().1
    }

    /// Returns the name of a variable representing a parameter value.
    pub fn name_param_val<'c>(&'c self, param: ParamValKey<'c>) -> Register<'a> {
        self.params
            .get(&param)
            .unwrap_or_else(|| panic!("cannot find {:?} entry", param))
            .0
    }

    /// Returns the name of the address of a memory block.
    pub fn name_addr(&self, id: ir::MemId) -> Register<'a> {
        self.mem_blocks[&id]
    }

    /// Assigns a name to an induction variable.
    // TODO(cleanup): split into name induction var and name induction level
    pub fn name_induction_var(
        &self,
        var: ir::IndVarId,
        dim: Option<ir::DimId>,
    ) -> llir::Operand<'a> {
        if let Some(dim) = dim {
            self.induction_levels[&(var, dim)].into()
        } else {
            self.induction_vars[&var].clone()
        }
    }

    /// Declares a size cast. Returns the name of the variable only if a new variable was
    /// allcoated.
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
                llir::Operand::int(i32::try_from(size.factor()).unwrap())
            }
            ([p], Type::I(32)) if size.factor() == 1 && size.divisor() == 1 => {
                self.name_param_val(ParamValKey::External(&**p)).into()
            }
            (_, Type::I(32)) => self.name_param_val(ParamValKey::Size(size)).into(),
            _ => self.size_casts.get(&(size, expected_t)).unwrap().into(),
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

        fn name_param(&mut self, _: ParamValKey) -> String {
            self.name(ir::Type::I(0))
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
