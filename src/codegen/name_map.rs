use crate::codegen::{
    self, AllocationScheme, Dimension, Function, Instruction, ParamValKey,
};
use crate::ir::{self, dim, DimMap, InstId, Type};
use fxhash::FxHashMap;
use itertools::Itertools;
use num::bigint::BigInt;
use num::rational::Ratio;
use std;
use std::borrow::Cow;
use std::collections::hash_map;
use utils::*;

// TODO(cleanup): refactor

/// A value that can be named.
#[derive(Copy, Clone)]
pub enum Operand<'a> {
    InductionLevel(ir::IndVarId, ir::DimId),
    Operand(&'a ir::Operand),
}

/// Print the appropriate String for a given value on target
/// Values could be constants or variables.
pub trait ValuePrinter {
    /// Provides a string representing a floating point constant.
    fn get_const_float(&self, _: &Ratio<BigInt>, _: u16) -> String;
    /// Provides a string representing an integer constant.
    fn get_const_int(&self, _: &BigInt, _: u16) -> String;
    /// Provides a name for a variable of the given type.
    fn name(&mut self, t: Type) -> String;
    /// Generates a name for a parameter.
    fn name_param(&mut self, p: ParamValKey) -> String;
}

/// Maps variables to names.
// TODO(cc_perf): use an arena rather that ref-counted strings
pub struct NameMap<'a, 'b, VP: ValuePrinter> {
    /// Provides fresh names.
    value_printer: &'b mut VP,
    /// Keeps track of the names of the variables used in the kernel
    variables: FxHashMap<ir::VarId, VariableNames>,
    /// Keeps track of the name of the values produced by instructions.
    insts: FxHashMap<InstId, VariableNames>,
    /// Keeps track of loop index names.
    indexes: FxHashMap<ir::DimId, RcStr>,
    /// Keeps track of parameter names, both in the code and in the arguments.
    params: FxHashMap<ParamValKey<'a>, (String, String)>,
    /// Keeps track of memory block address names.
    mem_blocks: FxHashMap<ir::MemId, String>,
    /// Keeps track of the next fresh ID that can be assigned to a loop.
    num_loop: u32,
    /// Tracks the current index on expanded dimensions.
    current_indexes: FxHashMap<ir::DimId, usize>,
    /// Total number threads.
    #[cfg(feature = "mppa")]
    total_num_threads: u32,
    /// Tracks the name of induction variables partial names.
    induction_vars: FxHashMap<ir::IndVarId, String>,
    induction_levels: FxHashMap<(ir::IndVarId, ir::DimId), String>,
    /// Casted sizes.
    size_casts: FxHashMap<(&'a codegen::Size, ir::Type), String>,
    /// Guard to use in front of instructions with side effects.
    side_effect_guard: Option<RcStr>,
}

#[derive(Clone)]
pub enum Nameable<'a> {
    Ident(Cow<'a, str>),
    Operand(Operand<'a>),
    InductionVar(ir::IndVarId),
    DimIndex(ir::DimId),
    Constant(BigInt, u16),
    Size(Cow<'a, codegen::Size>, ir::Type),
}

impl<'a> Nameable<'a> {
    pub fn name<'c, VP: ValuePrinter>(
        &'c self,
        namer: &'c NameMap<'_, '_, VP>,
    ) -> Cow<'c, str> {
        use Nameable::*;

        match self {
            Ident(name) => Cow::Borrowed(&name),
            &Operand(operand) => namer.name(operand),
            &InductionVar(ind_var) => namer.name_induction_var(ind_var, None),
            &DimIndex(dim_id) => Cow::Borrowed(namer.name_index(dim_id)),
            &Constant(ref value, bits) => {
                Cow::Owned(namer.value_printer().get_const_int(&value, bits))
            }
            Size(size, t) => namer.name_size(&size, *t),
        }
    }
}

pub trait IntoNameable<'a> {
    fn into_nameable(self) -> Nameable<'a>;
}

impl<'a> IntoNameable<'a> for &'a str {
    fn into_nameable(self) -> Nameable<'a> {
        Nameable::Ident(Cow::Borrowed(self))
    }
}

impl<'a> IntoNameable<'a> for String {
    fn into_nameable(self) -> Nameable<'a> {
        Nameable::Ident(Cow::Owned(self))
    }
}

impl<'a> IntoNameable<'a> for Operand<'a> {
    fn into_nameable(self) -> Nameable<'a> {
        Nameable::Operand(self)
    }
}

impl<'a> IntoNameable<'a> for &'a ir::Operand {
    fn into_nameable(self) -> Nameable<'a> {
        Operand::Operand(self).into_nameable()
    }
}

impl<'a> IntoNameable<'a> for ir::IndVarId {
    fn into_nameable(self) -> Nameable<'a> {
        Nameable::InductionVar(self)
    }
}

impl<'a> IntoNameable<'a> for (ir::IndVarId, Option<ir::DimId>) {
    fn into_nameable(self) -> Nameable<'a> {
        if let Some(dim_id) = self.1 {
            Operand::InductionLevel(self.0, dim_id).into_nameable()
        } else {
            self.0.into_nameable()
        }
    }
}

impl<'a> IntoNameable<'a> for ir::DimId {
    fn into_nameable(self) -> Nameable<'a> {
        Nameable::DimIndex(self)
    }
}

impl<'a> IntoNameable<'a> for &'_ Dimension<'_> {
    fn into_nameable(self) -> Nameable<'a> {
        self.id().into_nameable()
    }
}

impl<'a> IntoNameable<'a> for bool {
    fn into_nameable(self) -> Nameable<'a> {
        Nameable::Constant(if self { 1.into() } else { 0.into() }, 1)
    }
}

impl<'a> IntoNameable<'a> for i8 {
    fn into_nameable(self) -> Nameable<'a> {
        Nameable::Constant(self.into(), 8)
    }
}

impl<'a> IntoNameable<'a> for i16 {
    fn into_nameable(self) -> Nameable<'a> {
        Nameable::Constant(self.into(), 16)
    }
}

impl<'a> IntoNameable<'a> for i32 {
    fn into_nameable(self) -> Nameable<'a> {
        Nameable::Constant(self.into(), 32)
    }
}

impl<'a> IntoNameable<'a> for i64 {
    fn into_nameable(self) -> Nameable<'a> {
        Nameable::Constant(self.into(), 64)
    }
}

impl<'a> IntoNameable<'a> for &'a codegen::Size {
    fn into_nameable(self) -> Nameable<'a> {
        (self, ir::Type::I(32)).into_nameable()
    }
}

impl<'a> IntoNameable<'a> for codegen::Size {
    fn into_nameable(self) -> Nameable<'a> {
        (self, ir::Type::I(32)).into_nameable()
    }
}

impl<'a> IntoNameable<'a> for (&'a codegen::Size, ir::Type) {
    fn into_nameable(self) -> Nameable<'a> {
        Nameable::Size(Cow::Borrowed(self.0), self.1)
    }
}

impl<'a> IntoNameable<'a> for (codegen::Size, ir::Type) {
    fn into_nameable(self) -> Nameable<'a> {
        Nameable::Size(Cow::Owned(self.0), self.1)
    }
}

impl<'a, 'b, VP: ValuePrinter> NameMap<'a, 'b, VP> {
    /// Creates a new `NameMap`.
    pub fn new(function: &'a Function<'a>, value_printer: &'b mut VP) -> Self {
        let mut mem_blocks = FxHashMap::default();
        // Setup parameters names.
        let params = function
            .device_code_args()
            .map(|val| {
                let var_name = value_printer.name(val.t());
                let param_name = value_printer.name_param(val.key());
                if let ParamValKey::GlobalMem(id) = val.key() {
                    mem_blocks.insert(id, var_name.clone());
                }
                (val.key(), (var_name, param_name))
            })
            .collect();
        // Name dimensions indexes.
        let mut indexes = FxHashMap::default();
        for dim in function.dimensions() {
            let name = RcStr::new(value_printer.name(Type::I(32)));
            for id in dim.dim_ids() {
                indexes.insert(id, name.clone());
            }
        }
        // Name induction levels.
        let mut induction_levels = FxHashMap::default();
        let mut induction_vars = FxHashMap::default();
        for level in function.induction_levels() {
            let name = value_printer.name(level.t());
            if let Some((dim, _)) = level.increment {
                induction_levels.insert((level.ind_var, dim), name);
            } else {
                induction_vars.insert(level.ind_var, name);
            }
        }
        // Name shared memory blocks. Global mem blocks are named by parameters.
        for mem_block in function.mem_blocks() {
            if mem_block.alloc_scheme() == AllocationScheme::Shared {
                let name = value_printer.name(mem_block.ptr_type());
                mem_blocks.insert(mem_block.id(), name);
            }
        }
        let variables = VariableNames::create(function, value_printer);
        let mut name_map = NameMap {
            value_printer,
            insts: FxHashMap::default(),
            variables,
            num_loop: 0,
            current_indexes: FxHashMap::default(),
            #[cfg(feature = "mppa")]
            total_num_threads: function.num_threads(),
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
                    let name = name_map.name(value).to_string();
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

    pub fn value_printer(&self) -> &VP {
        self.value_printer
    }

    pub fn value_printer_mut(&mut self) -> &mut VP {
        self.value_printer
    }

    /// Returns the total number of threads.
    #[cfg(feature = "mppa")]
    pub fn total_num_threads(&self) -> u32 {
        self.total_num_threads
    }

    /// Generates a variable of the given `Type`.
    pub fn gen_name(&mut self, t: Type) -> String {
        self.value_printer.name(t)
    }

    /// Generates an ID for a loop.
    pub fn gen_loop_id(&mut self) -> u32 {
        let id = self.num_loop;
        self.num_loop += 1;
        id
    }

    pub fn name<'c>(&'c self, operand: Operand<'c>) -> Cow<'c, str> {
        match operand {
            Operand::Operand(op) => self.name_op(op),
            Operand::InductionLevel(ind_var, level) => {
                self.name_induction_var(ind_var, Some(level))
            }
        }
    }

    /// Asigns a name to an operand.
    pub fn name_op<'c>(&'c self, operand: &'c ir::Operand) -> Cow<'c, str> {
        self.name_op_with_indexes(operand, Cow::Borrowed(&self.current_indexes))
    }

    /// Returns the name of the operand, for the given indexes on the given dimensions.
    fn name_op_with_indexes<'c>(
        &'c self,
        operand: &'c ir::Operand,
        indexes: Cow<FxHashMap<ir::DimId, usize>>,
    ) -> Cow<'c, str> {
        match operand {
            ir::Operand::Int(val, len) => {
                Cow::Owned(self.value_printer.get_const_int(val, *len))
            }
            ir::Operand::Float(val, len) => {
                Cow::Owned(self.value_printer.get_const_float(val, *len))
            }
            ir::Operand::Inst(id, _, dim_map, _)
            | ir::Operand::Reduce(id, _, dim_map, _) => {
                Cow::Borrowed(self.name_mapped_inst(*id, indexes.into_owned(), dim_map))
            }
            ir::Operand::Index(id) => {
                if let Some(idx) = indexes.get(id) {
                    Cow::Owned(format!("{}", idx))
                } else {
                    Cow::Borrowed(&self.indexes[id])
                }
            }
            ir::Operand::Param(p) => self.name_param_val(ParamValKey::External(&*p)),
            ir::Operand::Addr(id) => self.name_addr(*id),
            ir::Operand::InductionVar(id, _) => self.name_induction_var(*id, None),
            ir::Operand::Variable(val_id, _t) => {
                Cow::Borrowed(&self.variables[val_id].get_name(&indexes))
            }
        }
    }

    /// Returns the name of the instruction.
    pub fn name_inst(&self, inst: ir::InstId) -> &str {
        self.name_mapped_inst(inst, self.current_indexes.clone(), &dim::Map::empty())
    }

    pub fn indexed_inst_name(
        &self,
        inst: ir::InstId,
        indexes: &[(ir::DimId, u32)],
    ) -> &str {
        let mut indexes_map = self.current_indexes.clone();
        indexes_map.extend(indexes.iter().map(|&(dim, idx)| (dim, idx as usize)));
        self.name_mapped_inst(inst, indexes_map, &DimMap::empty())
    }

    pub fn indexed_op_name<'c>(
        &'c self,
        op: &'c ir::Operand,
        indexes: &[(ir::DimId, u32)],
    ) -> Cow<'c, str> {
        let mut indexes_map = self.current_indexes.clone();
        indexes_map.extend(indexes.iter().map(|&(dim, idx)| (dim, idx as usize)));
        self.name_op_with_indexes(op, Cow::Owned(indexes_map))
    }

    /// Returns the name of the instruction.
    fn name_mapped_inst(
        &self,
        id: InstId,
        mut indexes: FxHashMap<ir::DimId, usize>,
        dim_map: &DimMap,
    ) -> &str {
        for &(lhs, rhs) in dim_map.iter() {
            indexes.remove(&rhs).map(|idx| indexes.insert(lhs, idx));
        }
        self.insts[&id].get_name(&indexes)
    }

    /// Declares an instruction to the value_printer.
    fn decl_inst(&mut self, inst: &Instruction) {
        // We temporarily rely on `VariableNames` to generate instruction names until we
        // remove the need to rename variables altogether.
        let t = inst.t().unwrap();
        let dims = inst
            .instantiation_dims()
            .iter()
            .map(|&(dim, size)| (dim, size as usize));
        let names = VariableNames::new(t, dims, self.value_printer);
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
    pub fn name_index(&self, dim_id: ir::DimId) -> &str {
        &self.indexes[&dim_id]
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
    pub fn name_param(&self, param: ParamValKey<'a>) -> Cow<str> {
        Cow::Borrowed(&self.params[&param].1)
    }

    /// Returns the name of a variable representing a parameter value.
    pub fn name_param_val(&self, param: ParamValKey<'a>) -> Cow<str> {
        let name = &unwrap!(self.params.get(&param), "cannot find {:?} entry", param).0;
        Cow::Borrowed(name)
    }

    /// Returns the name of the address of a memory block.
    pub fn name_addr(&self, id: ir::MemId) -> Cow<str> {
        Cow::Borrowed(&self.mem_blocks[&id])
    }

    /// Assigns a name to an induction variable.
    // TODO(cleanup): split into name induction var and name induction level
    pub fn name_induction_var(
        &self,
        var: ir::IndVarId,
        dim: Option<ir::DimId>,
    ) -> Cow<str> {
        if let Some(dim) = dim {
            Cow::Borrowed(&self.induction_levels[&(var, dim)])
        } else {
            Cow::Borrowed(&self.induction_vars[&var])
        }
    }

    /// Declares a size cast. Returns the name of the variable only if a new variable was
    /// allcoated.
    pub fn declare_size_cast(
        &mut self,
        size: &'a codegen::Size,
        t: ir::Type,
    ) -> Option<String> {
        if size.dividend().is_empty() || t == Type::I(32) {
            return None;
        }
        match self.size_casts.entry((size, t)) {
            hash_map::Entry::Occupied(..) => None,
            hash_map::Entry::Vacant(entry) => {
                Some(entry.insert(self.value_printer.name(t)).to_string())
            }
        }
    }

    /// Assigns a name to a size.
    pub fn name_size<'c>(
        &'c self,
        size: &'c codegen::Size,
        expected_t: ir::Type,
    ) -> Cow<'c, str> {
        match (size.dividend(), expected_t) {
            (&[], _) => {
                assert_eq!(size.divisor(), 1);
                Cow::Owned(size.factor().to_string())
            }
            ([p], Type::I(32)) if size.factor() == 1 && size.divisor() == 1 => {
                self.name_param_val(ParamValKey::External(&**p))
            }
            (_, Type::I(32)) => self.name_param_val(ParamValKey::Size(size)),
            _ => {
                let size = unwrap!(self.size_casts.get(&(size, expected_t)));
                Cow::Borrowed(size)
            }
        }
    }

    /// Returns the side-effect guard, if any is set.
    pub fn side_effect_guard(&self) -> Option<RcStr> {
        self.side_effect_guard.clone()
    }

    /// Sets the predicate to use in front of side-effect instruction.
    pub fn set_side_effect_guard(&mut self, guard: Option<RcStr>) {
        self.side_effect_guard = guard;
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
struct VariableNames {
    indexes: Vec<VarNameIndex>,
    names: std::rc::Rc<NDArray<String>>,
}

/// Indicates how to compute indexes in the n-dimensional arrays that store a value names.
#[derive(Clone, Copy, Debug)]
enum VarNameIndex {
    /// Use the current index of a dimension.
    FromDim(ir::DimId),
    /// Take the last index.
    Last,
}

impl VariableNames {
    /// Returns the name of the variable for the given dimension indexes.
    fn get_name(&self, dim_indexes: &FxHashMap<ir::DimId, usize>) -> &str {
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

    /// Creates the mapping from variables to names.
    fn create(
        function: &codegen::Function,
        value_printer: &mut dyn ValuePrinter,
    ) -> FxHashMap<ir::VarId, Self> {
        let mut vars_names = FxHashMap::<_, Self>::default();
        for var in function.variables() {
            let names = if let Some(alias) = var.alias() {
                // `codegen::Function` lists variables that depend on an alias after the
                // alias so we know the alias is already in `vars_names`.
                vars_names[&alias.other_variable()].create_alias(alias.dim_mapping())
            } else {
                Self::new(var.t(), var.instantiation_dims(), value_printer)
            };
            vars_names.insert(var.id(), names);
        }
        vars_names
    }

    /// Creates a new set of names for a variable instantiated along the given dimensions.
    fn new<IT>(t: ir::Type, dims: IT, value_printer: &mut dyn ValuePrinter) -> Self
    where
        IT: IntoIterator<Item = (ir::DimId, usize)>,
    {
        let (indexes, dim_sizes): (Vec<_>, Vec<_>) = dims
            .into_iter()
            .map(|(dim, size)| (VarNameIndex::FromDim(dim), size))
            .unzip();
        let num_names = dim_sizes.iter().product::<usize>();
        let names = (0..num_names).map(|_| value_printer.name(t)).collect();
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
    use super::*;
    use crate::ir;

    /// A `ValuePrinter` for use in tests.
    #[derive(Default)]
    pub struct FakeValuePrinter {
        next_name: usize,
    }

    impl ValuePrinter for FakeValuePrinter {
        fn name(&mut self, _: ir::Type) -> String {
            let name = format!("%{}", self.next_name);
            self.next_name += 1;
            name
        }

        fn name_param(&mut self, _: ParamValKey) -> String {
            self.name(ir::Type::I(0))
        }

        fn get_const_float(&self, _: &Ratio<BigInt>, _: u16) -> String {
            "1.0".to_owned()
        }

        fn get_const_int(&self, _: &BigInt, _: u16) -> String {
            "1".to_owned()
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
        let mut value_printer = FakeValuePrinter::default();
        let root_names =
            VariableNames::new(t, vec![(dim0, 3), (dim1, 5)], &mut value_printer);
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
