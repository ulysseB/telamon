use crate::codegen::{
    self, AllocationScheme, Dimension, Function, Instruction, ParamValKey,
};
use crate::ir::{self, dim, DimMap, InstId, Type};
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
    Operand(&'a ir::Operand<'a>),
}

/// Assign names to variables.
pub trait Namer {
    /// Provides a name for a variable of the given type.
    fn name(&mut self, t: Type) -> String;
    /// Generates a name for a parameter.
    fn name_param(&mut self, p: ParamValKey) -> String;
    /// Provides a name for a floating point constant.
    fn name_float(&self, _: &Ratio<BigInt>, _: u16) -> String;
    /// Provides a name for an integer constant.
    fn name_int(&self, _: &BigInt, _: u16) -> String;
}

/// Maps variables to names.
// TODO(cc_perf): use an arena rather that ref-counted strings
pub struct NameMap<'a, 'b, N: Namer> {
    /// Provides fresh names.
    namer: &'b mut N,
    /// Keeps track of the names of the variables used in the kernel
    variables: FnvHashMap<ir::VarId, VariableNames>,
    /// Keeps track of the name of the values produced by instructions.
    insts: FnvHashMap<InstId, VariableNames>,
    /// Keeps track of loop index names.
    indexes: FnvHashMap<ir::DimId, RcStr>,
    /// Keeps track of parameter names, both in the code and in the arguments.
    params: FnvHashMap<ParamValKey<'a>, (String, String)>,
    /// Keeps track of memory block address names.
    mem_blocks: FnvHashMap<ir::MemId, String>,
    /// Keeps track of the next fresh ID that can be assigned to a loop.
    num_loop: u32,
    /// Tracks the current index on expanded dimensions.
    current_indexes: FnvHashMap<ir::DimId, usize>,
    /// Total number threads.
    #[cfg(feature = "mppa")]
    total_num_threads: u32,
    /// Tracks the name of induction variables partial names.
    induction_vars: FnvHashMap<ir::IndVarId, String>,
    induction_levels: FnvHashMap<(ir::IndVarId, ir::DimId), String>,
    /// Casted sizes.
    size_casts: FnvHashMap<(&'a codegen::Size<'a>, ir::Type), String>,
    /// Guard to use in front of instructions with side effects.
    side_effect_guard: Option<RcStr>,
}

impl<'a, 'b, N: Namer> NameMap<'a, 'b, N> {
    /// Creates a new `NameMap`.
    pub fn new(function: &'a Function<'a>, namer: &'b mut N) -> Self {
        let mut mem_blocks = FnvHashMap::default();
        // Setup parameters names.
        let params = function
            .device_code_args()
            .map(|val| {
                let var_name = namer.name(val.t());
                let param_name = namer.name_param(val.key());
                if let ParamValKey::GlobalMem(id) = val.key() {
                    mem_blocks.insert(id, var_name.clone());
                }
                (val.key(), (var_name, param_name))
            })
            .collect();
        // Name dimensions indexes.
        let mut indexes = FnvHashMap::default();
        for dim in function.dimensions() {
            let name = RcStr::new(namer.name(Type::I(32)));
            for id in dim.dim_ids() {
                indexes.insert(id, name.clone());
            }
        }
        // Name induction levels.
        let mut induction_levels = FnvHashMap::default();
        let mut induction_vars = FnvHashMap::default();
        for level in function.induction_levels() {
            let name = namer.name(level.t());
            if let Some((dim, _)) = level.increment {
                induction_levels.insert((level.ind_var, dim), name);
            } else {
                induction_vars.insert(level.ind_var, name);
            }
        }
        // Name shared memory blocks. Global mem blocks are named by parameters.
        for mem_block in function.mem_blocks() {
            if mem_block.alloc_scheme() == AllocationScheme::Shared {
                let name = namer.name(mem_block.ptr_type());
                mem_blocks.insert(mem_block.id(), name);
            }
        }
        let variables = VariableNames::create(function, namer);
        let mut name_map = NameMap {
            namer,
            insts: FnvHashMap::default(),
            variables,
            num_loop: 0,
            current_indexes: FnvHashMap::default(),
            #[cfg(feature = "mppa")]
            total_num_threads: function.num_threads(),
            size_casts: FnvHashMap::default(),
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

    pub fn get_namer(&self) -> &N {
        &*self.namer
    }

    pub fn get_mut_namer(&mut self) -> &mut N {
        self.namer
    }

    /// Returns the total number of threads.
    #[cfg(feature = "mppa")]
    pub fn total_num_threads(&self) -> u32 {
        self.total_num_threads
    }

    /// Generates a variable of the given `Type`.
    pub fn gen_name(&mut self, t: Type) -> String {
        self.namer.name(t)
    }

    /// Generates an ID for a loop.
    pub fn gen_loop_id(&mut self) -> u32 {
        let id = self.num_loop;
        self.num_loop += 1;
        id
    }

    pub fn name(&self, operand: Operand) -> Cow<str> {
        match operand {
            Operand::Operand(op) => self.name_op(op),
            Operand::InductionLevel(ind_var, level) => {
                self.name_induction_var(ind_var, Some(level))
            }
        }
    }

    /// Asigns a name to an operand.
    pub fn name_op(&self, operand: &ir::Operand) -> Cow<str> {
        self.name_op_with_indexes(operand, Cow::Borrowed(&self.current_indexes))
    }

    /// Returns the name of the operand, for the given indexes on the given dimensions.
    fn name_op_with_indexes(
        &self,
        operand: &ir::Operand,
        indexes: Cow<FnvHashMap<ir::DimId, usize>>,
    ) -> Cow<str> {
        match *operand {
            ir::Operand::Int(ref val, len) => Cow::Owned(self.namer.name_int(val, len)),
            ir::Operand::Float(ref val, len) => {
                Cow::Owned(self.namer.name_float(val, len))
            }
            ir::Operand::Inst(id, _, ref dim_map, _)
            | ir::Operand::Reduce(id, _, ref dim_map, _) => {
                Cow::Borrowed(self.name_mapped_inst(id, indexes.into_owned(), dim_map))
            }
            ir::Operand::Index(id) => {
                if let Some(idx) = indexes.get(&id) {
                    Cow::Owned(format!("{}", idx))
                } else {
                    Cow::Borrowed(&self.indexes[&id])
                }
            }
            ir::Operand::Param(p) => self.name_param_val(ParamValKey::External(p)),
            ir::Operand::Addr(id) => self.name_addr(id),
            ir::Operand::InductionVar(id, _) => self.name_induction_var(id, None),
            ir::Operand::Variable(val_id, _t) => {
                Cow::Borrowed(&self.variables[&val_id].get_name(&indexes))
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

    pub fn indexed_op_name(
        &self,
        op: &ir::Operand,
        indexes: &[(ir::DimId, u32)],
    ) -> Cow<str> {
        let mut indexes_map = self.current_indexes.clone();
        indexes_map.extend(indexes.iter().map(|&(dim, idx)| (dim, idx as usize)));
        self.name_op_with_indexes(op, Cow::Owned(indexes_map))
    }

    /// Returns the name of the instruction.
    fn name_mapped_inst(
        &self,
        id: InstId,
        mut indexes: FnvHashMap<ir::DimId, usize>,
        dim_map: &DimMap,
    ) -> &str {
        for &(lhs, rhs) in dim_map.iter() {
            indexes.remove(&rhs).map(|idx| indexes.insert(lhs, idx));
        }
        self.insts[&id].get_name(&indexes)
    }

    /// Declares an instruction to the namer.
    fn decl_inst(&mut self, inst: &Instruction) {
        // We temporarily rely on `VariableNames` to generate instruction names until we
        // remove the need to rename variables altogether.
        let t = inst.t().unwrap();
        let dims = inst
            .instantiation_dims()
            .iter()
            .map(|&(dim, size)| (dim, size as usize));
        let names = VariableNames::new(t, dims, self.namer);
        assert!(self.insts.insert(inst.id(), names).is_none());
    }

    /// Declares an instruction as an alias of another.
    fn decl_alias(&mut self, alias: &Instruction, base: InstId, dim_map: &DimMap) {
        // We temporarily rely on `VariableNames` to generate instruction names until we
        // remove the need to rename variables altogether.
        let mut mapping: FnvHashMap<_, _> = dim_map.iter().cloned().collect();
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
    pub fn name_param(&self, param: ParamValKey) -> Cow<str> {
        let param = unsafe { std::mem::transmute(param) };
        Cow::Borrowed(&self.params[&param].1)
    }

    /// Returns the name of a variable representing a parameter value.
    pub fn name_param_val(&self, param: ParamValKey) -> Cow<str> {
        let param = unsafe { std::mem::transmute(param) };
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
        size: &'a codegen::Size<'a>,
        t: ir::Type,
    ) -> Option<String> {
        if size.dividend().is_empty() || t == Type::I(32) {
            return None;
        }
        match self.size_casts.entry((size, t)) {
            hash_map::Entry::Occupied(..) => None,
            hash_map::Entry::Vacant(entry) => {
                Some(entry.insert(self.namer.name(t)).to_string())
            }
        }
    }

    /// Assigns a name to a size.
    pub fn name_size(&self, size: &codegen::Size, expected_t: ir::Type) -> Cow<str> {
        let size: &'a codegen::Size<'a> = unsafe { std::mem::transmute(size) };
        match (size.dividend(), expected_t) {
            (&[], _) => {
                assert_eq!(size.divisor(), 1);
                Cow::Owned(size.factor().to_string())
            }
            (&[p], Type::I(32)) if size.factor() == 1 && size.divisor() == 1 => {
                self.name_param_val(ParamValKey::External(p))
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
    fn get_name(&self, dim_indexes: &FnvHashMap<ir::DimId, usize>) -> &str {
        let indexes = self
            .indexes
            .iter()
            .zip_eq(&self.names.dims)
            .map(|(index, size)| match index {
                VarNameIndex::FromDim(dim) => dim_indexes.get(dim).cloned().unwrap_or(0),
                VarNameIndex::Last => size - 1,
            })
            .collect_vec();
        &self.names[&indexes]
    }

    /// Creates the mapping from variables to names.
    fn create(
        function: &codegen::Function,
        namer: &mut Namer,
    ) -> FnvHashMap<ir::VarId, Self> {
        let mut vars_names = FnvHashMap::<_, Self>::default();
        for var in function.variables() {
            let names = if let Some(alias) = var.alias() {
                // `codegen::Function` lists variables that depend on an alias after the
                // alias so we know the alias is already in `vars_names`.
                vars_names[&alias.other_variable()].create_alias(alias.dim_mapping())
            } else {
                Self::new(var.t(), var.instantiation_dims(), namer)
            };
            vars_names.insert(var.id(), names);
        }
        vars_names
    }

    /// Creates a new set of names for a variable instantiated along the given dimensions.
    fn new<IT>(t: ir::Type, dims: IT, namer: &mut Namer) -> Self
    where
        IT: IntoIterator<Item = (ir::DimId, usize)>,
    {
        let (indexes, dim_sizes): (Vec<_>, Vec<_>) = dims
            .into_iter()
            .map(|(dim, size)| (VarNameIndex::FromDim(dim), size))
            .unzip();
        let num_names = dim_sizes.iter().product::<usize>();
        let names = (0..num_names).map(|_| namer.name(t)).collect();
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
        dim_mapping: &FnvHashMap<ir::DimId, Option<ir::DimId>>,
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

    /// A `Namer` for use in tests.
    #[derive(Default)]
    pub struct FakeNamer {
        next_name: usize,
    }

    impl Namer for FakeNamer {
        fn name(&mut self, _: ir::Type) -> String {
            let name = format!("%{}", self.next_name);
            self.next_name += 1;
            name
        }

        fn name_param(&mut self, _: ParamValKey) -> String {
            self.name(ir::Type::I(0))
        }

        fn name_float(&self, _: &Ratio<BigInt>, _: u16) -> String {
            "1.0".to_owned()
        }

        fn name_int(&self, _: &BigInt, _: u16) -> String {
            "1".to_owned()
        }
    }

    /// Converts list of indexes into a `FnvHashMap`.
    fn mk_index(idx: &[(ir::DimId, usize)]) -> FnvHashMap<ir::DimId, usize> {
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
        let mut namer = FakeNamer::default();
        let root_names = VariableNames::new(t, vec![(dim0, 3), (dim1, 5)], &mut namer);
        let name_1_3 = root_names.get_name(&mk_index(&[(dim0, 1), (dim1, 3)]));
        let name_2_4 = root_names.get_name(&mk_index(&[(dim0, 2), (dim1, 4)]));
        assert!(name_1_3 != name_2_4);
        let name_2_0 = root_names.get_name(&mk_index(&[(dim0, 2), (dim1, 0)]));
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
