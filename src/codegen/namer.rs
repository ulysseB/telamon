use codegen::{self, Dimension, Function, ParamValKey, AllocationScheme, Instruction};
use ir::{self, dim, DimMap, InstId, mem, Operand, Type};
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
pub enum Value<'a> {
    InductionLevel(ir::IndVarId, ir::dim::Id),
    Operand(&'a ir::Operand<'a>),
}

/// Assign names to variables.
pub trait Namer {
    /// Provides a name for a variable of the given type.
    fn name(&mut self, t: Type) -> String;
    /// Generates a name for a parameter.
    fn name_param(&mut self, p: ParamValKey) -> String;
    /// Provides a name for a floating point constant.
    fn name_float(&self, &Ratio<BigInt>, u16) -> String;
    /// Provides a name for an integer constant.
    fn name_int(&self, &BigInt, u16) -> String;
}

/// Maps variables to names.
// TODO(cc_perf): use an arena rather that ref-counted strings
pub struct NameMap<'a, 'b> {
    /// Provides fresh names.
    namer: std::cell::RefCell<&'b mut Namer>,
    /// Keeps track of the name of the values produced by instructions.
    insts: HashMap<InstId, (Vec<dim::Id>, NDArray<String>)>,
    /// Keeps track of loop index names.
    indexes: HashMap<dim::Id, RcStr>,
    /// Keeps track of parameter names, both in the code and in the arguments.
    params: HashMap<ParamValKey<'a>, (String, String)>,
    /// Keeps track of memory block address names.
    mem_blocks: HashMap<mem::InternalId, String>,
    /// Keeps track of the next fresh ID that can be assigned to a loop.
    num_loop: u32,
    /// Tracks the current index on expanded dimensions.
    current_indexes: HashMap<dim::Id, u32>,
    /// Total number threads.
    #[cfg(feature="mppa")]
    total_num_threads: u32,
    /// Tracks the name of induction variables partial names.
    induction_vars: HashMap<ir::IndVarId, String>,
    induction_levels: HashMap<(ir::IndVarId, ir::dim::Id), String>,
    /// Casted sizes.
    size_casts: HashMap<(&'a codegen::Size<'a>, ir::Type), String>,
    /// Guard to use in front of instructions with side effects.
    side_effect_guard: Option<RcStr>,
}

impl<'a, 'b> NameMap<'a, 'b> {
    /// Creates a new `NameMap`.
    pub fn new(function: &'a Function<'a>, namer: &'b mut Namer) -> Self {
        let mut mem_blocks = HashMap::default();
        // Setup parameters names.
        let params = function.device_code_args().map(|val| {
            let var_name = namer.name(val.t());
            let param_name = namer.name_param(val.key());
            if let ParamValKey::GlobalMem(id) = val.key() {
                mem_blocks.insert(id, var_name.clone());
            }
            (val.key(), (var_name, param_name))
        }).collect();
        // Name dimensions indexes.
        let mut indexes = HashMap::default();
        for dim in function.dimensions() {
            let name = RcStr::new(namer.name(Type::I(32)));
            for id in dim.dim_ids() { indexes.insert(id, name.clone()); }
        }
        // Name induction levels.
        let mut induction_levels = HashMap::default();
        let mut induction_vars = HashMap::default();
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

        let mut name_map = NameMap {
            namer: std::cell::RefCell::new(namer),
            insts: HashMap::default(),
            num_loop: 0,
            current_indexes: HashMap::default(),
            #[cfg(feature="mppa")]
            total_num_threads: function.num_threads(),
            size_casts: HashMap::default(),
            indexes, params, mem_blocks, induction_vars, induction_levels,
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
                },
                _ => panic!("inductions variables with more than two components must be
                            computed by the outermost induction level")
            };
        }
        // Setup the name of variables holding instruction results.
        for inst in function.cfg().instructions() {
            if let Some((inst_id, dim_map)) = inst.as_reduction() {
                name_map.decl_alias(inst, inst_id, dim_map);
            } else if inst.t().is_some() {
                name_map.decl_inst(inst);
            }
        }
        name_map
    }

    /// Returns the total number of threads.
    #[cfg(feature="mppa")]
    pub fn total_num_threads(&self) -> u32 { self.total_num_threads }

    /// Generates a variable of the given `Type`.
    pub fn gen_name(&self, t: Type) -> String { self.namer.borrow_mut().name(t) }

    /// Generates an ID for a loop.
    pub fn gen_loop_id(&mut self) -> u32 {
        let id = self.num_loop;
        self.num_loop += 1;
        id
    }

    /// Allocate a predicate name.
    #[cfg(feature = "cuda")]
    pub fn allocate_pred(&self) -> String { self.namer.borrow_mut().name(Type::I(1)) }

    pub fn name(&self, value: Value) -> Cow<str> {
        match value {
            Value::Operand(op) => self.name_op(op),
            Value::InductionLevel(ind_var, level) =>
                self.name_induction_var(ind_var, Some(level)),
        }
    }

    /// Asigns a name to an operand.
    pub fn name_op(&self, operand: &Operand) -> Cow<str> {
        match *operand {
            Operand::Int(ref val, len) =>
                Cow::Owned(self.namer.borrow().name_int(val, len)),
            Operand::Float(ref val, len) =>
                Cow::Owned(self.namer.borrow().name_float(val, len)),
            Operand::Inst(id, _, ref dim_map, _) |
            Operand::Reduce(id, _, ref dim_map, _) =>
                Cow::Borrowed(self.name_inst_id(id, dim_map)),
            Operand::Index(id) =>
                if let Some(idx) =  self.current_indexes.get(&id) {
                    Cow::Owned(format!("{}", idx))
                } else {
                    Cow::Borrowed(&self.indexes[&id])
                },
            Operand::Param(p) => self.name_param_val(ParamValKey::External(p)),
            Operand::Addr(id) => self.name_addr(id),
            Operand::InductionVar(id, _) => self.name_induction_var(id, None),
        }
    }

    /// Returns the name of the instruction.
    pub fn name_inst(&self, inst: &Instruction) -> &str {
        self.name_inst_id(inst.id(), &dim::Map::empty())
    }

    /// Returns the name of the instruction.
    pub fn name_inst_id(&self, id: InstId, dim_map: &DimMap) -> &str {
        let mut indexes = self.current_indexes.clone();
        for &(lhs, rhs) in dim_map.iter() {
            indexes.remove(&rhs).map(|idx| indexes.insert(lhs, idx));
        }
        let (ref dims, ref name_array) = self.insts[&id];
        let idx_vals = dims.iter().zip(name_array.dims.iter())
            .map(|(x, s)| indexes.get(x).map_or(s-1, |x| *x as usize));
        &name_array[&idx_vals.collect_vec()[..]]
    }

    /// Declares an instruction to the namer.
    fn decl_inst(&mut self, inst: &Instruction) {
        let (dim_ids, dim_sizes) = self.inst_name_dims(inst);
        let num_name = dim_sizes.iter().product();
        let names = (0 .. num_name).map(|_| {
            self.gen_name(unwrap!(inst.t()))
        }).collect_vec();
        let array = NDArray::new(dim_sizes, names);
        assert!(self.insts.insert(inst.id(), (dim_ids, array)).is_none());
    }

    /// Declares an instruction as an alias of another.
    fn decl_alias(&mut self, alias: &Instruction, base: InstId, dim_map: &DimMap) {
        let (dim_ids, dim_sizes) = self.inst_name_dims(alias);
        let names = {
            let (ref base_dims, ref base_names) = self.insts[&base];
            let permutation = {
                let mut base_pos: HashMap<_, _> = base_dims.iter().enumerate()
                    .map(|(i, x)| (*x, i)).collect();
                for &(lhs, rhs) in dim_map.iter() {
                    base_pos.remove(&lhs).map(|pos| base_pos.insert(rhs, pos));
                }
                dim_ids.iter().map(|dim| base_pos.get(dim)).enumerate()
                    .flat_map(|(src, dst)| dst.map(|x| (src, *x))).collect_vec()
            };
            let mut names = vec![];
            let mut base_indexes = base_names.dims.iter().map(|x| x-1).collect_vec();
            for nd_index in NDRange::new(&dim_sizes) {
                for &(src, dst) in &permutation {
                    base_indexes[dst] = nd_index[src];
                }
                names.push(base_names[&base_indexes[..]].clone());
            }
            names
        };
        let array = NDArray::new(dim_sizes, names);
        assert!(self.insts.insert(alias.id(), (dim_ids, array)).is_none());
    }

    /// Returns the ids and the sizes of the dimensions on which the instructions must be
    /// named.
    fn inst_name_dims(&self, inst: &Instruction) -> (Vec<dim::Id>, Vec<usize>) {
        inst.instantiation_dims().iter().map(|&(dim, size)| (dim, size as usize)).unzip()
    }

    /// Returns the name of an index.
    pub fn name_index(&self, dim_id: dim::Id) -> &str { &self.indexes[&dim_id] }

    /// Set the current index of an unrolled dimension.
    pub fn set_current_index(&mut self, dim: &Dimension, idx: u32) {
        for id in dim.dim_ids() { self.current_indexes.insert(id, idx); }
    }

    /// Unset the current index of an unrolled dimension.
    pub fn unset_current_index(&mut self, dim: &Dimension) {
        for id in dim.dim_ids() { assert!(self.current_indexes.remove(&id).is_some()); }
    }

    pub fn indexed_inst_name(&mut self, inst: &Instruction,
                             dim: ir::dim::Id, idx: u32) -> String {
        self.current_indexes.insert(dim, idx);
        let name = self.name_inst(inst).to_string();
        self.current_indexes.remove(&dim);
        name
    }

    pub fn indexed_op_name(&mut self, op: &Operand,
                           dim: ir::dim::Id, idx: u32) -> String {
        self.current_indexes.insert(dim, idx);
        let name = self.name_op(op).to_string();
        self.current_indexes.remove(&dim);
        name
    }

    /// Returns the name of a variable representing a parameter.
    pub fn name_param(&self, param: ParamValKey) -> Cow<str> {
        let param = unsafe { std::mem::transmute(param) };
        Cow::Borrowed(&self.params[&param].1)
    }

    /// Returns the name of a variable representing a parameter value.
    pub fn name_param_val(&self, param: ParamValKey) -> Cow<str> {
        let param = unsafe { std::mem::transmute(param) };
        Cow::Borrowed(&self.params[&param].0)
    }

    /// Returns the name of the address of a memory block.
    pub fn name_addr(&self, id: mem::InternalId) -> Cow<str> {
        Cow::Borrowed(&self.mem_blocks[&id])
    }

    /// Assigns a name to an induction variable.
    // TODO(cleanup): split into name induction var and name induction level
    pub fn name_induction_var(&self, var: ir::IndVarId, dim: Option<ir::dim::Id>)
            -> Cow<str> {
        if let Some(dim) = dim {
            Cow::Borrowed(&self.induction_levels[&(var, dim)])
        } else {
            Cow::Borrowed(&self.induction_vars[&var])
        }
    }

    /// Declares a size cast. Returns the name of the variable only if a new variable was
    /// allcoated.
    pub fn declare_size_cast(&mut self, size: &'a codegen::Size<'a>, t: ir::Type)
        -> Option<String>
    {
        if size.dividend().is_empty() || t == Type::I(32) { return None; }
        match self.size_casts.entry((size, t)) {
            hash_map::Entry::Occupied(..) => None,
            hash_map::Entry::Vacant(entry) =>
                Some(entry.insert(self.namer.borrow_mut().name(t)).to_string()),
        }
    }

    /// Assigns a name of a value to a size.
    pub fn name_size(&self, size: &codegen::Size, expected_t: ir::Type) -> Cow<str> {
        let size: &'a codegen::Size<'a> = unsafe { std::mem::transmute(size) };
        match (size.dividend(), expected_t) {
            (&[], _) => {
                assert_eq!(size.divisor(), 1);
                Cow::Owned(size.factor().to_string())
            },
            (&[p], Type::I(32)) if size.factor() == 1 && size.divisor() == 1 =>
                self.name_param_val(ParamValKey::External(p)),
            (_, Type::I(32)) => self.name_param_val(ParamValKey::Size(size)),
            _ => {
                let size = unwrap!(self.size_casts.get(&(size, expected_t)));
                Cow::Borrowed(size)
            },
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
