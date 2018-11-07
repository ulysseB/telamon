//! Encodes the data-flow information.
use ir;
use utils::*;

/// Uniquely identifies variables.
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize,
)]
#[repr(C)]
/// cbindgen:field-names=[id]
pub struct VarId(pub u16);

impl From<VarId> for usize {
    fn from(val_id: VarId) -> Self {
        val_id.0 as usize
    }
}

/// A variable produced by the code.
#[derive(Clone, Debug)]
pub struct Variable {
    id: VarId,
    t: ir::Type,
    def: VarDef,
    memory_level: MemoryLevel,
    dimensions: VecSet<ir::DimId>,
    def_points: VecSet<ir::StmtId>,
    use_points: VecSet<ir::StmtId>,
    predecessors: VecSet<ir::VarId>,
    successors: VecSet<ir::VarId>,
    consumer: Consumer,
}

/// Indicates the slowest memory level where a variable may be stored.
///
/// This is usefull to limit the size of the search space by removing useless decisions.
/// For example, we don't want to store in memory the operand of a store. Also, we don't
/// want to store in RAM a value we just loaded from RAM.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub enum MemoryLevel {
    /// The variable must be stored in registers and the producer and consumer must not be
    /// separated by synchronisations.
    RegisterNoSync,
    /// The variable must be stored in registers.
    Register,
    /// The variable must be stored in registers or a local, fast, memory.
    FastMemory,
    /// The variable may be stored anywhere.
    SlowMemory,
}

impl Variable {
    /// Creates a new variable with the given Id.
    pub fn new<L>(id: VarId, def: VarDef, fun: &ir::Function<L>) -> Self {
        let t = def.t(fun);
        let def_points = def.def_points(fun);
        let dimensions = def.dimensions(fun);
        let predecessors = def.predecessors();
        Variable {
            id,
            t,
            def,
            // TODO(ulysse): allow lowering to memory.
            memory_level: MemoryLevel::Register,
            dimensions,
            def_points,
            use_points: Default::default(),
            predecessors,
            successors: Default::default(),
            consumer: Consumer::None,
        }
    }

    /// Return the unique identifiers of the `Variable`.
    pub fn id(&self) -> VarId {
        self.id
    }

    /// Specifies how the variable is defined.
    pub fn def(&self) -> &VarDef {
        &self.def
    }

    /// Indicates the type of the variable.
    pub fn t(&self) -> ir::Type {
        self.t
    }

    /// Indicates the statements that define the variable.
    pub fn def_points(&self) -> impl Iterator<Item = ir::StmtId> + '_ {
        self.def_points.iter().cloned()
    }

    /// Indicates the statements that uses the variable.
    pub fn use_points(&self) -> impl Iterator<Item = ir::StmtId> + '_ {
        self.use_points.iter().cloned()
    }

    /// Returns the dimensions along which the variable can vary.
    pub fn dimensions(&self) -> &VecSet<ir::DimId> {
        &self.dimensions
    }

    /// Registers that the variable is used by a statement.
    pub fn add_use(&mut self, stmt: ir::StmtId) {
        self.use_points.insert(stmt);
    }

    /// Registers the variable in the structures it references in the function.
    pub fn register<L>(&self, fun: &mut ir::Function<L>) {
        // If the variable is not fully built, register will be called again later.
        if !self.def().is_complete() {
            return;
        }
        for &def_point in &self.def_points {
            fun.statement_mut(def_point).register_defined_var(self.id());
        }
        for &dim in &self.dimensions {
            fun.dim_mut(dim).register_inner_var(self.id());
        }
        for mapping in self.def.mapped_dims() {
            fun.dim_mapping_mut(mapping).register_user(self.id());
        }
        match self.def() {
            VarDef::Inst(inst_id) => {
                fun.inst_mut(*inst_id).set_result_variable(self.id());
            }
            VarDef::Fby { init, prev, dims } => {
                for &dim in dims {
                    fun.dim_mut(dim).set_sequential();
                    fun.register_var_use(*init, Into::<ir::StmtId>::into(dim).into());
                }
                set_consumer(*init, Consumer::FbyInit, fun);
                set_consumer(unwrap!(*prev), Consumer::FbyPrev, fun);
            }
            VarDef::Last(_, dims) => {
                for &dim in dims {
                    fun.dim_mut(dim).set_sequential();
                }
            }
            _ => (),
        }
        for &predecessor in &self.predecessors {
            fun.variable_mut(predecessor).add_successor(self.id);
        }
    }

    /// Indicates where the variable can be stored.
    pub fn max_memory_level(&self) -> MemoryLevel {
        self.memory_level
    }

    /// Indicates the variables this variable directly depends on.
    pub fn predecessors(&self) -> &VecSet<ir::VarId> {
        &self.predecessors
    }

    /// Indicates the variables that directly depend on this one.
    pub fn successors(&self) -> &VecSet<ir::VarId> {
        &self.successors
    }

    /// Indicates the variable is consumed to create another. A consumed variable can only
    /// have a single successor.
    fn set_consumer(&mut self, consumer: Consumer) {
        assert_eq!(self.consumer, Consumer::None);
        if consumer == Consumer::FbyInit {
            assert!(self.successors.len() <= 1);
        }
        self.consumer = consumer;
    }

    /// Registers a successor of the variable.
    fn add_successor(&mut self, successor: ir::VarId) {
        self.successors.insert(successor);
        if self.successors.len() > 1 {
            assert!(self.consumer != Consumer::FbyInit);
        }
    }

    /// Sets the `prev` field of a `Fby` variable.
    pub fn set_loop_carried_variable(&mut self, loop_carried_var: VarId) {
        if let VarDef::Fby { ref mut prev, .. } = self.def {
            assert!(prev.is_none());
            *prev = Some(loop_carried_var);
        } else {
            panic!("set_loop_carried_variable is only valid on Fby variables");
        }
        self.predecessors.insert(loop_carried_var);
    }
}

/// Specifies how is a `Variable` defined.
#[derive(Clone, Debug)]
// TODO(value): ExternalMem
pub enum VarDef {
    /// Takes the variable produced by an instruction.
    Inst(ir::InstId),
    /// Takes point-to-point the values of a variable produced in another loop nest.
    DimMap(ir::VarId, VecSet<ir::DimMappingId>),
    /// Takes the last value of a variable in a loop nest.
    Last(ir::VarId, VecSet<ir::DimId>),
    /// Takes the value of `init` at the first iteration of `dims` and the values of
    /// `prev` afterward.
    ///
    /// The name `fby` comes from synchronous dataflow languages and means `followed by`.
    /// We currently dissallow `Fby` to be used in any other variable as it would make it
    /// harder to find where it is used and enforce constraints.
    ///
    /// `Fby` should never be instantiated on `dims`. It is currently the job of the user
    /// enforce that, by not mapping these dimensions with `DimMap`. Also, `init` cannot
    /// be used by any other variable and `prev` in any other `Fby`. This ensures we can
    /// implement the variable in-place, without any move or copies. We may later relax
    /// this constraint.
    Fby {
        init: ir::VarId,
        /// `prev` is an option to allow creating cyclic depdencies accross loop
        /// iterations. When exploring the search space, prev must be set.
        prev: Option<ir::VarId>,
        dims: VecSet<ir::DimId>,
    },
}

impl VarDef {
    /// Returns the type of the variable if used on the context of `function`.
    pub fn t<L>(&self, fun: &ir::Function<L>) -> ir::Type {
        match self {
            VarDef::Inst(inst_id) => unwrap!(fun.inst(*inst_id).t()),
            // A variable can't depend on itself so this doesn't loop.
            VarDef::DimMap(var_id, ..)
            | VarDef::Last(var_id, ..)
            | VarDef::Fby { init: var_id, .. } => fun.variable(*var_id).t(),
        }
    }

    /// Ensures the definition is valid.
    pub fn check<L>(&self, fun: &ir::Function<L>) -> Result<(), ir::TypeError> {
        match self {
            VarDef::Inst(inst) if fun.inst(*inst).t().is_none() => {
                Err(ir::TypeError::ExpectedReturnType { inst: *inst })?;
            }
            VarDef::Fby { init, prev, .. } => {
                if let Some(prev) = *prev {
                    let init = fun.variable(*init);
                    let prev = fun.variable(prev);
                    if init.t() != prev.t() {
                        ir::TypeError::check_equals(init.t(), prev.t())?;
                    }
                }
            }
            _ => (),
        }
        Ok(())
    }

    /// Indicates in which statment the variable is defined. Also returns the definition
    /// points of the variables it depends on.
    pub fn def_points<L>(&self, fun: &ir::Function<L>) -> VecSet<ir::StmtId> {
        match self {
            VarDef::Inst(inst_id) => VecSet::new(vec![(*inst_id).into()]),
            VarDef::DimMap(var_id, ..) | VarDef::Fby { init: var_id, .. } => {
                fun.variable(*var_id).def_points.clone()
            }
            VarDef::Last(var_id, dims) => {
                let dims = fun
                    .variable(*var_id)
                    .def_points
                    .iter()
                    .cloned()
                    .chain(dims.iter().map(|&id| id.into()))
                    .collect();
                VecSet::new(dims)
            }
        }
    }

    /// Returns the dimensions on which the variable can take different values.
    pub fn dimensions<L>(&self, fun: &ir::Function<L>) -> VecSet<ir::DimId> {
        match self {
            VarDef::Inst(inst_id) => VecSet::new(
                fun.inst(*inst_id)
                    .iteration_dims()
                    .iter()
                    .cloned()
                    .collect(),
            ),
            VarDef::Last(var_id, dims) => {
                let var = fun.variable(*var_id);
                VecSet::new(var.dimensions.difference(dims).cloned().collect())
            }
            VarDef::DimMap(var_id, mapping_ids) => {
                let mapping: HashMap<_, _> = mapping_ids
                    .iter()
                    .map(|&id| {
                        let dims = fun.dim_mapping(id).dims();
                        (dims[0], dims[1])
                    }).collect();
                let dims = fun
                    .variable(*var_id)
                    .dimensions()
                    .iter()
                    .map(|&dim| mapping.get(&dim).cloned().unwrap_or(dim));
                VecSet::new(dims.collect())
            }
            VarDef::Fby { init, dims, .. } => {
                let dims = fun
                    .variable(*init)
                    .dimensions()
                    .iter()
                    .chain(dims)
                    .cloned()
                    .collect();
                VecSet::new(dims)
            }
        }
    }

    /// Lists the point-to-point communications implied by this value.
    pub fn mapped_dims(&self) -> impl Iterator<Item = ir::DimMappingId> + '_ {
        match self {
            VarDef::DimMap(_, mappings) => mappings.iter().cloned(),
            _ => [].iter().cloned(),
        }
    }

    /// Returns the instruction that produce this value and the mapping from the
    /// dimensions to the value dimensions.
    pub fn production_inst<'a>(&'a self, fun: &'a ir::Function) -> ProductionPoint<'a> {
        match self {
            VarDef::Inst(inst) => {
                let dim_map = HashMap::default();
                ProductionPoint {
                    inst: *inst,
                    dim_map,
                    back_edges: vec![],
                }
            }
            VarDef::Last(prev, dims) => {
                let mut prod_point = fun.variable(*prev).def().production_inst(fun);
                for dim in dims {
                    prod_point.dim_map.remove(dim);
                }
                prod_point
            }
            VarDef::DimMap(prev, mapping_ids) => {
                let mut prod_point = fun.variable(*prev).def().production_inst(fun);
                for &mapping_id in mapping_ids {
                    let [src, dst] = fun.dim_mapping(mapping_id).dims();
                    prod_point.dim_map.insert(src, dst);
                }
                prod_point
            }
            VarDef::Fby { init, prev, dims } => {
                let mut prod_point = fun.variable(*init).def().production_inst(fun);
                prod_point.back_edges.push((unwrap!(*prev), dims));
                prod_point
            }
        }
    }

    /// Indicates if the variable is a `Fby`.
    pub fn is_fby(&self) -> bool {
        if let VarDef::Fby { .. } = self {
            true
        } else {
            false
        }
    }

    /// Indicates if `self` is a `Fby` that takes the values of `expected_prev` at the
    /// previous iteration of some dimensions.
    pub fn is_fby_prev(&self, expected_prev: VarId) -> bool {
        if let VarDef::Fby { prev, .. } = self {
            unwrap!(*prev) == expected_prev
        } else {
            false
        }
    }

    /// Indicates the variables `self` directly depends on.
    pub fn predecessors(&self) -> VecSet<ir::VarId> {
        match self {
            VarDef::Inst { .. } => VecSet::default(),
            VarDef::Last(pred, ..)
            | VarDef::DimMap(pred, ..)
            | VarDef::Fby {
                init: pred,
                prev: None,
                ..
            } => VecSet::new(vec![*pred]),
            VarDef::Fby {
                init,
                prev: Some(prev),
                ..
            } => VecSet::new(vec![*init, *prev]),
        }
    }

    /// Returns the variable `self` is defined from. This is equivalent to `predecessors`
    /// but without loop-carried variables.
    pub fn origin(&self) -> Option<ir::VarId> {
        match self {
            VarDef::Inst { .. } => None,
            VarDef::Last(origin, ..)
            | VarDef::DimMap(origin, ..)
            | VarDef::Fby { init: origin, .. } => Some(*origin),
        }
    }

    /// Indicates if the variable is fully built.
    pub fn is_complete(&self) -> bool {
        if let VarDef::Fby { prev, .. } = self {
            prev.is_some()
        } else {
            true
        }
    }
}

/// Indicates how a value is produced.
pub struct ProductionPoint<'a> {
    /// Instruction that produce the variable.
    pub inst: ir::InstId,
    /// Comunication pattern between the production and consumption point.
    pub dim_map: HashMap<ir::DimId, ir::DimId>,
    /// Indicates a loop-carried dependency to another variable.
    pub back_edges: Vec<(ir::VarId, &'a [ir::DimId])>,
}

/// Indicates if a variable is used and overwritten by another.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Consumer {
    /// The variable is not overwritten by another.
    None,
    /// The variable is used to initialize a loop-carried variable.
    FbyInit,
    /// The variable is a loop-carried variable and is consumed by the next step of the
    /// loop nest.
    FbyPrev,
}

/// Sets the consumer in a variable and its predecessors.
fn set_consumer<L>(var: VarId, consumer: Consumer, fun: &mut ir::Function<L>) {
    let predecessors = {
        let var = fun.variable_mut(var);
        var.set_consumer(consumer);
        var.predecessors().clone()
    };
    for pred in predecessors {
        set_consumer(pred, consumer, fun);
    }
}
