//! Encodes the data-flow information.
use std::fmt;

use crate::ir;

use serde::{Deserialize, Serialize};
use utils::*;

/// Uniquely identifies variables.
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct VarId(pub u16);

impl From<VarId> for usize {
    fn from(val_id: VarId) -> Self {
        val_id.0 as usize
    }
}

impl fmt::Display for VarId {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "${}", self.0)
    }
}

/// A variable produced by the code.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Variable {
    id: VarId,
    t: ir::Type,
    def: VarDef,
    memory_level: MemoryLevel,
    dimensions: VecSet<ir::DimId>,
    def_points: VecSet<ir::StmtId>,
    use_points: VecSet<ir::StmtId>,
}

/// Indicates the slowest memory level where a variable may be stored.
///
/// This is usefull to limit the size of the search space by removing useless decisions.
/// For example, we don't want to store in memory the operand of a store. Also, we don't
/// want to store in RAM a value we just loaded from RAM.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Serialize, Deserialize)]
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
        Variable {
            id,
            t,
            def,
            // TODO(ulysse): allow lowering to memory.
            memory_level: MemoryLevel::Register,
            dimensions,
            def_points,
            use_points: Default::default(),
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
        if let VarDef::Inst(inst_id) = self.def {
            fun.inst_mut(inst_id).set_result_variable(self.id());
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
    }

    /// Indicates where the variable can be stored.
    pub fn max_memory_level(&self) -> MemoryLevel {
        self.memory_level
    }
}

/// Specifies how is a `Variable` defined.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VarDef {
    /// Takes the variable produced by an instruction.
    Inst(ir::InstId),
    /// Takes point-to-point the values of a variable produced in another loop nest.
    DimMap(ir::VarId, VecSet<ir::DimMappingId>),
    /// Takes the last value of a variable in a loop nest.
    Last(ir::VarId, VecSet<ir::DimId>),
    // TODO(value): Fby and ExternalMem
}

impl VarDef {
    /// Returns the type of the variable if used on the context of `function`.
    pub fn t<L>(&self, fun: &ir::Function<L>) -> ir::Type {
        match self {
            VarDef::Inst(inst_id) => unwrap!(fun.inst(*inst_id).t()),
            VarDef::DimMap(var_id, ..) | VarDef::Last(var_id, ..) => {
                // A variable can't depend on itself so this doesn't loop.
                fun.variable(*var_id).t()
            }
        }
    }

    /// Ensures the definition is valid.
    pub fn check<L>(&self, fun: &ir::Function<L>) -> Result<(), ir::TypeError> {
        if let VarDef::Inst(inst) = self {
            if fun.inst(*inst).t().is_none() {
                Err(ir::TypeError::ExpectedReturnType { inst: *inst })?;
            }
        }
        Ok(())
    }

    /// Indicates in which statment the variable is defined.
    pub fn def_points<L>(&self, fun: &ir::Function<L>) -> VecSet<ir::StmtId> {
        match self {
            VarDef::Inst(inst_id) => VecSet::new(vec![(*inst_id).into()]),
            VarDef::DimMap(var_id, ..) => fun.variable(*var_id).def_points.clone(),
            VarDef::Last(_, dims) => {
                VecSet::new(dims.iter().map(|&id| id.into()).collect())
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
                let mapping: FnvHashMap<_, _> = mapping_ids
                    .iter()
                    .map(|&id| {
                        let dims = fun.dim_mapping(id).dims();
                        (dims[0], dims[1])
                    })
                    .collect();
                let dims = fun
                    .variable(*var_id)
                    .dimensions()
                    .iter()
                    .map(|dim| mapping[dim]);
                VecSet::new(dims.collect())
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
    pub fn production_inst(
        &self,
        fun: &ir::Function,
    ) -> (ir::InstId, FnvHashMap<ir::DimId, ir::DimId>) {
        match self {
            VarDef::Inst(inst) => (*inst, FnvHashMap::default()),
            VarDef::Last(prev, dims) => {
                let (inst, mut mapping) = fun.variable(*prev).def().production_inst(fun);
                for dim in dims {
                    mapping.remove(dim);
                }
                (inst, mapping)
            }
            VarDef::DimMap(prev, mapping_ids) => {
                let (inst, mut mapping) = fun.variable(*prev).def().production_inst(fun);
                for &mapping_id in mapping_ids {
                    let [src, dst] = fun.dim_mapping(mapping_id).dims();
                    mapping.insert(src, dst);
                }
                (inst, mapping)
            }
        }
    }
}
