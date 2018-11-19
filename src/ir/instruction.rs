//! Describes the instructions.
use ir::{self, Operand, Operator, Statement, StmtId, Type};
use std;
use utils::*;

/// Uniquely identifies an instruction.
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(C)]
/// cbindgen:field-names=[id]
pub struct InstId(pub u32);

impl Into<usize> for InstId {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for InstId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "instruction {}", self.0)
    }
}

/// Represents an instruction.
#[derive(Clone, Debug)]
pub struct Instruction<'a> {
    operator: Operator<'a>,
    id: InstId,
    iter_dims: HashSet<ir::DimId>,
    variable: Option<ir::VarId>,
    defined_vars: VecSet<ir::VarId>,
    used_vars: VecSet<ir::VarId>,
    mem_access_layout: VecSet<ir::LayoutDimId>,
}

impl<'a> Instruction<'a> {
    /// Creates a new instruction and type-check the operands.
    pub fn new<L>(
        operator: Operator<'a>,
        id: InstId,
        iter_dims: HashSet<ir::DimId>,
        mem_access_layout: VecSet<ir::LayoutDimId>,
        fun: &mut ir::Function<L>,
    ) -> Result<Self, ir::Error> {
        operator.check(&iter_dims, fun)?;
        for operand in operator.operands() {
            if let ir::Operand::Variable(var_id, ..) = *operand {
                for &dim in fun.variable(var_id).dimensions() {
                    if !iter_dims.contains(&dim) {
                        Err(ir::Error::MissingIterationDim { dim })?;
                    }
                }
            }
        }
        let used_vars = operator
            .operands()
            .iter()
            .flat_map(|op| {
                if let ir::Operand::Variable(v, ..) = op {
                    Some(*v)
                } else {
                    None
                }
            }).chain(operator.loaded_mem_var())
            .collect();
        let defined_vars = operator.stored_mem_var().into_iter().collect();
        // Registers `self` in the corresponding `DmaStart` if applicable.
        if let ir::op::DmaWait { dma_start, .. } = operator {
            let start_op = &mut fun.inst_mut(dma_start).operator;
            if let ir::op::DmaStart { dma_wait, .. } = start_op {
                *dma_wait = Some(id);
            } else {
                panic!("expected a DMA start operator");
            };
        }
        Ok(Instruction {
            operator,
            id,
            iter_dims,
            variable: None,
            defined_vars,
            used_vars,
            mem_access_layout,
        })
    }

    /// Returns an iterator over the operands of this instruction.
    pub fn operands(&self) -> Vec<&Operand<'a>> {
        self.operator.operands()
    }

    /// Returns the type of the variable produced by an instruction.
    pub fn t(&self) -> Option<Type> {
        self.operator.t()
    }

    /// Returns the operator of the instruction.
    pub fn operator(&self) -> &Operator<'_> {
        &self.operator
    }

    /// Returns the `InstId` representing the instruction.
    pub fn id(&self) -> InstId {
        self.id
    }

    /// Returns true if the instruction has side effects.
    pub fn has_side_effects(&self) -> bool {
        self.operator.has_side_effects()
    }

    /// Returns 'self' if it is a memory instruction.
    pub fn as_mem_inst(&self) -> Option<&Instruction<'_>> {
        if self.operator.is_mem_access() {
            Some(self)
        } else {
            None
        }
    }

    /// The list of dimensions the instruction must be nested in.
    pub fn iteration_dims(&self) -> &HashSet<ir::DimId> {
        &self.iter_dims
    }

    /// Adds a new iteration dimension. Indicates if the dimension was not already an
    /// iteration dimension.
    pub fn add_iteration_dimension(&mut self, dim: ir::DimId) -> bool {
        self.iter_dims.insert(dim)
    }

    /// Returns the `Variable` holding the result of this instruction.
    pub fn result_variable(&self) -> Option<ir::VarId> {
        self.variable
    }

    /// Sets the `Variable` holdings the result of this instruction.
    pub fn set_result_variable(&mut self, variable: ir::VarId) {
        // An instruction variable cannot be set twice.
        assert_eq!(std::mem::replace(&mut self.variable, Some(variable)), None);
    }

    /// Indicates the layout of the accessed memory, if any.
    pub fn mem_access_layout(&self) -> &VecSet<ir::LayoutDimId> {
        &self.mem_access_layout
    }
}

impl<'a> Instruction<'a> {
    /// Retargets operands referencing `old` to use `new` instead. Also adds `new` to the
    /// set of variables used.
    pub fn rename_var(&mut self, old: ir::VarId, new: ir::VarId) {
        self.used_vars.insert(new);
        for operand in self.operator.operands_mut() {
            match operand {
                Operand::Variable(var, ..) if *var == old => *var = new,
                _ => (),
            }
        }
    }
}

impl<'a> Statement<'a> for Instruction<'a> {
    fn stmt_id(&self) -> StmtId {
        self.id.into()
    }

    fn defined_vars(&self) -> &VecSet<ir::VarId> {
        &self.defined_vars
    }

    fn as_inst(&self) -> Option<&Instruction<'a>> {
        Some(self)
    }

    fn used_vars(&self) -> &VecSet<ir::VarId> {
        &self.used_vars
    }

    fn register_defined_var(&mut self, var: ir::VarId) {
        self.defined_vars.insert(var);
    }

    fn register_used_var(&mut self, var: ir::VarId) {
        self.used_vars.insert(var);
    }
}
