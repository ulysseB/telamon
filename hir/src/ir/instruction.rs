//! Describes the instructions.
use crate::ir::{
    self, DimMapScope, LoweringMap, Operand, Operator, Statement, StmtId, Type,
};
use serde::{Deserialize, Serialize};
use std::{self, fmt};

use fxhash::FxHashSet;
use itertools::Itertools;
use utils::*;

/// Uniquely identifies an instruction.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct InstId(pub u32);

impl fmt::Debug for InstId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "@{}", self.0)
    }
}

impl From<InstId> for usize {
    fn from(id: InstId) -> usize {
        id.0 as usize
    }
}

impl std::fmt::Display for InstId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "instruction {}", self.0)
    }
}

/// Represents an instruction.
#[derive(Clone, Debug)]
pub struct Instruction<L = LoweringMap> {
    operator: Operator<L>,
    id: InstId,
    iter_dims: FxHashSet<ir::DimId>,
    variable: Option<ir::VarId>,
    defined_vars: VecSet<ir::VarId>,
    used_vars: VecSet<ir::VarId>,
    dependencies: VecSet<ir::InstId>,
    users: VecSet<ir::InstId>,
}

impl<L> Instruction<L> {
    /// Creates a new instruction and type-check the operands.
    pub fn new(
        operator: Operator<L>,
        id: InstId,
        iter_dims: FxHashSet<ir::DimId>,
        fun: &ir::Function<L>,
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
            })
            .collect();

        // Collect dependencies
        let dependencies = operator
            .operands()
            .iter()
            .flat_map(|op| match op {
                ir::Operand::Inst(src, ..) | ir::Operand::Reduce(src, ..) => Some((*src)),
                _ => None,
            })
            .collect();

        Ok(Instruction {
            operator,
            id,
            iter_dims,
            variable: None,
            defined_vars: VecSet::default(),
            used_vars,
            dependencies,
            users: VecSet::default(),
        })
    }

    /// Returns an iterator over the operands of this instruction.
    pub fn operands(&self) -> Vec<&Operand<L>> {
        self.operator.operands()
    }

    /// Iterate over the dimensions that appear in `Reduce` operands for this instruction.
    pub fn iter_reduced_dims<'a>(&'a self) -> impl Iterator<Item = ir::DimId> + 'a {
        self.operator.operands().into_iter().flat_map(|operand| {
            match operand {
                Operand::Reduce(_, _, _, reduce_dims) => {
                    Some(reduce_dims.iter().cloned())
                }
                _ => None,
            }
            .into_iter()
            .flatten()
        })
    }

    /// Returns the type of the variable produced by an instruction.
    pub fn t(&self) -> Option<Type> {
        self.operator.t()
    }

    /// Returns the operator of the instruction.
    pub fn operator(&self) -> &Operator<L> {
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

    /// Applies the lowering of a layout to the instruction.
    pub fn lower_layout(
        &mut self,
        ld_idx: Operand<L>,
        ld_pattern: ir::AccessPattern,
        st_idx: Operand<L>,
        st_pattern: ir::AccessPattern,
    ) where
        L: Clone,
    {
        self.operator = match self.operator.clone() {
            Operator::TmpLd(t, id2) => {
                assert_eq!(ld_pattern.mem_block(), Some(id2));
                Operator::Ld(t, ld_idx, ld_pattern)
            }
            Operator::TmpSt(val, id2) => {
                assert_eq!(st_pattern.mem_block(), Some(id2));
                Operator::St(st_idx, val, false, st_pattern)
            }
            _ => panic!("Only TmpLd/TmpSt are changed on a layout lowering"),
        };
    }

    /// Indicates the operands for wich a `DimMap` must be lowered if lhs and rhs are
    /// not mapped.
    pub fn dim_maps_to_lower(&self, lhs: ir::DimId, rhs: ir::DimId) -> Vec<usize> {
        self.operator()
            .operands()
            .iter()
            .enumerate()
            .filter(|&(_, op)| op.should_lower_map(lhs, rhs))
            .map(|(id, _)| id)
            .collect()
    }

    /// Returns 'self' if it is a memory instruction.
    pub fn as_mem_inst(&self) -> Option<&Instruction<L>> {
        if self.operator.is_mem_access() {
            Some(self)
        } else {
            None
        }
    }

    /// Indicates if the instruction performs a reduction.
    pub fn as_reduction(&self) -> Option<(InstId, &ir::DimMap, &[ir::DimId])> {
        at_most_one(self.operands().iter().flat_map(|x| x.as_reduction()))
    }

    /// Returns 'true' if `self` is a reduction initialized by init, and if 'dim' should
    /// have the same nesting with 'init' that with 'self'.
    pub fn is_reduction_common_dim(&self, init: InstId, dim: ir::DimId) -> bool {
        self.as_reduction()
            .map(|(i, map, rd)| {
                i == init && !rd.contains(&dim) && map.iter().all(|&(_, rhs)| dim != rhs)
            })
            .unwrap_or(false)
    }

    /// Rename a dimension to another ID.
    pub fn merge_dims(&mut self, lhs: ir::DimId, rhs: ir::DimId) {
        self.operator.merge_dims(lhs, rhs);
    }

    /// The list of dimensions the instruction must be nested in.
    pub fn iteration_dims(&self) -> &FxHashSet<ir::DimId> {
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

    pub(super) fn register_user(&mut self, user: ir::InstId) {
        self.users.insert(user);
    }

    pub(super) fn register_dependency(&mut self, dependency: ir::InstId) {
        self.dependencies.insert(dependency);
    }

    pub(crate) fn dependencies(&self) -> &VecSet<ir::InstId> {
        &self.dependencies
    }

    pub(crate) fn users(&self) -> &VecSet<ir::InstId> {
        &self.users
    }
}

impl Instruction<()> {
    pub fn freeze(self, cnt: &mut ir::Counter) -> Instruction {
        Instruction {
            operator: self.operator.freeze(cnt),
            id: self.id,
            iter_dims: self.iter_dims,
            variable: self.variable,
            used_vars: self.used_vars,
            defined_vars: self.defined_vars,
            dependencies: self.dependencies,
            users: self.users,
        }
    }
}

impl Instruction {
    /// Lowers the `DimMap` of an operand into an access to a temporary memory.
    pub fn lower_dim_map(
        &mut self,
        op_id: usize,
        new_src: InstId,
        new_dim_map: ir::DimMap,
    ) {
        let operand = &mut *self.operator.operands_mut()[op_id];
        match *operand {
            Operand::Inst(ref mut src, _, ref mut dim_map, ref mut can_lower) => {
                {
                    let stmt_id = ir::StmtId::from(*src);
                    self.dependencies.insert(new_src);
                }

                *src = new_src;
                *dim_map = new_dim_map;
                *can_lower = DimMapScope::Local;
            }
            _ => panic!(),
        }
    }
}

impl<L> Statement<L> for Instruction<L> {
    fn stmt_id(&self) -> StmtId {
        self.id.into()
    }

    fn defined_vars(&self) -> &VecSet<ir::VarId> {
        &self.defined_vars
    }

    fn as_inst(&self) -> Option<&Instruction<L>> {
        Some(self)
    }

    fn used_vars(&self) -> &VecSet<ir::VarId> {
        &self.used_vars
    }

    fn register_defined_var(&mut self, var: ir::VarId) {
        self.defined_vars.insert(var);
    }
}

impl<L> fmt::Display for Instruction<L> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}:  {}", self.id, self.operator)
    }
}

impl<L> ir::IrDisplay<L> for Instruction<L> {
    fn fmt(&self, fmt: &mut fmt::Formatter, function: &ir::Function<L>) -> fmt::Result {
        write!(
            fmt,
            "{:?}[{}]: {} [deps: {:?}, users: {:?}]",
            self.id,
            self.iteration_dims().iter().sorted().format(", "),
            self.operator.display(function),
            self.dependencies(),
            self.users(),
        )
    }
}
