use super::context::CheckerContext;
use super::error::TypeError;
use super::typing_context::TypingContext;
use super::{ir, Condition, TypedConstraint, VarDef, VarMap};

use itertools::Itertools;

/// A constraint that must be enforced by the IR.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Variables for which the conditions must be respected.
    pub forall_vars: Vec<VarDef>,
    /// Conjunction of disjuction of condition that must be respected.
    pub disjunctions: Vec<Vec<Condition>>,
    /// Indicates if the constraint should restrict fragile values.
    pub restrict_fragile: bool,
}

impl Constraint {
    /// Creates a new constraint.
    pub fn new(forall_vars: Vec<VarDef>, disjunctions: Vec<Vec<Condition>>) -> Self {
        Constraint {
            forall_vars,
            disjunctions,
            restrict_fragile: true,
        }
    }

    /// Type check the constraint.
    pub fn type_check(self, ir_desc: &ir::IrDesc) -> Vec<TypedConstraint> {
        let mut var_map = VarMap::default();
        let sets = self
            .forall_vars
            .into_iter()
            .map(|v| var_map.decl_forall(ir_desc, v))
            .collect_vec();
        let restrict_fragile = self.restrict_fragile;
        self.disjunctions
            .into_iter()
            .map(|disjuction| {
                let mut inputs = Vec::new();
                let conditions = disjuction
                    .into_iter()
                    .map(|x| x.type_check(ir_desc, &var_map, &mut inputs))
                    .collect();
                TypedConstraint {
                    vars: sets.clone(),
                    restrict_fragile,
                    inputs,
                    conditions,
                }
            })
            .collect_vec()
    }

    /// Type checks the define's condition.
    pub fn define(
        self,
        context: &CheckerContext,
        tc: &mut TypingContext,
    ) -> Result<(), TypeError> {
        tc.constraints.push(self);
        Ok(())
    }
}
