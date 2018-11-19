//! Handle operands invariants.
use ir::Operand::*;
use ir::{self, Statement};
use search_space::choices::{Action, Order};

/// Generates actions to enforce operands invariants.
pub fn invariants(fun: &ir::Function, op: &ir::Operand, user: ir::StmtId) -> Vec<Action> {
    match *op {
        Int(..) | Float(..) | Param(..) | Addr(..) | Variable(..) => vec![],
        Index(dim) => vec![Action::Order(dim.into(), user, Order::OUTER)],
        InductionVar(var_id, _) => {
            let var = fun.induction_var(var_id);
            let mut actions = invariants(fun, var.base(), user);
            for &(dim, _) in var.dims().iter() {
                actions.extend(invariants(fun, var.base(), dim.into()));
                actions.push(Action::Order(dim.into(), user, Order::OUTER));
            }
            actions
        }
    }
}

/// Generates the invariants of the operands of an instuction.
pub fn inst_invariants(fun: &ir::Function, inst: &ir::Instruction) -> Vec<Action> {
    inst.operands()
        .into_iter()
        .flat_map(move |op| invariants(fun, op, inst.stmt_id()))
        .collect()
}
