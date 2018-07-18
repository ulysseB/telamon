//! Handle operands invariants.
use ir::{self, BasicBlock, DimMapScope};
use ir::Operand::*;
use search_space::choices::{Action, DimKind, DimMapping, Order};

/// Generates actions to enforce operands invariants.
pub fn invariants(fun: &ir::Function, op: &ir::Operand, user: ir::BBId) -> Vec<Action> {
    match *op {
        Int(..) | Float(..) | Param(..) | Addr(..) => vec![],
        Inst(src, _, ref dim_map, scope) => {
            // Order dimensions in the dim map.
            let order = Order::BEFORE | Order::MERGED;
            let mut actions = Vec::new();
            for &(lhs, rhs) in dim_map.iter() {
                actions.push(Action::Order(lhs.into(), rhs.into(), order));
                let mapping = match scope {
                    DimMapScope::Local => DimMapping::UNROLL_MAP,
                    DimMapScope::Thread => DimMapping::MAPPED,
                    DimMapScope::Global => DimMapping::ALL,
                };
                actions.push(Action::DimMapping(lhs, rhs, mapping));
                // FIXME: allow tmp mem with dynamic size when the scope is global.
                if fun.dim(lhs).possible_sizes().is_none() {
                    actions.push(Action::Order(lhs.into(), rhs.into(), Order::MERGED));
                }
            }
            // Order the with the source instruction.
            actions.push(Action::Order(src.into(), user, Order::BEFORE));
            actions
        },
        Reduce(src, _, ref dim_map, ref reduce_dims) => {
            let order = Order::BEFORE | Order::MERGED;
            let mut actions = Vec::new();
            // TODO(search_space): allow tmp mem to be generated for reduce dim maps.
            for &(lhs, rhs) in dim_map.iter() {
                actions.push(Action::Order(lhs.into(), rhs.into(), order));
                actions.push(Action::DimMapping(lhs, rhs, DimMapping::MAPPED));
            };
            actions.push(Action::Order(src.into(), user, Order::BEFORE));
            for &dim in reduce_dims {
                actions.push(Action::Order(src.into(), dim.into(), Order::BEFORE));
                actions.push(Action::DimKind(dim, DimKind::LOOP | DimKind::UNROLL));
            }
            actions
        },
        Index(dim) => vec![Action::Order(dim.into(), user, Order::OUTER)],
        InductionVar(var_id, _) => {
            let var = fun.induction_var(var_id);
            let mut actions = invariants(fun, var.base(), user);
            for &(dim, _) in var.dims().iter() {
                actions.extend(invariants(fun, var.base(), dim.into()));
                actions.push(Action::Order(dim.into(), user, Order::OUTER));
            }
            actions
        },
    }
}

/// Generates the invariants of the operands of an instuction.
pub fn inst_invariants(fun: &ir::Function, inst: &ir::Instruction) -> Vec<Action> {
    inst.operands().into_iter().flat_map(move |op| {
        invariants(fun, op, inst.bb_id())
    }).collect()
}
