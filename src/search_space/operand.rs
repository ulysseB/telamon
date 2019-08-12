//! Handle operands invariants.
use crate::ir::Operand::*;
use crate::ir::{self, DimMapScope, Statement};
use crate::search_space::choices::{Action, DimKind, DimMapping, Order};
use fxhash::FxHashSet;
use log::debug;

/// Adds an order action to `actions` for all reduction dimensions of
/// reductions that the instruction `inst_id` depends on, such that
/// the instruction is placed after the reduction dimensions. This
/// ensures that the reductions have finished before the instruction
/// executes.
///
/// Relevant reductions are those that are operands of instructions or
/// reductions whose results are used by `inst_id`. E.g., for the
/// following code,
///
///   @0[%1]: add(..., reduce(..., [%1]))
///   @1[%2]: mul(..., reduce(..., [%2]))
///
///   @2[%3]: add(@0[%3], @1[%3])
///
/// the dimensions `%1` and `%2` need to be placed before `@2`, since
/// `@2` uses the results of `@0` and `@1`, which are reduced over
/// `%1` and `%2`, respectively.
fn order_reduce_dims(fun: &ir::Function, inst_id: ir::InstId, actions: &mut Vec<Action>) {
    let mut red_dim_set: FxHashSet<_> = Default::default();
    let instr = fun.inst(inst_id);

    // Collect reduction dimensions of reductions that are operands of
    // instructions that this instruction depends on
    for operand in instr.operands() {
        match *operand {
            Inst(op_inst_id, ..) | Reduce(op_inst_id, ..) => {
                red_dim_set.extend(fun.inst(op_inst_id).iter_reduced_dims());
            }
            _ => {}
        }
    }

    // Order reduction dimensions before this instruction
    for &red_dim in red_dim_set.iter() {
        let action = Action::Order(red_dim.into(), instr.stmt_id(), Order::BEFORE);
        debug!(
            "Adding action ordering reduction dimension before the using instruction: {:?}",
            action
        );
        actions.push(action);
    }
}

/// Generates actions to enforce operands invariants.
pub fn invariants(fun: &ir::Function, op: &ir::Operand, user: ir::StmtId) -> Vec<Action> {
    match *op {
        Int(..) | Float(..) | Param(..) | Addr(..) | Variable(..) => vec![],
        Inst(src, _, ref dim_map, ref scope) => {
            // Order dimensions in the dim map.
            let order = Order::BEFORE | Order::MERGED;
            let mut actions = Vec::new();
            for &(lhs, rhs) in dim_map.iter() {
                actions.push(Action::Order(lhs.into(), rhs.into(), order));
                let mapping = match scope {
                    DimMapScope::Local => DimMapping::UNROLL_MAP,
                    DimMapScope::Thread => DimMapping::MAPPED,
                    DimMapScope::Global(..) => DimMapping::ALL,
                };
                actions.push(Action::DimMapping(lhs, rhs, mapping));
                // FIXME: allow tmp mem with dynamic size when the scope is global.
                if fun.dim(lhs).possible_sizes().is_none() {
                    actions.push(Action::Order(lhs.into(), rhs.into(), Order::MERGED));
                }
            }

            order_reduce_dims(fun, src, &mut actions);

            // Order the with the source instruction.
            actions.push(Action::Order(src.into(), user, Order::BEFORE));
            actions
        }
        Reduce(src, _, ref dim_map, ref reduce_dims) => {
            let order = Order::BEFORE | Order::MERGED;
            let mut actions = Vec::new();
            // TODO(search_space): allow tmp mem to be generated for reduce dim maps.
            for &(lhs, rhs) in dim_map.iter() {
                actions.push(Action::Order(lhs.into(), rhs.into(), order));
                actions.push(Action::DimMapping(lhs, rhs, DimMapping::MAPPED));
            }
            actions.push(Action::Order(src.into(), user, Order::BEFORE));
            for &dim in reduce_dims {
                actions.push(Action::Order(src.into(), dim.into(), Order::BEFORE));
                actions.push(Action::DimKind(dim, DimKind::LOOP | DimKind::UNROLL));
            }

            order_reduce_dims(fun, src, &mut actions);

            actions
        }
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
        ComputedAddress(ref access) => {
            use ir::IndexExpr;

            let mut actions = Vec::new();

            for index in access.indices() {
                let id = match *index {
                    IndexExpr::LogicalDim(id) => id,
                    IndexExpr::Unpack(id) => fun.packed_dims()[id].logical_dim(),
                    _ => continue,
                };

                for dim in fun.logical_dim(id).dimensions() {
                    actions.push(Action::Order(dim.into(), user, Order::OUTER));
                }
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
