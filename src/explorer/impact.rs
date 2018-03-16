//! Compute the maximal impact of a choice.
use ir;
use ir::dim::kind;
use std::collections::hash_map;
use utils::*;

/// Indicates if an `Action` can have a negative impact on the execution time.
pub fn has_impact(action: ir::Action, fun: &ir::Function) -> bool {
    let mut action_set = ActionSet::default();
    action_set.insert(&action);
    let mut queue = vec![action];
    while let Some(action) = queue.pop() {
        // TODO(impact) exit if too many actions are generated
        if has_indivudal_impact(&action, fun) { return true; }
        let implied_actions = match fun.implied_actions(&action) {
            Some(actions) => actions,
            None => return true,
        };
        for action in implied_actions {
            if action_set.insert(&action) {
                queue.push(action);
            }
        }
    }
    false
}

/// Indicates if an `Action` may have a negative impact on the exectuion time,
/// independently on the other `Action`s it may trigger.
fn has_indivudal_impact(action: &ir::Action, fun: &ir::Function) -> bool {
    match *action {
        ir::Action::Identity => false,
        ir::Action::Merge(..) =>
            // May reduce memory blocks size: positive.
            // May remove dimensions from `DimMap`s: neutral.
            // Other effects are taken into account by Order actions.
            false,
        ir::Action::LowerLayout(..) =>
            // Layouts are all eventually lowered: neutral.
            false,
        ir::Action::MemSpace(mem, space) =>
            // If the memory spaces are incompatible, the impact is infinite. Otherwise,
            // the impact is accounted for by instruction flags.
            (space & fun.mem_block(mem).space()).is_empty(),
        ir::Action::OuterThreadDim(..) =>
            // The effect is accounted for by order actions.
            false,
        ir::Action::InstFlag(inst, new_flag) => {
            let old_flag = fun.inst(inst).flags();
            // If the flags are incompatible, the impact is infinite. Otherwise, a
            // negative impact is possible if the flags changes.
            // TODO(impact): finer analysis of the impact of cache directives.
            (old_flag & new_flag).is_empty() || !new_flag.contains(old_flag)
        },
        ir::Action::LowerDimMap(..) =>
            // Creating a temporary array may impact performance negatively.
            true,
        ir::Action::Kind(dim, new_kind) =>
            kind_individual_impact(fun.dim(dim).kind(), new_kind),
        ir::Action::Order(lhs, rhs, order) =>
            order_individual_impact(lhs, rhs, order, fun),
        ir::Action::NestBBIn(..) =>
            // The negative impact of inhibiting the vectorization is taken into accout by
            // Kind actions.
            false,
    }
}

/// Indicates id a kind action may have a negative impact on the execution time,
/// independently of the other actions it may trigger.
fn kind_individual_impact(old_kind: ir::dim::Kind, new_kind: ir::dim::Kind) -> bool {
    let mut new_kind = new_kind & old_kind;
    // If the kind are incompatible, the impact is infinite.
    if new_kind.is_empty() { return true; }
    // Otherwise, a negative impact is possible if the kind changes to a possibly
    // worst kind. Vector is better than Unroll which is better than Loop.
    if (kind::UNROLL | kind::VECTOR).contains(new_kind) {
        new_kind.insert(kind::LOOP | kind::UNROLL);
    }
    !new_kind.contains(old_kind)
}

/// Indicates if an ordering action my have a negative impact on the execution time,
/// independently of the other actions it may trigger.
fn order_individual_impact(lhs: ir::BBId, rhs: ir::BBId, new_order: ir::Order,
                           fun: &ir::Function) -> bool {
    let old_order = fun.order(lhs, rhs);
    let new_order = (old_order & new_order).apply_restrictions();
    // If the orders are incompatible, the impact is infinite.
    if new_order.is_empty() { return true; }
    // If the order does not change, the no impact is possible.
    if new_order == old_order { return false; }
    match (fun.block(lhs).as_dim(), fun.block(rhs).as_dim()) {
        (Some(lhs), Some(rhs)) => {
            // {unroll,vector}-{unroll,vector} ordering has 0 negative impact
            let expanded = kind::UNROLL | kind::VECTOR;
            if expanded.contains(lhs.kind()) && expanded.contains(rhs.kind()) {
                return false;
            }
            // Merging two dimensions has no negative impact.
            if new_order.base() == ir::order::MERGED { return false; }
            // Nesting a thread outer an non-thread has no negative impact.
            let lhs_thread = lhs.is_a(kind::THREAD);
            let rhs_thread = rhs.is_a(kind::THREAD);
            if lhs_thread.is_true() && rhs_thread.is_false()
                && ir::order::OUTER.contains(new_order.base()) { return false; }
            if lhs_thread.is_false() && rhs_thread.is_true()
                && ir::order::INNER.contains(new_order.base()) { return false; }
            true
        },
        (Some(lhs), None) => inst_dim_order_impact(lhs, old_order, new_order),
        (None, Some(rhs)) =>
            inst_dim_order_impact(rhs, old_order.invert(),  new_order.invert()),
        (None, None) =>
            // Instructions are reordered by the driver.
            false,
    }
}

/// Indicates if the ordering action between an instruction and a dimension may have a
/// negative impact on the execution time, indenpendently of the the other actions it may
/// trigger.
fn inst_dim_order_impact(dim: &ir::Dimension, old_order: ir::Order, new_order: ir::Order)
        -> bool {
    // {unroll,vector,block}-instruction ordering has no impact. Indeed, the
    // compiler reorders instruction within a PTX basic block and blocks may only
    // increase the number of time an instruction is executed which is taken into
    // account be ActionInstIn actions.
    if (kind::UNROLL | kind::VECTOR | kind::BLOCK).contains(dim.kind()) { return false; }
    // Merged-in has no impact.
    if new_order.base() == ir::order::MERGED_IN { return false; }
    // Active-in has an impact only if a sequential ordering was possible
    if ir::order::INNER.contains(old_order.base()) { return false; }
    true
}

/// Stores a set of actions. Automatically combine actions if necessary when a new one is
/// added.
#[derive(Default)]
struct ActionSet {
    set: HashMap<ir::ActionArea, ir::Action>,
}

impl ActionSet {
    /// Adds an `Action` to the set. Indicates if the set was modified.
    fn insert(&mut self, action: &ir::Action) -> bool {
        match self.set.entry(action.area()) {
            hash_map::Entry::Vacant(entry) => {
                entry.insert(action.clone());
                true
            },
            hash_map::Entry::Occupied(ref mut entry) =>
                combine_actions(entry.get_mut(), action),
        }
    }
}

/// Combines an action into a other one. Indicates if the action was changed.
fn combine_actions(lhs: &mut ir::Action, rhs: &ir::Action) -> bool {
    match (lhs, rhs) {
        (&mut ir::Action::Order(_, _, ref mut lhs), &ir::Action::Order(_, _, rhs)) => {
            let old_lhs = *lhs;
            *lhs = *lhs & rhs;
            *lhs != old_lhs
        },
        (&mut ir::Action::Kind(_, ref mut lhs), &ir::Action::Kind(_, rhs)) => {
            let old_lhs = *lhs;
            *lhs = *lhs & rhs;
            *lhs != old_lhs
        },
        (&mut ir::Action::MemSpace(_, ref mut lhs), &ir::Action::MemSpace(_, rhs)) => {
            let old_lhs = *lhs;
            *lhs = *lhs & rhs;
            *lhs != old_lhs
        },
        (&mut ir::Action::InstFlag(_, ref mut lhs), &ir::Action::InstFlag(_, rhs)) => {
            let old_lhs = *lhs;
            *lhs = *lhs & rhs;
            *lhs != old_lhs
        },
        (&mut ir::Action::OuterThreadDim(_, _, ref mut lhs),
            &ir::Action::OuterThreadDim(_, _, rhs)) => {
            match (*lhs, rhs) {
                (Trivalent::Maybe, _) |
                (Trivalent::True, Trivalent::True) |
                (Trivalent::False, Trivalent::False) => false,
                (_, _) => {
                    *lhs = Trivalent::Maybe;
                    true
                },
            }
        },
        (&mut ir::Action::Identity, &ir::Action::Identity) |
        (&mut ir::Action::Merge(..), &ir::Action::Merge(..)) |
        (&mut ir::Action::ActiveInstIn(..), &ir::Action::ActiveInstIn(..)) |
        (&mut ir::Action::NestBBIn(..), &ir::Action::NestBBIn(..)) |
        (&mut ir::Action::LowerLayout(..), &ir::Action::LowerLayout(..)) |
        (&mut ir::Action::LowerDimMap(..), &ir::Action::LowerDimMap(..)) => false,
        _ => panic!()
    }
}
