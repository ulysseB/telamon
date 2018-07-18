//! Choices that can be applied to split the search space.
use ir::{self, BasicBlock};
use ir::mem::Block;
use search_space::{Action, Domain, Order, SearchSpace};
use search_space::NumDomain;
use itertools::Itertools;

/// Represents a choice that splits a search space in multiple ones.
// TODO(search_space): explore and lower loayouts directly from the regular actions.
pub type Choice = Vec<ActionEx>;

/// Either a regular action or a manually applied action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionEx {
    TileSizes(Vec<Vec<u32>>),
    Action(Action),
    LowerLayout {
        mem: ir::mem::InternalId,
        st_dims: Vec<ir::dim::Id>,
        ld_dims: Vec<ir::dim::Id>,
    }
}

/// Lists the choices that can be applied to a function.
pub fn list<'a>(space: &'a SearchSpace<'a>) -> impl Iterator<Item=Choice> + 'a {
    // FIXME: explore tile sizes
    let fun = space.ir_instance();
    let static_dims = fun.dims().filter(|d| d.possible_sizes().is_some());
    fun.layouts_to_lower().iter().map(move |&layout| {
        lower_layout_choice(space, layout)
    }).chain(fun.dims().flat_map(move |dim| {
        let kinds = space.domain().get_dim_kind(dim.id());
        gen_choice(kinds.list(), &|k| Action::DimKind(dim.id(), k))
    })).chain(static_dims.clone().enumerate().flat_map(move |(i, lhs)| {
        static_dims.clone().take(i).flat_map(move |rhs| {
            let mappings = space.domain().get_thread_mapping(lhs.id(), rhs.id());
            gen_choice(mappings.list(), &|m| Action::ThreadMapping(lhs.id(), rhs.id(), m))
        })
    })).chain(fun.internal_mem_blocks().flat_map(move |block| {
        let mem_spaces = space.domain().get_mem_space(block.mem_id());
        gen_choice(mem_spaces.list(), &|s| Action::MemSpace(block.mem_id(), s))
    })).chain(fun.dims().enumerate().flat_map(move |(i, lhs)| {
        // TODO(search_space): avoid picking ordering decisions that have little impact.
        // For this, we should avoid dimension-instruction and dimension-vector dim
        // orderings. The problem is that we do not know wich choice to pick in the end.
        let lhs = lhs.bb_id();
        let dims = fun.dims().take(i).map(|x| x.bb_id());
        dims.chain(fun.insts().map(|x| x.bb_id())).flat_map(move |rhs| {
            let orders = space.domain().get_order(lhs.into(), rhs);
            gen_choice(orders.list(), &|o| Action::Order(lhs, rhs, o))
        })
    })).chain(fun.mem_insts().flat_map(move |inst| {
        let flags = space.domain().get_inst_flag(inst.id()).list();
        gen_choice(flags, &|f| Action::InstFlag(inst.id(), f))
    }))
}

/// Generates a choice from a list of possible values.
fn gen_choice<T, IT>(values: IT, action_gen: &Fn(T) -> Action) -> Option<Choice>
        where IT: IntoIterator<Item=T> {
    let choice = values.into_iter().map(action_gen).map(ActionEx::Action).collect_vec();
    if choice.len() <= 1 { None } else { Some(choice) }
}

/// Chooses an order between instructions and dimensions when multiple are possible.
/// The function assumes the order between dimensions is already fixed.
// TODO(search_space): fix order has currently no effect. Should we remove it ?
// It is unused because inst-dim and dim-dim decisions are fixed by the explorer. We
// cannot make them free as we might end-up in a dead-end.
pub fn fix_order(mut space: SearchSpace) -> SearchSpace {
    // TODO(search_space): make fix_order useless with a differential model
    trace!("adding arbitrary constraints to the order");
    // Fix the order between instructions and dimensions.
    let pairs = space.ir_instance().blocks()
        .cartesian_product(space.ir_instance().dims())
        .map(|(lhs, rhs)| (lhs.bb_id(), rhs.bb_id()))
        .filter(|&(lhs, rhs)| lhs != rhs)
        .filter(|&(lhs, rhs)| !space.domain().get_order(lhs, rhs).is_constrained())
        .collect_vec();
    for (lhs, rhs) in pairs {
        let order = space.domain().get_order(lhs, rhs);
        if order.is_constrained() { continue; }
        let new_order = if order.intersects(Order::BEFORE) {
            Order::BEFORE
        } else if order.intersects(Order::AFTER) {
            Order::AFTER
        } else {
            panic!("unconstrained order between {:?} and {:?}: {:?}", lhs, rhs, order)
        };
        let action = Action::Order(lhs, rhs, new_order);
        unwrap!(space.apply_decisions(vec![action]), "{:?}", action);
    }
    space
}

/// Generates the different ways to lower a layout.
fn lower_layout_choice(space: &SearchSpace, mem: ir::mem::InternalId) -> Vec<ActionEx> {
    let mem_block = space.ir_instance().internal_mem_block(mem);
    let mapped_dims = mem_block.mapped_dims().iter().cloned().collect_vec();
    // Order dimensions until the stride is too big to matter in any way.
    let mut to_process = vec![(vec![], mapped_dims, mem_block.base_size())];
    let mut actions = Vec::new();
    while let Some((ordered_dims, remaining_dims, ordered_size)) = to_process.pop() {
        // TODO(search_space): parametrize the max stride for layout ordering
        if ordered_size >= 32 * 8 || remaining_dims.is_empty() {
            let (st_dims, ld_dims) = remaining_dims.into_iter()
                .chain(ordered_dims.into_iter().rev()).unzip();
            actions.push(ActionEx::LowerLayout { mem, st_dims, ld_dims });
        } else {
            for i in 0..remaining_dims.len() {
                let mut remaining_dims = remaining_dims.clone();
                let mut ordered_dims = ordered_dims.clone();
                let dim_pair = remaining_dims.swap_remove(i);
                let size = space.domain().get_size(dim_pair.0).min();
                let ordered_size = ordered_size * size;
                ordered_dims.push(dim_pair);
                to_process.push((ordered_dims, remaining_dims, ordered_size));
            }
        }
    }
    actions
}
