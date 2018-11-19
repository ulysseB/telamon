//! Choices that can be applied to split the search space.
use explorer::config;
use ir::Statement;
use itertools::Itertools;
use search_space::{Action, Domain, Order, SearchSpace};

/// Represents a choice that splits a search space in multiple ones.
// TODO(search_space): explore and lower loayouts directly from the regular actions.
pub type Choice = Vec<Action>;

/// An enum listing the Group of choices we can make
/// For example, we can make first all DimKind decisions, then all Order decisions, etc.
#[derive(Debug, Clone, Copy)]
pub enum ChoiceGroup {
    Size,
    DimKind,
    DimMap,
    Order,
    MemSpace,
    InstFlag,
    Rank,
}

impl From<config::ChoiceGroup> for ChoiceGroup {
    fn from(conf_ch_grp: config::ChoiceGroup) -> Self {
        match conf_ch_grp {
            config::ChoiceGroup::Size => ChoiceGroup::Size,
            config::ChoiceGroup::DimKind => ChoiceGroup::DimKind,
            config::ChoiceGroup::DimMap => ChoiceGroup::DimMap,
            config::ChoiceGroup::Order => ChoiceGroup::Order,
            config::ChoiceGroup::MemSpace => ChoiceGroup::MemSpace,
            config::ChoiceGroup::InstFlag => ChoiceGroup::InstFlag,
            config::ChoiceGroup::Rank => ChoiceGroup::Rank,
        }
    }
}

/// This struct nests two iterators inside each other and then implements Iterator with the Item of
/// the internal Iterator
/// We need that because the choices are generated in different fashion for the different cases
/// (DimMap, Order...) so we have to iterate on several kinds of iterators (Map, FlatMap...) in a
/// statically unknown order.
struct NestedIterator<I: Iterator>
where
    I::Item: Iterator,
{
    /// The high level Iterator
    glob_iterator: I,
    /// The internal iterator we are currently iterating on
    current_local_iterator: Option<I::Item>,
}

impl<I: Iterator> NestedIterator<I>
where
    I::Item: Iterator,
{
    fn new(iterator: I) -> Self {
        NestedIterator {
            glob_iterator: iterator,
            current_local_iterator: None,
        }
    }
}

impl<I: Iterator> Iterator for NestedIterator<I>
where
    I::Item: Iterator,
{
    type Item = <I::Item as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut current_it) = self.current_local_iterator {
                if let Some(choice) = current_it.next() {
                    break Some(choice);
                }
            }
            // If we are here, either there is no current_local_iterator or the current_local_iterator
            // is exhausted, we should update it. If glob_iterator itself is exhausted, we return None
            if let Some(local_it) = self.glob_iterator.next() {
                self.current_local_iterator = Some(local_it);
            } else {
                break None;
            }
        }
    }
}

pub fn list<'a>(
    iter_choice: impl Iterator<Item = ChoiceGroup> + 'a,
    space: &'a SearchSpace<'a>,
) -> impl Iterator<Item = Choice> + 'a {
    NestedIterator::new(iter_choice.map(
        move |choice_grp| -> Box<dyn Iterator<Item = Choice> + 'a> {
            let fun = space.ir_instance();
            match choice_grp {
                ChoiceGroup::Size => Box::new(fun.static_dims().flat_map(move |dim| {
                    let sizes = space.domain().get_size(dim.id());
                    gen_choice(sizes.list(), &|s| Action::Size(dim.id(), s))
                })),
                ChoiceGroup::DimKind => Box::new(fun.dims().flat_map(move |dim| {
                    let kinds = space.domain().get_dim_kind(dim.id());
                    gen_choice(kinds.list(), &|k| Action::DimKind(dim.id(), k))
                })),
                ChoiceGroup::DimMap => {
                    Box::new(fun.static_dims().enumerate().flat_map(move |(i, lhs)| {
                        fun.static_dims().take(i).flat_map(move |rhs| {
                            let mappings =
                                space.domain().get_thread_mapping(lhs.id(), rhs.id());
                            gen_choice(mappings.list(), &|m| {
                                Action::ThreadMapping(lhs.id(), rhs.id(), m)
                            })
                        })
                    }))
                }
                ChoiceGroup::Order => {
                    Box::new(fun.dims().enumerate().flat_map(move |(i, lhs)| {
                        // TODO(search_space): avoid picking ordering decisions that have little impact.
                        // For this, we should avoid dimension-instruction and dimension-vector dim
                        // orderings. The problem is that we do not know wich choice to pick in the end.
                        let lhs = lhs.stmt_id();
                        let dims = fun.dims().take(i).map(|x| x.stmt_id());
                        dims.chain(fun.insts().map(|x| x.stmt_id())).flat_map(
                            move |rhs| {
                                let orders = space.domain().get_order(lhs.into(), rhs);
                                gen_choice(orders.list(), &|o| Action::Order(lhs, rhs, o))
                            },
                        )
                    }))
                }
                ChoiceGroup::MemSpace => Box::new(fun.variables().flat_map(move |var| {
                    let mem_spaces = space.domain().get_memory_space(var.id());
                    gen_choice(mem_spaces.list(), &|s| Action::MemorySpace(var.id(), s))
                })),
                ChoiceGroup::InstFlag => {
                    Box::new(fun.mem_insts().flat_map(move |inst| {
                        let flags = space.domain().get_inst_flag(inst.id()).list();
                        gen_choice(flags, &|f| Action::InstFlag(inst.id(), f))
                    }))
                }
                ChoiceGroup::Rank => {
                    // TODO(ulysse): only explore ranks that are smaller or equal to the
                    // number of instantiated decisions.
                    Box::new(fun.mem_layout_dimensions().flat_map(move |dim| {
                        let ranks = space.domain().get_rank(dim.id()).list();
                        gen_choice(ranks, &|r| Action::Rank(dim.id(), r))
                    }))
                }
            }
        },
    ))
}

lazy_static! {
    static ref DEFAULT_ORDERING: Vec<ChoiceGroup> = vec![
        ChoiceGroup::Size,
        ChoiceGroup::DimKind,
        ChoiceGroup::DimMap,
        ChoiceGroup::MemSpace,
        ChoiceGroup::Rank,
        ChoiceGroup::Order,
        ChoiceGroup::InstFlag,
    ];
}

/// This function is to be either removed or reimplemented eventually. It is just a replacement for
/// the previous list implementation (exposes the choices in the same order). Default should
/// preferably be handled in config file
pub fn default_list<'a>(space: &'a SearchSpace<'a>) -> impl Iterator<Item = Choice> + 'a {
    list(DEFAULT_ORDERING.iter().cloned(), space)
}

pub fn list_with_conf<'a, 'b: 'a>(
    choice_ordering: &'b config::ChoiceOrdering,
    space: &'a SearchSpace<'a>,
) -> impl Iterator<Item = Choice> + 'a {
    list(
        choice_ordering
            .iter()
            .cloned()
            .map(|choice_grp| ChoiceGroup::from(choice_grp)),
        space,
    )
}

/// Generates a choice from a list of possible values.
fn gen_choice<T, IT>(values: IT, action_gen: &Fn(T) -> Action) -> Option<Choice>
where
    IT: IntoIterator<Item = T>,
{
    let choice = values.into_iter().map(action_gen).collect_vec();
    if choice.len() <= 1 {
        None
    } else {
        Some(choice)
    }
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
    let pairs = space
        .ir_instance()
        .statements()
        .cartesian_product(space.ir_instance().dims())
        .map(|(lhs, rhs)| (lhs.stmt_id(), rhs.stmt_id()))
        .filter(|&(lhs, rhs)| lhs != rhs)
        .filter(|&(lhs, rhs)| !space.domain().get_order(lhs, rhs).is_constrained())
        .collect_vec();
    for (lhs, rhs) in pairs {
        let order = space.domain().get_order(lhs, rhs);
        if order.is_constrained() {
            continue;
        }
        let new_order = if order.intersects(Order::BEFORE) {
            Order::BEFORE
        } else if order.intersects(Order::AFTER) {
            Order::AFTER
        } else {
            panic!(
                "unconstrained order between {:?} and {:?}: {:?}",
                lhs, rhs, order
            )
        };
        let action = Action::Order(lhs, rhs, new_order);
        unwrap!(space.apply_decisions(vec![action]), "{:?}", action);
    }
    space
}
