//! Choices that can be applied to split the search space.
use std::fmt;

use crate::explorer::config;
use crate::ir::{self, Statement};
use crate::search_space::{Action, DimKind, Domain, NumSet, Order, SearchSpace};
use itertools::Itertools;
use log::trace;
use serde::{Deserialize, Serialize};
use utils::unwrap;

/// Either a regular action or a manually applied action.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionEx {
    Action(Action),
    LowerLayout {
        mem: ir::MemId,
        st_dims: Vec<ir::DimId>,
        ld_dims: Vec<ir::DimId>,
    },
}

impl fmt::Debug for ActionEx {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // Actions are already explicitly self-describing enough
            ActionEx::Action(action) => write!(f, "{:?}", action),
            ActionEx::LowerLayout {
                mem,
                st_dims,
                ld_dims,
            } => write!(
                f,
                "LowerLayout {{ mem: {:?}, st_dims: {:?}, ld_dims: {:?} }}",
                mem, st_dims, ld_dims
            ),
        }
    }
}

impl ir::IrDisplay for ActionEx {
    fn fmt(&self, fmt: &mut fmt::Formatter, function: &ir::Function) -> fmt::Result {
        match self {
            ActionEx::Action(action) => write!(fmt, "{}", action.display(function)),
            ActionEx::LowerLayout {
                mem,
                st_dims,
                ld_dims,
            } => write!(
                fmt,
                "LowerLayout {{ mem: {:?}, st_dims: {:?}, ld_dims: {:?} }}",
                mem, st_dims, ld_dims
            ),
        }
    }
}

/// Represents a choice that splits a search space in multiple ones.
// TODO(search_space): explore and lower loayouts directly from the regular actions.
pub type Choice = Vec<ActionEx>;

pub fn list<'a>(
    iter_choice: impl IntoIterator<Item = &'a config::ChoiceGroup> + 'a,
    space: &'a SearchSpace,
) -> impl Iterator<Item = Choice> + 'a {
    iter_choice
        .into_iter()
        .map(move |choice_grp| -> Box<dyn Iterator<Item = Choice> + 'a> {
            use crate::explorer::config::ChoiceGroup;
            let fun = space.ir_instance();
            match choice_grp {
                ChoiceGroup::LowerLayout => Box::new(
                    fun.layouts_to_lower()
                        .iter()
                        .map(move |&layout| lower_layout_choice(space, layout)),
                ),
                ChoiceGroup::Size => Box::new(fun.static_dims().flat_map(move |dim| {
                    let sizes = space.domain().get_size(dim.id());
                    gen_choice(sizes.list(), &|s| Action::Size(dim.id(), s))
                })),
                ChoiceGroup::ThreadSize => {
                    Box::new(fun.static_dims().flat_map(move |dim| {
                        let kinds = space.domain().get_dim_kind(dim.id());
                        if kinds.intersects(DimKind::THREAD) {
                            let sizes = space.domain().get_size(dim.id());
                            gen_choice(sizes.list(), &|s| Action::Size(dim.id(), s))
                        } else {
                            None
                        }
                    }))
                }
                ChoiceGroup::DimKind => Box::new(fun.dims().flat_map(move |dim| {
                    let kinds = space.domain().get_dim_kind(dim.id());
                    gen_choice(kinds.list(), &|k| Action::DimKind(dim.id(), k))
                })),
                ChoiceGroup::Threads => Box::new(fun.dims().flat_map(move |dim| {
                    let kinds = space.domain().get_dim_kind(dim.id());
                    gen_choice(kinds.bisect(DimKind::THREAD), &|k| {
                        Action::DimKind(dim.id(), k)
                    })
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
                                let orders = space.domain().get_order(lhs, rhs);
                                gen_choice(orders.list(), &|o| Action::Order(lhs, rhs, o))
                            },
                        )
                    }))
                }

                ChoiceGroup::DimNesting => {
                    Box::new(fun.dims().enumerate().flat_map(move |(i, lhs)| {
                        let lhs = lhs.stmt_id();
                        let dims = fun.dims().take(i).map(|x| x.stmt_id());

                        dims.chain(fun.insts().map(|x| x.stmt_id())).flat_map(
                            move |rhs| {
                                let available_orders = space.domain().get_order(lhs, rhs);
                                let nesting_orders = Order::INNER | Order::OUTER;

                                let orders = (available_orders & Order::INNER)
                                    .into_option()
                                    .into_iter()
                                    .chain(
                                        (available_orders & Order::OUTER).into_option(),
                                    )
                                    .chain(
                                        (available_orders & !nesting_orders)
                                            .into_option(),
                                    );

                                gen_choice(orders, &|order| {
                                    Action::Order(lhs, rhs, order)
                                })
                            },
                        )
                    }))
                }

                ChoiceGroup::DimFusion => {
                    Box::new(fun.dims().enumerate().flat_map(move |(i, lhs)| {
                        let lhs = lhs.stmt_id();
                        let dims = fun.dims().take(i).map(|x| x.stmt_id());

                        dims.chain(fun.insts().map(|x| x.stmt_id())).flat_map(
                            move |rhs| {
                                let available_orders = space.domain().get_order(lhs, rhs);

                                gen_choice(
                                    available_orders.bisect(Order::MERGED),
                                    &|order| Action::Order(lhs, rhs, order),
                                )
                            },
                        )
                    }))
                }

                ChoiceGroup::MemSpace => {
                    Box::new(fun.mem_blocks().flat_map(move |block| {
                        let mem_spaces = space.domain().get_mem_space(block.mem_id());
                        gen_choice(mem_spaces.list(), &|s| {
                            Action::MemSpace(block.mem_id(), s)
                        })
                    }))
                }
                ChoiceGroup::InstFlag => {
                    Box::new(fun.mem_insts().flat_map(move |inst| {
                        let flags = space.domain().get_inst_flag(inst.id()).list();
                        gen_choice(flags, &|f| Action::InstFlag(inst.id(), f))
                    }))
                }
            }
        })
        .flatten()
}

/// This function is to be either removed or reimplemented eventually. It is just a replacement for
/// the previous list implementation (exposes the choices in the same order). Default should
/// preferably be handled in config file
pub fn default_list<'a>(space: &'a SearchSpace) -> impl Iterator<Item = Choice> + 'a {
    list(&config::DEFAULT_ORDERING, space)
}

/// Generates a choice from a list of possible values.
fn gen_choice<T, IT>(values: IT, action_gen: &dyn Fn(T) -> Action) -> Option<Choice>
where
    IT: IntoIterator<Item = T>,
{
    let choice = values
        .into_iter()
        .map(action_gen)
        .map(ActionEx::Action)
        .collect_vec();
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

/// Generates the different ways to lower a layout.
fn lower_layout_choice(space: &SearchSpace, mem: ir::MemId) -> Vec<ActionEx> {
    let mem_block = space.ir_instance().mem_block(mem);
    let mapped_dims = mem_block.mapped_dims().iter().cloned().collect_vec();
    // Order dimensions until the stride is too big to matter in any way.
    let mut to_process = vec![(vec![], mapped_dims, mem_block.base_size())];
    let mut actions = Vec::new();
    while let Some((ordered_dims, remaining_dims, ordered_size)) = to_process.pop() {
        // TODO(search_space): parametrize the max stride for layout ordering
        if ordered_size >= 32 * 8 || remaining_dims.is_empty() {
            let (st_dims, ld_dims) = remaining_dims
                .into_iter()
                .chain(ordered_dims.into_iter().rev())
                .unzip();
            actions.push(ActionEx::LowerLayout {
                mem,
                st_dims,
                ld_dims,
            });
        } else {
            for i in 0..remaining_dims.len() {
                let mut remaining_dims = remaining_dims.clone();
                let mut ordered_dims = ordered_dims.clone();
                let dim_pair = remaining_dims.swap_remove(i);
                let possible_sizes =
                    unwrap!(space.ir_instance().dim(dim_pair.0).possible_sizes());
                let size = space
                    .domain()
                    .get_size(dim_pair.0)
                    .min_value(possible_sizes);
                let ordered_size = ordered_size * size;
                ordered_dims.push(dim_pair);
                to_process.push((ordered_dims, remaining_dims, ordered_size));
            }
        }
    }
    actions
}

/// The error type for action application errors.
///
/// Errors mostly originate from constraint propagation failing.
pub struct ActionError {
    action: ActionEx,
    space: SearchSpace,
}

impl fmt::Debug for ActionError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("ActionError")
            .field("action", &self.action)
            .field("space", &"..")
            .finish()
    }
}

impl fmt::Display for ActionError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.action {
            ActionEx::Action(action) => write!(
                fmt,
                "failed to apply action `{}` to instance:\n{}",
                action.display(self.space.ir_instance()),
                self.space.ir_instance(),
            ),
            ActionEx::LowerLayout { mem, .. } => {
                // We can't use the IR instance here, since it might be in an inconsistent state.
                write!(fmt, "failed to lower layout for {}", mem)
            }
        }
    }
}

impl std::error::Error for ActionError {}

impl ActionEx {
    /// Apply this action to a search space
    pub fn apply_to(&self, mut space: SearchSpace) -> Result<SearchSpace, ActionError> {
        match match *self {
            ActionEx::Action(action) => space.apply_decisions(vec![action]),
            ActionEx::LowerLayout {
                mem,
                ref st_dims,
                ref ld_dims,
            } => space.lower_layout(mem, st_dims, ld_dims),
        } {
            Ok(()) => Ok(space),
            Err(()) => {
                // This contains the space to which the action was initially applied.  Note that
                // since action application is a destructive operation, the space might in general
                // be in an inconsistent state.  However, the IR instance is still valid when
                // `apply_decisions` failed (but *not* when `lower_layout` failed!), and that is
                // all we need to display a human-readable error message.
                Err(ActionError {
                    action: self.clone(),
                    space,
                })
            }
        }
    }
}
