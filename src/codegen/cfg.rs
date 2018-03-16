use codegen::{Dimension, InductionLevel};
use ir;
use search_space::{DimKind, Order, SearchSpace};
use itertools::Itertools;
use std::{self, fmt};

/// Represents a CFG of the targeted device.
pub enum Cfg<'a> {
    /// Represents the root node of the CFG.
    Root(Vec<Cfg<'a>>),
    /// Represents a loop in the CFG.
    Loop(Dimension<'a>, Vec<Cfg<'a>>),
    /// Represents an instruction in the CFG.
    Instruction(&'a ir::Instruction<'a>),
    /// Represent a syncthread instruction of the targeted device.
    Barrier,
    /// Computes an induction variable level, compatible with parallel dimension.
    ParallelInductionLevel(InductionLevel<'a>),
}

impl<'a> Cfg<'a> {
    /// Iterates over the dimensions of the `Cfg`.
    pub fn dimensions(&self) -> impl Iterator<Item=&Dimension<'a>> {
        match *self {
            Cfg::Root(ref body) =>
                box body.iter().flat_map(|cfg| cfg.dimensions()) as Box<Iterator<Item=_>>,
            Cfg::Loop(ref dim, ref body) => {
                let body_dims = body.iter().flat_map(|cfg| cfg.dimensions());
                box std::iter::once(dim).chain(body_dims) as _
            },
            Cfg::Instruction(_) | Cfg::Barrier | Cfg::ParallelInductionLevel(..) =>
                box std::iter::empty() as _,

        }
    }

    /// Iterates over the induction levels of the `Cfg`.
    pub fn induction_levels(&self) -> impl Iterator<Item=&InductionLevel> {
        match *self {
            Cfg::Root(ref body) |
            Cfg::Loop(_, ref body) => {
                let inner = body.iter().flat_map(|cfg| cfg.induction_levels());
                box inner as Box<Iterator<Item=_>>
            },
            Cfg::Instruction(_) | Cfg::Barrier => box std::iter::empty() as _,
            Cfg::ParallelInductionLevel(ref level) => box std::iter::once(level) as _,
        }
    }

    /// Builds a CFG from a list of `CfgEvent`.
    fn from_events<IT>(events: &mut IT) -> Cfg<'a> where IT: Iterator<Item=CfgEvent<'a>> {
        use self::CfgEvent::*;
        let mut body = vec![];
        loop {
            match events.next() {
                Some(Exec(inst)) => body.push(Cfg::Instruction(inst)),
                Some(Enter(_, EntryEvent::SeqDim)) => body.push(Cfg::from_events(events)),
                Some(Enter(_, EntryEvent::ParallelInductionLevel(ind_level))) =>
                    body.push(Cfg::ParallelInductionLevel(ind_level)),
                Some(Exit(_, ExitEvent::Threads)) => body.push(Cfg::Barrier),
                Some(Exit(_, ExitEvent::SeqDim(dim))) =>
                    return Cfg::Loop(dim, body),
                None => {
                    if let Some(&Cfg::Barrier) = body.last() { body.pop(); }
                    return Cfg::Root(body)
                },
            }
        }
    }
}

impl<'a> fmt::Debug for Cfg<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Cfg::Root(ref inners) => write!(f, "{:?}", inners),
            Cfg::Loop(ref dim, ref inners) =>
                write!(f, "Loop([{:?}], {:?})", dim.dim_ids().format(","), inners),
            Cfg::Instruction(inst) => write!(f, "inst {:?}", inst.id()),
            Cfg::Barrier => write!(f, "Barrier"),
            Cfg::ParallelInductionLevel(InductionLevel {
                ind_var, increment: Some((dim, _)), ..
            }) => write!(f, "induction lvl (dim {:?}, {:?}) ", dim, ind_var),
            Cfg::ParallelInductionLevel(ref level) =>
                write!(f, "induction var {:?}", level.ind_var),
        }
    }
}

/// Builds the CFG from the list of dimensions and instructions. Also returns the list of
/// thread and block dimensions.
pub fn build<'a>(space: &'a SearchSpace<'a>,
                 dims: Vec<Dimension<'a>>,
                 precomputed_ind_levels: Vec<InductionLevel<'a>>)
    -> (Vec<Dimension<'a>>, Vec<Dimension<'a>>, Cfg<'a>)
{
    let precomputed_ind_levels = precomputed_ind_levels.into_iter().map(|level| {
        let dim = unwrap!(level.increment).0;
        CfgEvent::Enter(dim, EntryEvent::ParallelInductionLevel(level))
    });
    let (block_dims, thread_dims, mut events) = gen_events(space, dims);
    events.sort_by(|lhs, rhs| lhs.cmp(rhs, space));
    let mut events = precomputed_ind_levels.chain(events);
    let cfg = Cfg::from_events(&mut events);
    assert!(events.next().is_none());
    (block_dims, thread_dims, cfg)
}

/// Describes the program points encountered when walking a CFG.
enum CfgEvent<'a> {
    Exec(&'a ir::Instruction<'a>),
    Enter(ir::dim::Id, EntryEvent<'a>),
    Exit(ir::dim::Id, ExitEvent<'a>),
}

enum EntryEvent<'a> { SeqDim, ParallelInductionLevel(InductionLevel<'a>) }

enum ExitEvent<'a> { SeqDim(Dimension<'a>), Threads }

impl<'a> CfgEvent<'a> {
    /// Indiciates the order of `self` with regards to `other`.
    fn cmp(&self, other: &CfgEvent, space: &SearchSpace) -> std::cmp::Ordering {
        let (lhs_bb, rhs_bb) = (self.bb_id(), other.bb_id());
        if lhs_bb == rhs_bb { self.cmp_within_bb(other) } else {
            use self::CfgEvent::*;
            use std::cmp::Ordering::*;
            let order = space.domain().get_order(lhs_bb, rhs_bb);
            match (self, other, order) {
                (_, _, Order::MERGED) => self.cmp_within_bb(other),
                (_, _, Order::BEFORE) => Less,
                (_, _, Order::AFTER) => Greater,
                (&Exec(_), &Exec(_), Order::ORDERED) => lhs_bb.cmp(&rhs_bb),
                (&Enter(..), _, Order::OUTER) => Less,
                (_, &Enter(..), Order::INNER) => Greater,
                (_, &Exit(..), Order::INNER)  => Less,
                (&Exit(..), _, Order::OUTER) => Greater,
                (lhs, rhs, ord) =>
                    panic!("Invalid order between {:?} and {:?}: {:?}.", lhs, rhs, ord),
            }
        }
    }

    /// Indicates the order of `self with `other`, assuming they are events on the same
    /// basic block.
    fn cmp_within_bb(&self, other: &CfgEvent) -> std::cmp::Ordering {
        use self::CfgEvent::*;
        match (self, other) {
            (&Enter(..), &Enter(..)) | (&Exec(..), &Exec(..)) | (&Exit(..), &Exit(..))
                => std::cmp::Ordering::Equal,
            (&Enter(..), _) | (_, &Exit(..)) => std::cmp::Ordering::Less,
            (&Exit(..), _) | (_, &Enter(..)) => std::cmp::Ordering::Greater,
        }
    }

    /// Returns an id of a `BasicBlock` mentioned by the event.
    fn bb_id(&self) -> ir::BBId {
        match *self {
            CfgEvent::Exec(inst) => inst.id().into(),
            CfgEvent::Enter(dim, _) | CfgEvent::Exit(dim, _) => dim.into(),
        }
    }
}

impl<'a> fmt::Debug for CfgEvent<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            CfgEvent::Exec(inst) => write!(f, "inst {:?}", inst.id()),
            CfgEvent::Enter(dim, _) => write!(f, "enter dim {:?}", dim),
            CfgEvent::Exit(dim, _) => write!(f, "exit dim {:?}", dim),
        }
    }
}

/// Generates the list of `CfgEvent`s and the list of block and thread dimensions.
fn gen_events<'a>(space: &'a SearchSpace<'a>, dims: Vec<Dimension<'a>>)
    -> (Vec<Dimension<'a>>, Vec<Dimension<'a>>, Vec<CfgEvent<'a>>)
{
    let mut block_dims = Vec::new();
    let mut thread_dims: Vec<Option<Dimension<'a>>> = (0..3).map(|_| None).collect_vec();
    let mut events = space.ir_instance().insts().map(CfgEvent::Exec).collect_vec();
    for dim in dims {
        let mut add_thread_dim =
            |mut dim: Dimension<'a>, nesting: usize, events: &mut Vec<_>|
        {
            for level in dim.drain_induction_levels() {
                let event = EntryEvent::ParallelInductionLevel(level);
                events.push(CfgEvent::Enter(dim.id(), event));
            }
            match thread_dims[nesting] {
                Some(ref mut other_dim) => other_dim.merge_from(dim),
                ref mut x @ None => *x = Some(dim),
            }
        };
        match dim.kind() {
            DimKind::BLOCK => block_dims.push(dim),
            DimKind::THREAD_X => {
                events.push(CfgEvent::Exit(dim.id(), ExitEvent::Threads));
                add_thread_dim(dim, 0, &mut events);
            },
            DimKind::THREAD_Y => add_thread_dim(dim, 1, &mut events),
            DimKind::THREAD_Z => add_thread_dim(dim, 2, &mut events),
            _ => {
                events.push(CfgEvent::Enter(dim.id(), EntryEvent::SeqDim));
                events.push(CfgEvent::Exit(dim.id(), ExitEvent::SeqDim(dim)));
            },
        }
    }
    block_dims.sort_by(|lhs, rhs| {
        if lhs.id() == rhs.id() { return std::cmp::Ordering::Equal; }
        match space.domain().get_order(lhs.id().into(), rhs.id().into()) {
            Order::OUTER => std::cmp::Ordering::Less,
            Order::INNER => std::cmp::Ordering::Greater,
            _ => panic!("invalid order between block dim {:?} and {:?}",
                        lhs.id(), rhs.id()),
        }
    });
    (block_dims, Itertools::flatten(thread_dims.into_iter()).collect(), events)
}
