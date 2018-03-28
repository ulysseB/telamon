use codegen::{Dimension, InductionLevel, Instruction};
use ir;
use search_space::{DimKind, Order, SearchSpace, ThreadMapping};
use itertools::Itertools;
use std::{self, fmt};

/// Represents a CFG of the targeted device.
pub enum Cfg<'a> {
    /// Represents the root node of the CFG.
    Root(Vec<Cfg<'a>>),
    /// Represents a loop in the CFG.
    Loop(Dimension<'a>, Vec<Cfg<'a>>),
    /// Represents an instruction in the CFG.
    Instruction(Instruction<'a>),
    /// Enables side effects on selected thread dimensions.
    EnableThreads(Vec<bool>),
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
                Box::new(body.iter().flat_map(|cfg| cfg.dimensions()))
                    as Box<Iterator<Item=_>>,
            Cfg::Loop(ref dim, ref body) => {
                let body_dims = body.iter().flat_map(|cfg| cfg.dimensions());
                Box::new(std::iter::once(dim).chain(body_dims)) as _
            },
            _ => Box::new(std::iter::empty()) as _,
        }
    }

    /// Iterates over the instructions of the `Cfg`.
    pub fn instructions(&self) -> impl Iterator<Item=&Instruction<'a>> {
        match *self {
            Cfg::Root(ref body) | Cfg::Loop(_, ref body) => {
                let iter = body.iter().flat_map(|cfg| cfg.instructions());
                Box::new(iter) as Box<Iterator<Item=_>>
            }
            Cfg::Instruction(ref inst) => Box::new(std::iter::once(inst)) as _,
            _ => Box::new(std::iter::empty()) as _,
        }
    }

    /// Iterates over the induction levels of the `Cfg`.
    pub fn induction_levels(&self) -> impl Iterator<Item=&InductionLevel> {
        match *self {
            Cfg::Root(ref body) |
            Cfg::Loop(_, ref body) => {
                let inner = body.iter().flat_map(|cfg| cfg.induction_levels());
                Box::new(inner) as Box<Iterator<Item=_>>
            },
            Cfg::ParallelInductionLevel(ref level) => Box::new(std::iter::once(level)) as _,
            _ => Box::new(std::iter::empty()) as _,
        }
    }

    /// Builds a CFG from a list of `CfgEvent`.
    fn from_events<IT>(events: &mut IT,
                       has_barrier: &mut bool,
                       last_thread_mask: &mut Vec<bool>,
                       thread_mask: &mut Vec<bool>)
        -> Cfg<'a> where IT: Iterator<Item=CfgEvent<'a>>
    {
        use self::CfgEvent::*;
        let mut body = vec![];
        // FIXME: move thread mask before unrolled and vector dimensions
        loop {
            match events.next() {
                Some(Exec(inst)) => {
                    *has_barrier = false;
                    if last_thread_mask != thread_mask {
                        last_thread_mask.copy_from_slice(thread_mask);
                        body.push(Cfg::EnableThreads(thread_mask.clone()));
                    }
                    body.push(Cfg::Instruction(inst))
                },
                Some(Enter(_, EntryEvent::SeqDim(kind))) => {
                    if kind == DimKind::UNROLL || kind == DimKind::VECTOR {
                        // We can safely enforce the thread mask here since thread
                        // dimensions are forced outside unroll and vector dimensions.
                        if last_thread_mask != thread_mask {
                            last_thread_mask.copy_from_slice(thread_mask);
                            body.push(Cfg::EnableThreads(thread_mask.clone()));
                        }
                    }
                    let cfg = Cfg::from_events(
                        events, has_barrier, last_thread_mask, thread_mask);
                    body.push(cfg);
                },
                Some(Enter(_, EntryEvent::ParallelInductionLevel(ind_level))) =>
                    body.push(Cfg::ParallelInductionLevel(ind_level)),
                Some(Enter(_, EntryEvent::ThreadDim(pos))) => {
                    thread_mask[pos] = true;
                    if !*has_barrier {
                        *has_barrier = true;
                        body.push(Cfg::Barrier)
                    }
                }
                Some(Exit(_, ExitEvent::ThreadDim)) => {
                    for item in thread_mask.iter_mut() { *item = false; }
                    if !*has_barrier {
                        *has_barrier = true;
                        body.push(Cfg::Barrier)
                    }
                }
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
            Cfg::Instruction(ref inst) => write!(f, "inst {:?}", inst.id()),
            Cfg::Barrier => write!(f, "Barrier"),
            Cfg::EnableThreads(ref dims) => write!(f, "Threads({:?})", dims),
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
                 insts: Vec<Instruction<'a>>,
                 dims: Vec<Dimension<'a>>,
                 precomputed_ind_levels: Vec<InductionLevel<'a>>)
    -> (Vec<Dimension<'a>>, Vec<Dimension<'a>>, Cfg<'a>)
{
    let precomputed_ind_levels = precomputed_ind_levels.into_iter().map(|level| {
        let dim = unwrap!(level.increment).0;
        CfgEvent::Enter(dim, EntryEvent::ParallelInductionLevel(level))
    });
    let (block_dims, thread_dims, mut events) = gen_events(space, insts, dims);
    events.sort_by(|lhs, rhs| lhs.cmp(rhs, space));
    let mut events = precomputed_ind_levels.chain(events);
    let mut thread_mask = vec![true; thread_dims.len()];
    let cfg = Cfg::from_events(&mut events, &mut true, &mut thread_mask.clone(),
                               &mut thread_mask);
    assert!(events.next().is_none());
    (block_dims, thread_dims, cfg)
}

/// Describes the program points encountered when walking a CFG.
enum CfgEvent<'a> {
    Exec(Instruction<'a>),
    Enter(ir::dim::Id, EntryEvent<'a>),
    Exit(ir::dim::Id, ExitEvent<'a>),
}

/// An event to process when entering a dimension.
enum EntryEvent<'a> {
    /// Enter a sequential dimension.
    SeqDim(DimKind),
    /// Enter a thread dimension.
    ThreadDim(usize),
    /// Compute a parallel induction level.
    ParallelInductionLevel(InductionLevel<'a>)
}

/// An event to process when exiting a dimension.
enum ExitEvent<'a> { SeqDim(Dimension<'a>), ThreadDim }

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
                (&Exec(_), &Exec(_), Order::ORDERED) => lhs_bb.cmp(&rhs_bb),
                (_, _, Order::BEFORE) |
                (&Enter(..), _, Order::OUTER) |
                (_, &Exit(..), Order::INNER) => Less,
                (_, _, Order::AFTER) |
                (_, &Enter(..), Order::INNER) |
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
            CfgEvent::Exec(ref inst) => inst.id().into(),
            CfgEvent::Enter(dim, _) | CfgEvent::Exit(dim, _) => dim.into(),
        }
    }
}

impl<'a> fmt::Debug for CfgEvent<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            CfgEvent::Exec(ref inst) => write!(f, "inst {:?}", inst.id()),
            CfgEvent::Enter(dim, _) => write!(f, "enter dim {:?}", dim),
            CfgEvent::Exit(dim, _) => write!(f, "exit dim {:?}", dim),
        }
    }
}

/// Generates the list of `CfgEvent`s and the list of block and thread dimensions.
fn gen_events<'a>(space: &'a SearchSpace<'a>,
                  insts: Vec<Instruction<'a>>,
                  dims: Vec<Dimension<'a>>)
    -> (Vec<Dimension<'a>>, Vec<Dimension<'a>>, Vec<CfgEvent<'a>>)
{
    let mut block_dims = Vec::new();
    let mut thread_dims = Vec::new();
    let mut events = insts.into_iter().map(CfgEvent::Exec).collect_vec();
    // Create dimension events and sort thread and block dims.
    for mut dim in dims {
        match dim.kind() {
            DimKind::BLOCK => block_dims.push(dim),
            DimKind::THREAD => {
                events.push(CfgEvent::Exit(dim.id(), ExitEvent::ThreadDim));
                thread_dims.push(dim);
            },
            _ => {
                events.push(CfgEvent::Enter(dim.id(), EntryEvent::SeqDim(dim.kind())));
                events.push(CfgEvent::Exit(dim.id(), ExitEvent::SeqDim(dim)));
            },
        }
    }
    // Register thread and block induction levels.
    for dim in block_dims.iter_mut().chain(&mut thread_dims) {
        for level in dim.drain_induction_levels() {
            let event = EntryEvent::ParallelInductionLevel(level);
            events.push(CfgEvent::Enter(dim.id(), event));
        }
    }
    // Sort block dims.
    block_dims.sort_unstable_by(|lhs, rhs| {
        if lhs.id() == rhs.id() { return std::cmp::Ordering::Equal; }
        match space.domain().get_order(lhs.id().into(), rhs.id().into()) {
            Order::OUTER => std::cmp::Ordering::Less,
            Order::INNER => std::cmp::Ordering::Greater,
            _ => panic!("invalid order between block dim {:?} and {:?}",
                        lhs.id(), rhs.id()),
        }
    });
    // Sort and group thread dims.
    let mut merged_thread_dims = Vec::with_capacity(3);
    for dim in thread_dims {
        let pos = merged_thread_dims.binary_search_by(|probe: &Dimension| {
            if probe.id() == dim.id() { return std::cmp::Ordering::Equal; }
            match space.domain().get_thread_mapping(probe.id(), dim.id()) {
                ThreadMapping::MAPPED_OUT => std::cmp::Ordering::Less,
                ThreadMapping::MAPPED_IN => std::cmp::Ordering::Greater,
                ThreadMapping::MAPPED => std::cmp::Ordering::Equal,
                mapping => {
                    panic!("invalid mapping between thread dims {:?} and {:?}: {:?}",
                           probe.id(), dim.id(), mapping)
                }
            }
        });
        match pos {
            Ok(pos) => merged_thread_dims[pos].merge_from(dim),
            Err(pos) => merged_thread_dims.insert(pos, dim),
        }
    }
    // Register thread entering events.
    for (pos, dim) in merged_thread_dims.iter().enumerate() {
        for id in dim.dim_ids() {
            events.push(CfgEvent::Enter(id, EntryEvent::ThreadDim(pos)));
        }
    }
    (block_dims, merged_thread_dims, events)
}
