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
    /// Defines the set of active thread dimensions.
    Threads(Vec<bool>, Vec<InductionLevel<'a>>, Vec<Cfg<'a>>),
}

impl<'a> Cfg<'a> {
    /// Iterates over the dimensions of the `Cfg`.
    pub fn dimensions(&self) -> impl Iterator<Item=&Dimension<'a>> {
        match *self {
            Cfg::Root(ref body) | Cfg::Threads(_, _, ref body) =>
                Box::new(body.iter().flat_map(|cfg| cfg.dimensions()))
                    as Box<Iterator<Item=_>>,
            Cfg::Loop(ref dim, ref body) => {
                let body_dims = body.iter().flat_map(|cfg| cfg.dimensions());
                Box::new(std::iter::once(dim).chain(body_dims)) as _
            },
            Cfg::Instruction(..) => Box::new(std::iter::empty()) as _,
        }
    }

    /// Iterates over the instructions of the `Cfg`.
    pub fn instructions(&self) -> impl Iterator<Item=&Instruction<'a>> {
        match *self {
            Cfg::Root(ref body) |
            Cfg::Loop(_, ref body) |
            Cfg::Threads(_, _, ref body) => {
                let iter = body.iter().flat_map(|cfg| cfg.instructions());
                Box::new(iter) as Box<Iterator<Item=_>>
            }
            Cfg::Instruction(ref inst) => Box::new(std::iter::once(inst)) as _,
        }
    }

    /// Iterates over the induction levels in the `Cfg`.
    pub fn induction_levels(&self) -> impl Iterator<Item=&InductionLevel<'a>> {
        match *self {
            Cfg::Threads(_, ref ind_levels, ref body) => {
                let levels = body.iter().flat_map(|c| c.induction_levels())
                    .chain(ind_levels);
                Box::new(levels) as Box<Iterator<Item=_>>
            }
            Cfg::Root(ref body) =>
                Box::new(body.iter().flat_map(|c| c.induction_levels())), 
            Cfg::Loop(ref dim, ref body) =>
                Box::new(body.iter().flat_map(|c| c.induction_levels())
                         .chain(dim.induction_levels())), 
            Cfg::Instruction(..) => Box::new(std::iter::empty())
        }
    }

    /// Builds a CFG from a list of `CfgEvent`.
    fn body_from_events<IT>(events: &mut std::iter::Peekable<IT>,
                            num_thread_dims: usize) -> Vec<Cfg<'a>>
        where IT: Iterator<Item=CfgEvent<'a>>
    {
        use self::CfgEvent::*;
        let mut body = vec![];
        while let Some(event) = events.next() {
            match event {
                Exec(inst) => body.push(Cfg::Instruction(inst)),
                Enter(_, EntryEvent::SeqDim(dim)) => {
                    let cfg = Cfg::body_from_events(events, num_thread_dims);
                    body.push(Cfg::Loop(dim, cfg))
                }
                Exit(_, ExitEvent::SeqDim) => break,
                Enter(_, EntryEvent::ThreadDim(pos, mut ind_levels)) => {
                    let mut dim_poses = vec![false; num_thread_dims];
                    dim_poses[pos] = true;
                    while let Some(Enter(_, EntryEvent::ThreadDim(..))) = events.peek() {
                        let next = unwrap!(events.next());
                        if let Enter(_, EntryEvent::ThreadDim(pos, levels)) = next {
                            dim_poses[pos] = true;
                            ind_levels.extend(levels);
                        } else { unreachable!() };
                    }
                    let inner = Cfg::body_from_events(events, 0);
                    body.push(Cfg::Threads(dim_poses, ind_levels, inner));
                }
                Exit(_, ExitEvent::ThreadDim) => {
                    while let Some(Exit(_, ExitEvent::ThreadDim)) = events.peek() {
                        events.next();
                    }
                    break;
                }
            }
        }
        return body;
    }

    /// Builds a CFG from a list of `CfgEvent`.
    fn from_events(events: Vec<CfgEvent<'a>>, num_thread_dims: usize) -> Cfg<'a> {
        let mut events = events.into_iter().peekable();
        let body = Cfg::body_from_events(&mut events, num_thread_dims);
        assert!(events.next().is_none());
        let body = Self::add_empty_threads(body, num_thread_dims);
        Cfg::Root(body)
    }

    /// Ensure every instruction is nested in a thread dimension.
    fn add_empty_threads(body: Vec<Cfg>, num_thread_dims: usize) -> Vec<Cfg> {
        let groups = body.into_iter().group_by(|cfg| cfg.handle_threads());
        let mut new_body = Vec::new();
        for (handle_threads, cfgs) in &groups {
            if handle_threads {
                new_body.extend(cfgs.map(|cfg| match cfg {
                    Cfg::Root(inner) =>
                        Cfg::Root(Self::add_empty_threads(inner, num_thread_dims)),
                    Cfg::Loop(dim, inner) =>
                        Cfg::Loop(dim, Self::add_empty_threads(inner, num_thread_dims)),
                    cfg => cfg,
                }))
            } else {
                let thread_dims = vec![false; num_thread_dims];
                new_body.push(Cfg::Threads(thread_dims, vec![], cfgs.collect()));
            }
        }
        new_body
    }

    /// Indicates if the `Cfg` handles thread parallelism.
    fn handle_threads(&self) -> bool {
        match *self {
            Cfg::Root(ref inners) |
            Cfg::Loop(_, ref inners) => inners.iter().any(|c| c.handle_threads()),
            Cfg::Threads(..) => true,
            Cfg::Instruction(..) => false,
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
            Cfg::Threads(ref dims, _, ref inners) =>
                write!(f, "threads({:?}, {:?})", dims, inners),
        }
    }
}

/// Builds the CFG from the list of dimensions and instructions. Also returns the list of
/// thread and block dimensions.
pub fn build<'a>(space: &'a SearchSpace<'a>,
                 insts: Vec<Instruction<'a>>,
                 dims: Vec<Dimension<'a>>)
    -> (Vec<Dimension<'a>>, Vec<Dimension<'a>>, Cfg<'a>)
{
    let (block_dims, thread_dims, mut events) = gen_events(space, insts, dims);
    events.sort_by(|lhs, rhs| lhs.cmp(rhs, space));
    let cfg = Cfg::from_events(events, thread_dims.len());
    (block_dims, thread_dims, cfg)
}

/// Describes the program points encountered when walking a CFG.
enum CfgEvent<'a> {
    Exec(Instruction<'a>),
    Enter(ir::DimId, EntryEvent<'a>),
    Exit(ir::DimId, ExitEvent),
}

/// An event to process when entering a dimension.
enum EntryEvent<'a> {
    /// Enter a sequential dimension.
    SeqDim(Dimension<'a>),
    /// Enter a thread dimension.
    ThreadDim(usize, Vec<InductionLevel<'a>>),
}

/// An event to process when exiting a dimension.
enum ExitEvent { SeqDim, ThreadDim }

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
                events.push(CfgEvent::Exit(dim.id(), ExitEvent::SeqDim));
                events.push(CfgEvent::Enter(dim.id(), EntryEvent::SeqDim(dim)));
            },
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
    let mut sorted_thread_dims = Vec::with_capacity(3);
    for dim in thread_dims {
        let pos = sorted_thread_dims.binary_search_by(|probe: &Vec<Dimension>| {
            if probe[0].id() == dim.id() { return std::cmp::Ordering::Equal; }
            match space.domain().get_thread_mapping(probe[0].id(), dim.id()) {
                ThreadMapping::MAPPED_OUT => std::cmp::Ordering::Less,
                ThreadMapping::MAPPED_IN => std::cmp::Ordering::Greater,
                ThreadMapping::MAPPED => std::cmp::Ordering::Equal,
                mapping => {
                    panic!("invalid mapping between thread dims {:?} and {:?}: {:?}",
                           probe[0].id(), dim.id(), mapping)
                }
            }
        });
        match pos {
            Ok(pos) => sorted_thread_dims[pos].push(dim),
            Err(pos) => sorted_thread_dims.insert(pos, vec![dim]),
        }
    }
    // Register thread entering events.
    let thread_dims = sorted_thread_dims.into_iter().enumerate().map(|(pos, mut dims)| {
        for dim in &mut dims {
            let event = EntryEvent::ThreadDim(pos, dim.drain_induction_levels());
            events.push(CfgEvent::Enter(dim.id(), event));
        }
        unwrap!(dims.into_iter().fold1(|mut x, y| { x.merge_from(y); x }))
    }).collect();
    (block_dims, thread_dims, events)
}
