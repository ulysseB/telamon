use crate::codegen::{Dimension, Instruction};
use crate::ir;
use crate::search_space::*;
use itertools::Itertools;
use log::debug;
use std::{self, fmt};
use utils::unwrap;

/// Represents a CFG of the targeted device.
pub enum Cfg<'a> {
    /// Represents the root node of the CFG.
    Root(Vec<Cfg<'a>>),
    /// Represents a loop in the CFG.
    Loop(Dimension, Vec<Cfg<'a>>),
    /// An instruction in the CFG, potentially vectorized on 2 levels.
    Instruction([Vec<Dimension>; 2], Instruction<'a>),
    /// Defines the set of active thread dimensions.
    Threads(Vec<Option<ir::DimId>>, Vec<Cfg<'a>>),
}

impl<'a> Cfg<'a> {
    /// Iterates over the dimensions of the `Cfg`.
    pub fn dimensions(&self) -> impl Iterator<Item = &Dimension> {
        match self {
            Cfg::Root(body) | Cfg::Threads(_, body) => {
                Box::new(body.iter().flat_map(|cfg| cfg.dimensions()))
                    as Box<dyn Iterator<Item = _>>
            }
            Cfg::Loop(dim, body) => {
                let body_dims = body.iter().flat_map(|cfg| cfg.dimensions());
                Box::new(std::iter::once(dim).chain(body_dims)) as _
            }
            Cfg::Instruction(dims, _) => Box::new(dims.iter().flatten()),
        }
    }

    /// Iterates over the instructions of the `Cfg`.
    pub fn instructions(&self) -> impl Iterator<Item = &Instruction<'a>> {
        match self {
            Cfg::Root(body) | Cfg::Loop(_, body) | Cfg::Threads(_, body) => {
                let iter = body.iter().flat_map(|cfg| cfg.instructions());
                Box::new(iter) as Box<dyn Iterator<Item = _>>
            }
            Cfg::Instruction(_, inst) => Box::new(std::iter::once(inst)) as _,
        }
    }

    /// Creates a vector instruction from a list of events.
    fn vector_inst_from_events<IT>(
        dim: Dimension,
        events: &mut std::iter::Peekable<IT>,
    ) -> Self
    where
        IT: Iterator<Item = CfgEvent<'a>>,
    {
        fn get_level(kind: DimKind) -> usize {
            match kind {
                DimKind::OUTER_VECTOR => 0,
                DimKind::INNER_VECTOR => 1,
                kind => panic!("expected a VECTOR mapping decision, got {:?}", kind),
            }
        }
        let mut dims = [vec![], vec![]];
        dims[get_level(dim.kind())].push(dim);
        // Pop dimensions entry points until we reach the instruction.
        let inst = loop {
            match events.next().unwrap() {
                CfgEvent::Exec(inst) => break inst,
                CfgEvent::Enter(_, EntryEvent::SeqDim(dim)) => {
                    dims[get_level(dim.kind())].push(dim);
                }
                event => panic!("unexpected event {:?}", event),
            }
        };
        // Pop dimensions exit points.
        for _ in dims.iter().flatten() {
            match events.next().unwrap() {
                CfgEvent::Exit(_, ExitEvent::SeqDim) => (),
                event => panic!("unexpected event {:?}", event),
            }
        }
        Cfg::Instruction(dims, inst)
    }

    /// Builds a CFG from a list of `CfgEvent`.
    fn body_from_events<IT>(
        events: &mut std::iter::Peekable<IT>,
        num_thread_dims: usize,
    ) -> Vec<Cfg<'a>>
    where
        IT: Iterator<Item = CfgEvent<'a>>,
    {
        use self::CfgEvent::*;
        let mut body = vec![];
        while let Some(event) = events.next() {
            match event {
                Exec(inst) => body.push(Cfg::Instruction(Default::default(), inst)),
                Enter(dim_id, EntryEvent::SeqDim(dim)) => {
                    assert_eq!(dim_id, dim.id());

                    if dim.kind().is(DimKind::VECTOR).as_bool().unwrap() {
                        body.push(Cfg::vector_inst_from_events(dim, events));
                    } else {
                        let cfg = Cfg::body_from_events(events, num_thread_dims);
                        body.push(Cfg::Loop(dim, cfg))
                    }
                }
                Exit(_, ExitEvent::SeqDim) => break,
                Enter(dim_id, EntryEvent::ThreadDim(pos)) => {
                    let mut dim_ids = vec![None; num_thread_dims];
                    dim_ids[pos] = Some(dim_id);
                    while let Some(Enter(_, EntryEvent::ThreadDim(..))) = events.peek() {
                        let next = unwrap!(events.next());
                        if let Enter(dim_id, EntryEvent::ThreadDim(pos)) = next {
                            assert_eq!(dim_ids[pos], None);

                            dim_ids[pos] = Some(dim_id);
                        } else {
                            unreachable!()
                        };
                    }
                    let inner = Cfg::body_from_events(events, 0);
                    body.push(Cfg::Threads(dim_ids, inner));
                }
                Exit(_, ExitEvent::ThreadDim) => {
                    while let Some(Exit(_, ExitEvent::ThreadDim)) = events.peek() {
                        events.next();
                    }
                    break;
                }
            }
        }
        body
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
                    Cfg::Root(inner) => {
                        Cfg::Root(Self::add_empty_threads(inner, num_thread_dims))
                    }
                    Cfg::Loop(dim, inner) => {
                        Cfg::Loop(dim, Self::add_empty_threads(inner, num_thread_dims))
                    }
                    cfg => cfg,
                }))
            } else {
                let thread_dims = vec![None; num_thread_dims];
                new_body.push(Cfg::Threads(thread_dims, cfgs.collect()));
            }
        }
        new_body
    }

    /// Indicates if the `Cfg` handles thread parallelism.
    fn handle_threads(&self) -> bool {
        match *self {
            Cfg::Root(ref inners) | Cfg::Loop(_, ref inners) => {
                inners.iter().any(|c| c.handle_threads())
            }
            Cfg::Threads(..) => true,
            Cfg::Instruction(..) => false,
        }
    }
}

/// A struct to indent on new lines when writing formatting traits.
///
/// This is inspired from the [`PadAdapter`] from the standard library, but without using private
/// [`std::fmt::Formatter`] APIs.
///
/// [`PadAdapter`]: https://github.com/rust-lang/rust/blob/316a391dcb7d66dc25f1f9a4ec9d368ef7615005/src/libcore/fmt/builders.rs
struct IndentAdapter<'a> {
    fmt: &'a mut (dyn fmt::Write + 'a),
    on_newline: bool,
}

impl<'a> IndentAdapter<'a> {
    /// Create a new [`IndentAdapter`].std
    ///
    /// # Notes
    ///
    /// This assumes that the adapter is created after a newline; indentation will be added to the
    /// first formatted value.
    fn new<'b: 'a>(fmt: &'a mut fmt::Formatter<'b>) -> Self {
        IndentAdapter {
            fmt,
            on_newline: true,
        }
    }
}

impl fmt::Write for IndentAdapter<'_> {
    fn write_str(&mut self, mut s: &str) -> fmt::Result {
        while !s.is_empty() {
            if self.on_newline {
                self.fmt.write_str("  ")?;
            }

            let split = match s.find('\n') {
                Some(pos) => {
                    self.on_newline = true;
                    pos + 1
                }
                None => {
                    self.on_newline = false;
                    s.len()
                }
            };

            self.fmt.write_str(&s[..split])?;
            s = &s[split..];
        }

        Ok(())
    }
}

impl ir::IrDisplay for Cfg<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter, fun: &ir::Function) -> fmt::Result {
        use fmt::Write;

        match self {
            Cfg::Root(inners) => {
                for inner in inners {
                    writeln!(fmt, "{}", inner.display(fun))?;
                }
            }
            Cfg::Loop(dim, inners) => {
                writeln!(
                    fmt,
                    "{:?}[{}]({:?}) {{",
                    dim.kind(),
                    dim.size(),
                    dim.dim_ids().format(" = ")
                )?;
                for inner in inners {
                    writeln!(IndentAdapter::new(fmt), "{}", inner.display(fun))?;
                }
                write!(fmt, "}}")?;
            }
            Cfg::Instruction([outer, inner], inst) => {
                if !outer.is_empty() {
                    assert!(
                        outer.iter().all(|d| d.kind() == DimKind::OUTER_VECTOR),
                        "Expected OUTER_VECTOR but found {:?}",
                        outer.iter().map(Dimension::kind).format(", "),
                    );
                    write!(fmt, "v")?;
                    for d in &*outer {
                        write!(
                            fmt,
                            "{}({})",
                            d.size().as_int().unwrap(),
                            d.dim_ids().format(" = ")
                        )?;
                    }
                }

                if !inner.is_empty() {
                    assert!(
                        inner.iter().all(|d| d.kind() == DimKind::INNER_VECTOR),
                        "Expected INNER_VECTOR but found {:?}",
                        inner.iter().map(Dimension::kind).format(", "),
                    );
                    write!(fmt, "v")?;
                    for d in &*inner {
                        write!(
                            fmt,
                            "{}({})",
                            d.size().as_int().unwrap(),
                            d.dim_ids().format(" = ")
                        )?;
                    }
                }

                write!(fmt, "{}", inst.ir_instruction().display(fun))?;
            }
            Cfg::Threads(dims, inners) => {
                writeln!(
                    fmt,
                    "THREAD[{}] {{",
                    dims.iter()
                        .map(|d| match d {
                            None => "_".to_string(),
                            Some(d) => format!("{:?}", d),
                        })
                        .format(", ")
                )?;
                for inner in inners {
                    writeln!(IndentAdapter::new(fmt), "{}", inner.display(fun))?;
                }
                write!(fmt, "}}")?;
            }
        }

        Ok(())
    }
}

impl<'a> fmt::Debug for Cfg<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Cfg::Root(inners) => f.debug_list().entries(inners).finish(),
            Cfg::Loop(dim, inners) => f
                .debug_tuple(&format!("{:?}", dim.kind()))
                .field(&format_args!("[{:?}]", &dim.dim_ids().format(",")))
                .field(inners)
                .finish(),
            Cfg::Instruction(dims, inst) => write!(f, "{:?} {}", dims, inst),
            Cfg::Threads(dims, inners) => {
                f.debug_tuple("Threads").field(dims).field(inners).finish()
            }
        }
    }
}

/// Builds the CFG from the list of dimensions and instructions. Also returns the list of
/// thread and block dimensions.
pub fn build<'a>(
    space: &'a SearchSpace,
    insts: Vec<Instruction<'a>>,
    dims: Vec<Dimension>,
) -> (Vec<Dimension>, Vec<Dimension>, Cfg<'a>) {
    let (block_dims, thread_dims, mut events) = gen_events(space, insts, dims);
    events.sort_by(|lhs, rhs| lhs.cmp(rhs, space));
    debug!("events: {:?}", events);
    let cfg = Cfg::from_events(events, thread_dims.len());
    (block_dims, thread_dims, cfg)
}

/// Describes the program points encountered when walking a CFG.
enum CfgEvent<'a> {
    Exec(Instruction<'a>),
    Enter(ir::DimId, EntryEvent),
    Exit(ir::DimId, ExitEvent),
}

/// An event to process when entering a dimension.
enum EntryEvent {
    /// Enter a sequential dimension.
    SeqDim(Dimension),
    /// Enter a thread dimension.
    ThreadDim(usize),
}

/// An event to process when exiting a dimension.
enum ExitEvent {
    SeqDim,
    ThreadDim,
}

impl<'a> CfgEvent<'a> {
    /// Indiciates the order of `self` with regards to `other`.
    fn cmp(&self, other: &CfgEvent, space: &SearchSpace) -> std::cmp::Ordering {
        let (lhs_stmt, rhs_stmt) = (self.stmt_id(), other.stmt_id());
        if lhs_stmt == rhs_stmt {
            self.cmp_within_stmt(other)
        } else {
            use self::CfgEvent::*;
            use std::cmp::Ordering::*;
            let order = space.domain().get_order(lhs_stmt, rhs_stmt);
            match (self, other, order) {
                (_, _, Order::MERGED) => self.cmp_within_stmt(other),
                (&Exec(_), &Exec(_), Order::ORDERED) => lhs_stmt.cmp(&rhs_stmt),
                (_, _, Order::BEFORE)
                | (&Enter(..), _, Order::OUTER)
                | (_, &Exit(..), Order::INNER) => Less,
                (_, _, Order::AFTER)
                | (_, &Enter(..), Order::INNER)
                | (&Exit(..), _, Order::OUTER) => Greater,
                (lhs, rhs, ord) => {
                    panic!("Invalid order between {:?} and {:?}: {:?}.", lhs, rhs, ord)
                }
            }
        }
    }

    /// Indicates the order of `self with `other`, assuming they are events on the same
    /// basic block.
    fn cmp_within_stmt(&self, other: &CfgEvent) -> std::cmp::Ordering {
        use self::CfgEvent::*;
        match (self, other) {
            (&Enter(..), &Enter(..))
            | (&Exec(..), &Exec(..))
            | (&Exit(..), &Exit(..)) => std::cmp::Ordering::Equal,
            (&Enter(..), _) | (_, &Exit(..)) => std::cmp::Ordering::Less,
            (&Exit(..), _) | (_, &Enter(..)) => std::cmp::Ordering::Greater,
        }
    }

    /// Returns an id of a `Statement` mentioned by the event.
    fn stmt_id(&self) -> ir::StmtId {
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
fn gen_events<'a>(
    space: &'a SearchSpace,
    insts: Vec<Instruction<'a>>,
    dims: Vec<Dimension>,
) -> (Vec<Dimension>, Vec<Dimension>, Vec<CfgEvent<'a>>) {
    let mut block_dims = Vec::new();
    let mut thread_dims = Vec::new();
    let mut events = insts.into_iter().map(CfgEvent::Exec).collect_vec();
    // Create dimension events and sort thread and block dims.
    for dim in dims {
        match dim.kind() {
            DimKind::BLOCK => block_dims.push(dim),
            DimKind::THREAD => {
                events.push(CfgEvent::Exit(dim.id(), ExitEvent::ThreadDim));
                thread_dims.push(dim);
            }
            _ => {
                events.push(CfgEvent::Exit(dim.id(), ExitEvent::SeqDim));
                events.push(CfgEvent::Enter(dim.id(), EntryEvent::SeqDim(dim)));
            }
        }
    }
    // Sort block dims.
    block_dims.sort_unstable_by(|lhs, rhs| {
        space
            .nesting_order(lhs.id())
            .partial_cmp(&rhs.id())
            .unwrap_or_else(|| {
                panic!(
                    "invalid order between block dim {:?} and {:?}",
                    lhs.id(),
                    rhs.id()
                )
            })
    });
    // Sort and group thread dims.
    let mut sorted_thread_dims = Vec::with_capacity(3);
    for dim in thread_dims {
        let pos = sorted_thread_dims.binary_search_by(|probe: &Vec<Dimension>| {
            if probe[0].id() == dim.id() {
                return std::cmp::Ordering::Equal;
            }
            match space.domain().get_thread_mapping(probe[0].id(), dim.id()) {
                ThreadMapping::MAPPED_OUT => std::cmp::Ordering::Less,
                ThreadMapping::MAPPED_IN => std::cmp::Ordering::Greater,
                ThreadMapping::MAPPED => std::cmp::Ordering::Equal,
                mapping => panic!(
                    "invalid mapping between thread dims {:?} and {:?}: {:?}",
                    probe[0].id(),
                    dim.id(),
                    mapping
                ),
            }
        });
        match pos {
            Ok(pos) => sorted_thread_dims[pos].push(dim),
            Err(pos) => sorted_thread_dims.insert(pos, vec![dim]),
        }
    }
    // Register thread entering events.
    let thread_dims = sorted_thread_dims
        .into_iter()
        .enumerate()
        .map(|(pos, mut dims)| {
            for dim in &mut dims {
                let event = EntryEvent::ThreadDim(pos);
                events.push(CfgEvent::Enter(dim.id(), event));
            }
            unwrap!(dims.into_iter().fold1(|mut x, y| {
                x.merge_from(y);
                x
            }))
        })
        .collect();
    (block_dims, thread_dims, events)
}
