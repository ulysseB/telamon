use crate::codegen::{Dimension, Instruction};
use crate::ir;
use crate::search_space::*;
use itertools::Itertools;
use log::debug;
use std::{self, fmt};
use utils::unwrap;

/// Represents a CFG of the targeted device.
#[derive(Clone)]
pub enum Cfg<'a> {
    /// Represents the root node of the CFG.
    Root(Vec<Cfg<'a>>),
    /// Represents a loop in the CFG.
    ///
    /// Conceptually (except the loop could be unrolled), this expands to the following template:
    ///
    ///   index = 0;
    ///   ${prologue}
    ///   loop {
    ///     ${body}
    ///     index = index + 1;
    ///     if (index < size) {
    ///       ${advanced}
    ///     } else {
    ///       break;
    ///     }
    ///   }
    Loop {
        // The iteration dimension
        dimension: Dimension,
        // Size override
        size: Option<i32>,
        // Code to execute before entering the loop
        prologue: Vec<Cfg<'a>>,
        // Code to execute at each iteration of the loop
        body: Vec<Cfg<'a>>,
        // Code to execute with a predicate, for the next iteration of the loop
        advanced: Vec<Cfg<'a>>,
    },
    /// An instruction in the CFG, potentially vectorized on 2 levels.
    Instruction([Vec<Dimension>; 2], Instruction<'a>),
    /// Defines the set of active thread dimensions.
    Threads(Vec<Option<ir::DimId>>, Vec<Cfg<'a>>),
}

/// A helper struct to represent a sequential dimension's body.
///
/// The instructions and loops inside the body are split in two parts: in `body` are the regular
/// instructions as scheduled by Telamon; in `advanced` are the advanced instructions which need to
/// be duplicated in the prologue and scheduled (with a predicate to not be executed on the last
/// iteration) after the rest of the body.
struct SeqBody<'a> {
    prologue: Vec<Cfg<'a>>,
    body: Vec<Cfg<'a>>,
    advanced: Vec<Cfg<'a>>,
}

// Returns a pair (advanced, not_advanced)
fn split_prologue_cfgs<'a>(
    cfgs: Vec<Cfg<'a>>,
    dim_id: ir::DimId,
) -> (Vec<Cfg<'a>>, Vec<Cfg<'a>>) {
    let mut dim_advanced = Vec::new();
    let mut dim_not_advanced = Vec::new();

    for inner in cfgs {
        match inner {
            Cfg::Root(_) => unreachable!("cannot split root"),
            Cfg::Threads(dim_ids, body) => {
                let (body_advanced, body_not_advanced) =
                    split_prologue_cfgs(body, dim_id);
                if !body_advanced.is_empty() {
                    dim_advanced.push(Cfg::Threads(dim_ids.clone(), body_advanced));
                }

                dim_not_advanced.push(Cfg::Threads(dim_ids, body_not_advanced));
            }
            Cfg::Loop {
                dimension,
                size,
                prologue,
                body,
                advanced,
            } => {
                if dimension.is_advanced(dim_id) {
                    dim_advanced.push(Cfg::Loop {
                        dimension,
                        size,
                        prologue,
                        body,
                        advanced,
                    });
                } else {
                    let (prologue_advanced, prologue_not_advanced) =
                        split_prologue_cfgs(prologue, dim_id);
                    if !prologue_advanced.is_empty() {
                        dim_advanced.push(Cfg::Loop {
                            dimension: dimension.clone(),
                            size: Some(1),
                            prologue: Vec::new(),
                            body: prologue_advanced,
                            advanced: Vec::new(),
                        });
                    }
                    dim_not_advanced.push(Cfg::Loop {
                        dimension,
                        size,
                        prologue: prologue_not_advanced,
                        body,
                        advanced,
                    });
                }
            }
            Cfg::Instruction(vec_dims, inst) => {
                if inst.is_advanced(dim_id) {
                    dim_advanced.push(Cfg::Instruction(vec_dims, inst));
                } else {
                    dim_not_advanced.push(Cfg::Instruction(vec_dims, inst));
                }
            }
        }
    }

    (dim_advanced, dim_not_advanced)
}

// Returns a pair `(advanced, not_advanced)
fn split_body_cfgs<'a>(cfgs: Vec<Cfg<'a>>, dim_id: ir::DimId) -> SeqBody<'a> {
    let mut dim_prologue = Vec::new();
    let mut dim_advanced = Vec::new();
    let mut dim_body = Vec::new();

    for inner in cfgs {
        match inner {
            Cfg::Root(_) => unreachable!("cannot advance root"),
            Cfg::Threads(dim_ids, body) => {
                let split_body = split_body_cfgs(body, dim_id);

                if !split_body.prologue.is_empty() {
                    dim_prologue.push(Cfg::Threads(dim_ids.clone(), split_body.prologue));
                }

                if !split_body.advanced.is_empty() {
                    dim_advanced.push(Cfg::Threads(dim_ids.clone(), split_body.advanced));
                }

                // Ensure we always keep a non-advanced `Cfg::Threads`, even if empty, to prevent
                // `add_empty_threads` from messing with us.
                dim_body.push(Cfg::Threads(dim_ids, split_body.body));
            }
            Cfg::Instruction(vec_dims, inst) => {
                if inst.is_advanced(dim_id) {
                    dim_prologue.push(Cfg::Instruction(vec_dims.clone(), inst.clone()));
                    dim_advanced.push(Cfg::Instruction(vec_dims, inst));
                } else {
                    dim_body.push(Cfg::Instruction(vec_dims, inst));
                }
            }
            Cfg::Loop {
                dimension,
                size,
                prologue,
                body,
                advanced,
            } => {
                if dimension.is_advanced(dim_id) {
                    let loop_ = Cfg::Loop {
                        dimension,
                        size,
                        prologue,
                        body,
                        advanced,
                    };
                    dim_prologue.push(loop_.clone());
                    dim_advanced.push(loop_);
                } else {
                    let (prologue_advanced, prologue_not_advanced) =
                        split_prologue_cfgs(prologue, dim_id);
                    if !prologue_advanced.is_empty() {
                        dim_prologue.push(Cfg::Loop {
                            dimension: dimension.clone(),
                            size: Some(1),
                            prologue: Vec::new(),
                            body: prologue_advanced.clone(),
                            advanced: Vec::new(),
                        });
                        dim_advanced.extend(prologue_advanced);
                    }

                    dim_body.push(Cfg::Loop {
                        dimension,
                        size,
                        prologue: prologue_not_advanced,
                        body,
                        advanced,
                    });
                }
            }
        }
    }

    SeqBody {
        prologue: dim_prologue,
        body: dim_body,
        advanced: dim_advanced,
    }
}

impl<'a> Cfg<'a> {
    /// Iterates over the dimensions of the `Cfg`.
    pub fn dimensions(&self) -> impl Iterator<Item = &Dimension> {
        match self {
            Cfg::Root(body) | Cfg::Threads(_, body) => {
                Box::new(body.iter().flat_map(|cfg| cfg.dimensions()))
                    as Box<dyn Iterator<Item = _>>
            }
            Cfg::Loop {
                dimension,
                size: _size,
                prologue: _prologue,
                body,
                advanced,
            } => {
                let body_dims = body.iter().flat_map(|cfg| cfg.dimensions());
                let advanced_dims = advanced.iter().flat_map(|cfg| cfg.dimensions());
                Box::new(
                    std::iter::once(dimension)
                        .chain(body_dims)
                        .chain(advanced_dims),
                ) as _
            }
            Cfg::Instruction(dims, _) => Box::new(dims.iter().flatten()),
        }
    }

    /// Iterates over the instructions of the `Cfg`.
    pub fn instructions(&self) -> impl Iterator<Item = &Instruction<'a>> {
        match self {
            Cfg::Root(body) | Cfg::Threads(_, body) => {
                let iter = body.iter().flat_map(|cfg| cfg.instructions());
                Box::new(iter) as Box<dyn Iterator<Item = _>>
            }
            Cfg::Loop { prologue, body, .. } => {
                // Skip advance: the same instruction might be present multiple times in advance
                // for different loops (multi-level advance), but it will always be in either
                // exactly one prologue or exactly one body.
                let iter =
                    (prologue.iter().chain(body)).flat_map(|cfg| cfg.instructions());
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
                        let seq_body = split_body_cfgs(cfg, dim_id);

                        body.push(Cfg::Loop {
                            dimension: dim,
                            size: None,
                            prologue: seq_body.prologue,
                            body: seq_body.body,
                            advanced: seq_body.advanced,
                        });
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
                    Cfg::Loop {
                        dimension,
                        size,
                        prologue,
                        body,
                        advanced,
                    } => Cfg::Loop {
                        dimension,
                        size,
                        prologue: Self::add_empty_threads(prologue, num_thread_dims),
                        body: Self::add_empty_threads(body, num_thread_dims),
                        advanced: Self::add_empty_threads(advanced, num_thread_dims),
                    },
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
            Cfg::Root(ref body) | Cfg::Loop { ref body, .. } => {
                body.iter().any(|c| c.handle_threads())
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
    padding: &'a str,
}

impl<'a> IndentAdapter<'a> {
    /// Create a new [`IndentAdapter`].std
    ///
    /// # Notes
    ///
    /// This assumes that the adapter is created after a newline; indentation will be added to the
    /// first formatted value.
    fn new<'b: 'a>(fmt: &'a mut fmt::Formatter<'b>) -> Self {
        IndentAdapter::with_padding(fmt, "  ")
    }

    fn with_padding<'b: 'a>(fmt: &'a mut fmt::Formatter<'b>, padding: &'a str) -> Self {
        IndentAdapter {
            fmt,
            on_newline: true,
            padding,
        }
    }
}

impl fmt::Write for IndentAdapter<'_> {
    fn write_str(&mut self, mut s: &str) -> fmt::Result {
        while !s.is_empty() {
            if self.on_newline {
                self.fmt.write_str(self.padding)?;
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
            Cfg::Loop {
                dimension,
                size,
                prologue,
                body,
                advanced,
            } => {
                writeln!(
                    fmt,
                    "{:?}[{}{}]({:?}) {{",
                    dimension.kind(),
                    dimension.size(),
                    match size {
                        None => "".to_string(),
                        Some(size) => format!("={}", size),
                    },
                    dimension.dim_ids().format(" = ")
                )?;
                if !prologue.is_empty() {
                    writeln!(IndentAdapter::new(fmt), "prologue {{")?;
                    for inner in prologue {
                        writeln!(
                            IndentAdapter::with_padding(fmt, "    "),
                            "{}",
                            inner.display(fun)
                        )?;
                    }
                    writeln!(IndentAdapter::new(fmt), "}}")?;
                }

                for inner in body {
                    writeln!(IndentAdapter::new(fmt), "{}", inner.display(fun))?;
                }

                if !advanced.is_empty() {
                    writeln!(IndentAdapter::new(fmt), "advanced {{")?;
                    for inner in advanced {
                        writeln!(
                            IndentAdapter::with_padding(fmt, "    "),
                            "{}",
                            inner.display(fun)
                        )?;
                    }
                    writeln!(IndentAdapter::new(fmt), "}}")?;
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
            Cfg::Loop {
                dimension,
                size,
                prologue,
                body,
                advanced,
            } => f
                .debug_struct(&format!("{:?}", dimension.kind()))
                .field(
                    "dimension",
                    &format_args!("[{:?}]", &dimension.dim_ids().format(",")),
                )
                .field("size", size)
                .field("prologue", prologue)
                .field("body", body)
                .field("advanced", advanced)
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
