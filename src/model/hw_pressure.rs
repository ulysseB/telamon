//! Pressure on the hardware execution units.
use device::Device;
use ir;
use itertools::Itertools;
use model::{Level, CodePoint};
use search_space::{DimKind, Domain};
use std::{cmp, fmt, iter};
use std::rc::Rc;
use utils::*;

/// A lower bound on the execution time.
#[derive(Debug, Clone)]
pub struct ExplainedBound<ORIGIN> {
    value: f64,
    size: usize,
    origin: ORIGIN,
}

/// A lower bound on the execution time, fast to clone but relying on information internal
/// to the performance model.
pub type FastBound = ExplainedBound<Rc<FastOrigin>>;

/// A lower bound on the execution time, with a detailed explanation of the origin of the
/// bound.
pub type Bound = ExplainedBound<Origin>;

impl<ORIGIN> ExplainedBound<ORIGIN> {
    /// Returns the bound value.
    pub fn value(&self) -> f64 { self.value }

    /// Indicates if the bound should be used instead of another.
    pub fn is_better_than(&self, other: &ExplainedBound<ORIGIN>) -> bool {
        const F: f64 = 1.0+1.0e-6;
        if self.value > F*other.value { true }
        else if F*self.value < other.value { false }
        else { self.size < other.size }
    }
}

impl<T> cmp::Eq for ExplainedBound<T> {}

impl<T> cmp::PartialEq for ExplainedBound<T> {
    fn eq(&self, other: &Self) -> bool { self.value == other.value }
}

impl<T> cmp::Ord for ExplainedBound<T> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        unwrap!(self.partial_cmp(other))
    }
}

impl<T> cmp::PartialOrd for ExplainedBound<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl FastBound {
    /// An instantaneous dependency.
    pub fn zero() -> Self { FastBound::new(0f64, FastOrigin::Latency) }

    /// Creates a new latency with the given origin.
    fn new(value: f64, origin: FastOrigin) -> Self {
        assert!(!value.is_nan());
        FastBound { value, origin: Rc::new(origin), size: 1 }
    }

    /// Repeat the bound by iteration on a given loop level.
    pub fn iterate(self, iterations: u32, level: usize) -> Self {
        let origin = FastOrigin::Loop { iterations, level, inner: self.origin };
        FastBound {
            value: self.value * f64::from(iterations),
            origin: Rc::new(origin),
            size: self.size + 1,
        }
    }

    /// Chains two bounds. `self` and `other` must be a latencies to and from `mid_point`.
    pub fn chain(self, mid_point: usize, other: Self) -> Self {
        let value = self.value + other.value;
        let size = self.size + other.size;
        let origin = Rc::new(FastOrigin::Chain { before: self, mid_point, after: other });
        FastBound { value, origin, size }
    }

    /// Converts the bound so it can be explained without access to the performance model
    /// internals.
    pub fn explain(&self, device: &Device, levels: &[Level],
                   code_points: &[CodePoint]) -> Bound {
        let origin = self.origin.explain(device, levels, code_points).simplify().1;
        Bound { value: self.value, origin, size: self.size }
    }

    /// Scales the bound to account for a parallelism level.
    ///
    /// * `hw_parallelism` - the number of computations that can be performed in parallel.
    /// * `iterations` - the number of times the computation bounded by `self` is repeated.
    /// * `max_par` -  the maximum number of independent jobs that can be built to compute
    ///   `iterations` times the computation bounded by self.
    pub fn scale(self, hw_parallelism: u64, iterations: u64, max_par: u64) -> Self {
        let num_waves = div_ceil(max_par, hw_parallelism) as f64;
        let factor = num_waves * iterations as f64 / max_par as f64;
        let origin = Rc::new(FastOrigin::Scale { inner: self.origin, factor });
        FastBound { value: self.value * factor, origin, size: self.size+1 }
    }
}

impl Bound {
    /// Creates a bound from a measured execution time.
    pub fn from_actual_time(value: f64) -> Self {
        Bound { value, origin: Origin::HardwareEvaluation, size: 1 }
    }
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:.2e}ns because {}", self.value, self.origin)
    }
}

/// The justification behind a lower bound, based on information internal to the
/// performance model. Should not be used outside the model.
#[derive(Debug, Clone)]
pub enum FastOrigin {
    /// The bound is caused by the latency between two instructions.
    Latency,
    /// The bound is caused by the pressure on a bottleneck.
    Bottleneck(usize, BottleneckLevel),
    /// The bound is caused by a loop-carried dependency.
    Loop { level: usize, iterations: u32, inner: Rc<FastOrigin> },
    /// The bound is caused by a dependency chain.
    Chain { before: FastBound, mid_point: usize, after: FastBound },
    /// Scales the bound to account for a parallelism level.
    Scale { inner: Rc<FastOrigin>, factor: f64 }
}

/// The level at which a bottleneck is computed.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum BottleneckLevel { Global, Block, Thread }

impl BottleneckLevel {
    /// Indicates if a dimension should be taken into account for the bottleneck level.
    pub fn accounts_for_dim(&self, kind: DimKind) -> bool {
        match *self {
            _ if kind.intersects(DimKind::VECTOR) => false,
            BottleneckLevel::Global => true,
            _ if kind == DimKind::BLOCK => false,
            BottleneckLevel::Block => true,
            BottleneckLevel::Thread => !kind.intersects(DimKind::THREAD),
        }
    }
}

impl fmt::Display for BottleneckLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            BottleneckLevel::Global => write!(f, "global level"),
            BottleneckLevel::Block => write!(f, "block level"),
            BottleneckLevel::Thread => write!(f, "thread level"),
        }
    }
}

impl FastOrigin {
    /// Converts the origin so it does not use information internal to the performance
    /// model.
    fn explain(&self, device: &Device, levels: &[Level],
               code_points: &[CodePoint]) -> Origin {
        match *self {
            FastOrigin::Latency => Origin::Latency,
            FastOrigin::Bottleneck(id, level) =>
                Origin::Bottleneck(device.bottlenecks()[id], level),
            FastOrigin::Chain { ref before, mid_point, ref after } => {
                let mid_point = convert_point(levels, code_points[mid_point]);
                let before = Box::new(before.explain(device, levels, code_points));
                let after = Box::new(after.explain(device, levels, code_points));
                Origin::Chain { before, mid_point, after }
            },
            FastOrigin::Loop { level, iterations, ref inner } => {
                let inner = Box::new(inner.explain(device, levels, code_points));
                let dims = levels[level].dims.iter().cloned().collect();
                Origin::Loop { dims, iterations, inner }
            },
            FastOrigin::Scale { ref inner, factor } => {
                let inner = Box::new(inner.explain(device, levels, code_points));
                Origin::Scale { inner, factor }
            },
        }
    }
}

/// A `CodePoint`, but based on dimension ids rather than level ids.
#[derive(Clone)]
pub enum Point { Inst(ir::InstId), Entry(Vec<ir::DimId>), Exit(Vec<ir::DimId>) }

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Point::Inst(id) => write!(f, "instruction {}", id.0),
            Point::Entry(ref dims) =>
                write!(f, "the entry of dims [{}]", dims.iter().format(", ")),
            Point::Exit(ref dims) =>
                write!(f, "the exit of dims [{}]", dims.iter().format(", ")),
        }
    }
}

/// Converts a `CodePoint` based on level ids into a `Point`, based on dimension ids.
fn convert_point(levels: &[Level], point: CodePoint) -> Point {
    match point {
        CodePoint::Inst(id) => Point::Inst(id),
        CodePoint::LevelEntry(id) =>
            Point::Entry(levels[id].dims.iter().cloned().collect()),
        CodePoint::LevelExit(id) =>
            Point::Exit(levels[id].dims.iter().cloned().collect()),
    }
}

/// The justification behind a lower bound. Is slower to handle than `FastOrigin`
/// but does not refer internal information.
#[derive(Clone)]
pub enum Origin {
    /// The bound is caused by the latency between two instructions.
    Latency,
    /// The bound is caused by a bottleneck.
    Bottleneck(&'static str, BottleneckLevel),
    /// The bound is repeated in a loop.
    Loop { dims: Vec<ir::DimId>, iterations: u32, inner: Box<Origin> },
    /// The bound is caused by a dependency chain.
    Chain { before: Box<Bound>, mid_point: Point, after: Box<Bound> },
    /// The bound is scalled to account for a parallelism level.
    Scale { inner: Box<Origin>, factor: f64 },
    /// The bound was measured on hardware.
    HardwareEvaluation,
}

impl Origin {
    /// Simplifies the bound and indicates and previous and succedings bounds should be
    /// trimmed.
    fn simplify(self) -> (bool, Self, bool) {
        match self {
            x @ Origin::Latency |
            x @ Origin::Bottleneck(..) |
            x @ Origin::HardwareEvaluation => (false, x, false),
            Origin::Loop { iterations: 0, .. } => (true, Origin::Latency, true),
            Origin::Loop { dims, iterations, inner } => {
                let inner = Box::new(inner.simplify().1);
                (true, Origin::Loop { dims, iterations, inner }, true)
            },
            Origin::Scale { inner, factor } => {
                let inner = Box::new(inner.simplify().1);
                (false, Origin::Scale { inner, factor }, false)
            },
            Origin::Chain { before, mid_point, after } => {
                let after = *after;
                let before = *before;
                let (trim_preds, before_origin, trim_after) = before.origin.simplify();
                let (trim_before, after_origin, trim_succs) = after.origin.simplify();
                let trim_preds = trim_preds || before.value == 0f64;
                let trim_succs = trim_succs || after.value == 0f64;
                if trim_before && before.value == 0f64 {
                    (true, after_origin,  trim_succs)
                } else if trim_after && after.value == 0f64 {
                    (trim_preds, before_origin, true)
                } else {
                    let before = ExplainedBound {
                        value: before.value,
                        origin: before_origin,
                        size: before.size,
                    };
                    let after = ExplainedBound {
                        value: after.value,
                        origin: after_origin,
                        size: before.size,
                    };
                    let origin = Origin::Chain {
                        before: Box::new(before),
                        mid_point,
                        after: Box::new(after),
                    };
                    (trim_preds, origin, trim_succs)
                }
            },
        }
    }
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Origin::Latency => write!(f, "a dependency"),
            Origin::Bottleneck(name, level) =>
                write!(f, "the pressure on {} at the {}", name, level),
            Origin::HardwareEvaluation => write!(f, "the evaluation on the hardware"),
            Origin::Loop { ref dims, iterations, ref inner } => {
                write!(f, "{} iterations along dimensions [{}] of {{ {} }}",
                    iterations, dims.iter().format(", "), inner)
            },
            Origin::Scale { ref inner, factor } =>
                write!(f, "scale by {:.2e}: {}", factor, inner),
            Origin::Chain { ref before, ref mid_point, ref after } => {
                display_inline_chain(before, f)?;
                write!(f, " to {}, then ", mid_point)?;
                display_inline_chain(after, f)?;
                Ok(())
            },
        }
    }
}

/// Displays a bound. Only prints the origin if it is a chain.
fn display_inline_chain(bound: &Bound, f: &mut fmt::Formatter) -> fmt::Result {
    if let Origin::Chain { .. } = bound.origin {
        write!(f, "{}", bound.origin)
    } else {
        write!(f, "{:.2e}ns: {}", bound.value, bound.origin)
    }
}

/// The pressure on the hardware induced by a computation.
#[derive(Clone, Debug)]
pub struct HwPressure {
    latency: f64,
    bottlenecks: Vec<f64>,
}

impl HwPressure {
    /// Creates a new `Pressure`
    pub fn new(latency: f64, bottlenecks: Vec<f64>) -> Self {
        HwPressure { latency, bottlenecks }
    }

    /// Creates a null `Pressure` for the given device.
    pub fn zero(device: &Device) -> Self {
        HwPressure::new(0f64, device.bottlenecks().iter().map(|_| 0f64).collect())
    }

    /// Derive a bound on the execution time from the pressure on the hardware.
    pub fn bound(&self, level: BottleneckLevel, rates: &HwPressure) -> FastBound {
        let latency = FastBound::new(self.latency/rates.latency, FastOrigin::Latency);
        let bound = self.bottlenecks.iter().zip_eq(&rates.bottlenecks).enumerate()
            .map(|(id, (&value, &rate))| {
                FastBound::new(value/rate, FastOrigin::Bottleneck(id, level))
            }).chain(iter::once(latency)).max();
        unwrap!(bound)
    }

    /// Adds the pressure of another computation, performed in parallel.
    pub fn add_parallel(&mut self, other: &HwPressure) {
        self.latency = f64::max(self.latency, other.latency);
        for (&other, b) in other.bottlenecks.iter().zip_eq(&mut self.bottlenecks) {
            *b += other;
        }
    }

    /// Adds the pressure of another computation, performed sequentially.
    pub fn add_sequential(&mut self, other: &HwPressure) {
        self.latency += other.latency;
        for (&other, b) in other.bottlenecks.iter().zip_eq(&mut self.bottlenecks) {
            *b += other;
        }
    }

    /// Computes the pressure obtained by duplicating this one in parallel.
    pub fn repeat_parallel(&mut self, factor: f64) {
        for b in &mut self.bottlenecks { *b *= factor; }
    }

    /// Adds the pressure of another computation, repeated in parallel. Ignores the latency.
    pub fn repeat_and_add_bottlenecks(&mut self, factor: f64, other: &HwPressure) {
        for (&other, b) in other.bottlenecks.iter().zip_eq(&mut self.bottlenecks) {
            *b += other * factor;
        }
    }

    /// Computes the pressure obtained by repeating this one sequentially.
    pub fn repeat_sequential(&mut self, factor: f64) {
        self.latency *= factor;
        for b in &mut self.bottlenecks { *b *= factor; }
    }

    /// Take the minimum of `self` and `other` for each bottleneck.
    pub fn minimize(&mut self, other: &HwPressure) {
        self.latency = f64::min(self.latency, other.latency);
        for (&other, b) in other.bottlenecks.iter().zip_eq(&mut self.bottlenecks) {
            *b = f64::min(*b, other);
        }
    }

    /// Returns the pointwise minimum of a serie of `HwPressure`
    pub fn min<'a, IT>(mut it: IT) -> Option<Self> where IT: Iterator<Item=&'a Self> {
        it.next().cloned().map(|mut pressure| {
            for other in it { pressure.minimize(other) }
            pressure
        })
    }

    /// Returns the pressure on a bottleneck.
    #[cfg(test)]
    pub fn get_bottleneck(&self, index: usize) -> f64 { self.bottlenecks[index] }

    /// Pointwise multiplication of the pressure on each resource.
    pub fn multiply(&mut self, other: &HwPressure) {
        self.latency *= other.latency;
        for (b, &other_b) in self.bottlenecks.iter_mut().zip_eq(&other.bottlenecks) {
            *b *= other_b;
        }
    }
}
