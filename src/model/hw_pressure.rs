//! Pressure on the hardware execution units.
use crate::device::Device;
use crate::ir;
use crate::model::{
    size::{self, SymbolicFloat},
    CodePoint, Level,
};
use crate::search_space::{DimKind, Domain};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::rc::Rc;
use std::{cmp, fmt, iter};

use sym::Range;
use utils::*;

/// A lower bound on the execution time.
#[derive(Debug, Clone)]
pub struct ExplainedBound<ORIGIN> {
    value: SymbolicFloat,
    size: usize,
    origin: ORIGIN,
}

/// A lower bound on the execution time, fast to clone but relying on information internal
/// to the performance model.
pub type FastBound = ExplainedBound<Rc<FastOrigin>>;

/// A lower bound on the execution time, with a detailed explanation of the origin of the
/// bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bound {
    value: f64,
    pub lol: String,
    size: usize,
    origin: Origin,
}

impl cmp::Eq for Bound {}

impl cmp::PartialEq for Bound {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl cmp::Ord for Bound {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        unwrap!(self.partial_cmp(other))
    }
}

impl cmp::PartialOrd for Bound {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl FastBound {
    /// An instantaneous dependency.
    pub fn zero() -> Self {
        FastBound::new(0f64.into(), FastOrigin::Latency)
    }

    pub(super) fn value(&self) -> &SymbolicFloat {
        &self.value
    }

    /// Creates a new latency with the given origin.
    fn new(value: SymbolicFloat, origin: FastOrigin) -> Self {
        // TODO: assert!(!value.is_nan());
        FastBound {
            value,
            origin: Rc::new(origin),
            size: 1,
        }
    }

    /// Repeat the bound by iteration on a given loop level.
    // TODO: iterations: SymbolicFloat
    pub fn iterate(self, iterations: size::SymbolicFloat, level: usize) -> Self {
        let origin = FastOrigin::Loop {
            iterations: iterations.min_value() as u64, // TODO: be more precise?
            level,
            inner: self.origin,
        };
        FastBound {
            value: self.value * iterations,
            origin: Rc::new(origin),
            size: self.size + 1,
        }
    }

    /// Chains two bounds. `self` and `other` must be a latencies to and from `mid_point`.
    pub fn chain(self, mid_point: usize, other: Self) -> Self {
        let value = &self.value + &other.value;
        let size = self.size + other.size;
        let origin = Rc::new(FastOrigin::Chain {
            before: self,
            mid_point,
            after: other,
        });
        FastBound {
            value,
            origin,
            size,
        }
    }

    /// Converts the bound so it can be explained without access to the performance model
    /// internals.
    pub fn explain(
        &self,
        device: &dyn Device,
        levels: &[Level],
        code_points: &[CodePoint],
    ) -> Bound {
        let origin = self
            .origin
            .explain(device, levels, code_points)
            .simplify()
            .1;
        Bound {
            value: self.value.min_value(), // TODO:?
            lol: "".to_string(), // self.value.to_string(),   // format!("{}", self.value),
            origin,
            size: self.size,
        }
    }

    /// Scales the bound to account for a parallelism level.
    ///
    /// * `hw_parallelism` - the number of computations that can be performed in parallel.
    /// * `iterations` - the number of times the computation bounded by `self` is repeated.
    /// * `max_par` -  the maximum number of independent jobs that can be built to compute
    ///   `iterations` times the computation bounded by self.
    pub fn scale(
        self,
        hw_parallelism: u64,
        // TODO(sym): SymbolicFloat
        iterations: &size::Min,
        max_par: &size::Lcm,
    ) -> Self {
        assert!(hw_parallelism < u64::from(u32::max_value()));

        let max_par = size::SymbolicInt::from(max_par.clone());
        let num_waves = size::SymbolicFloat::div_ceil(&max_par, hw_parallelism as u32);
        // TODO(sym): let factor = num_waves * iterations / max_par.to_symbolic_float()
        // -> div_ceil_inv_magic(max_par, hw_parallelism) * iterations
        let factor = (num_waves * iterations.to_symbolic_float()) / max_par;
        let origin = Rc::new(FastOrigin::Scale {
            inner: self.origin,
            factor: factor.min_value(), // TODO: improve
        });
        FastBound {
            value: self.value * &factor,
            origin,
            size: self.size + 1,
        }
    }

    pub fn max_assign(&mut self, other: FastBound) {
        self.value.max_assign(&other.value);
        // TODO: explanation!!
        // TODO
        /*
        if other.is_better_than(self) {
            *self = other;
        }
        */
    }

    pub fn min_assign(&mut self, other: FastBound) {
        self.value.min_assign(&other.value);
        // TODO
        /*
        if self.is_better_than(&other) {
            *self = other;
        }
        */
    }

    pub fn min<I>(mut iter: I) -> Option<Self>
    where
        I: Iterator<Item = Self>,
    {
        if let Some(mut result) = iter.next() {
            for elem in iter {
                result.min_assign(elem);
            }
            Some(result)
        } else {
            None
        }
    }

    pub fn max<I>(mut iter: I) -> Option<Self>
    where
        I: Iterator<Item = Self>,
    {
        if let Some(mut result) = iter.next() {
            for elem in iter {
                result.max_assign(elem);
            }
            Some(result)
        } else {
            None
        }
    }

    /// Indicates if the bound should be used instead of another.
    fn is_better_than(&self, other: &FastBound) -> bool {
        unimplemented!()
        /*
        const F: f64 = 1.0 + 1.0e-6;
        if self.value > F * other.value {
            true
        } else if F * self.value < other.value {
            false
        } else {
            self.size < other.size
        }
        */
    }
}

impl Bound {
    /// Creates a bound from a measured execution time.
    pub fn from_actual_time(value: f64) -> Self {
        Bound {
            value,
            lol: format!("{}", value),
            origin: Origin::HardwareEvaluation,
            size: 1,
        }
    }

    /// Returns the bound value.
    pub fn value(&self) -> f64 {
        self.value
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
    Loop {
        level: usize,
        iterations: u64,
        inner: Rc<FastOrigin>,
    },
    /// The bound is caused by a dependency chain.
    Chain {
        before: FastBound,
        mid_point: usize,
        after: FastBound,
    },
    /// Scales the bound to account for a parallelism level.
    Scale { inner: Rc<FastOrigin>, factor: f64 },
}

/// The level at which a bottleneck is computed.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BottleneckLevel {
    Global,
    Block,
    Thread,
}

impl BottleneckLevel {
    /// Indicates if a dimension should be taken into account for the bottleneck level.
    pub fn accounts_for_dim(self, kind: DimKind) -> bool {
        match self {
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
    fn explain(
        &self,
        device: &dyn Device,
        levels: &[Level],
        code_points: &[CodePoint],
    ) -> Origin {
        match *self {
            FastOrigin::Latency => Origin::Latency,
            FastOrigin::Bottleneck(id, level) => {
                Origin::Bottleneck(device.bottlenecks()[id].into(), level)
            }
            FastOrigin::Chain {
                ref before,
                mid_point,
                ref after,
            } => {
                let mid_point = convert_point(levels, code_points[mid_point]);
                let before = Box::new(before.explain(device, levels, code_points));
                let after = Box::new(after.explain(device, levels, code_points));
                Origin::Chain {
                    before,
                    mid_point,
                    after,
                }
            }
            FastOrigin::Loop {
                level,
                iterations,
                ref inner,
            } => {
                let inner = Box::new(inner.explain(device, levels, code_points));
                let dims = levels[level].dims.iter().cloned().collect();
                Origin::Loop {
                    dims,
                    iterations,
                    inner,
                }
            }
            FastOrigin::Scale { ref inner, factor } => {
                let inner = Box::new(inner.explain(device, levels, code_points));
                Origin::Scale { inner, factor }
            }
        }
    }
}

/// A `CodePoint`, but based on dimension ids rather than level ids.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Point {
    Inst(ir::InstId),
    Entry(Vec<ir::DimId>),
    Exit(Vec<ir::DimId>),
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Point::Inst(id) => write!(f, "instruction {}", id.0),
            Point::Entry(ref dims) => {
                write!(f, "the entry of dims [{}]", dims.iter().format(", "))
            }
            Point::Exit(ref dims) => {
                write!(f, "the exit of dims [{}]", dims.iter().format(", "))
            }
        }
    }
}

/// Converts a `CodePoint` based on level ids into a `Point`, based on dimension ids.
fn convert_point(levels: &[Level], point: CodePoint) -> Point {
    match point {
        CodePoint::Inst(id) => Point::Inst(id),
        CodePoint::LevelEntry(id) => {
            Point::Entry(levels[id].dims.iter().cloned().collect())
        }
        CodePoint::LevelExit(id) => {
            Point::Exit(levels[id].dims.iter().cloned().collect())
        }
    }
}

/// The justification behind a lower bound. Is slower to handle than `FastOrigin`
/// but does not refer internal information.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Origin {
    /// The bound is caused by the latency between two instructions.
    Latency,
    /// The bound is caused by a bottleneck.
    Bottleneck(Cow<'static, str>, BottleneckLevel),
    /// The bound is repeated in a loop.
    Loop {
        dims: Vec<ir::DimId>,
        iterations: u64,
        inner: Box<Origin>,
    },
    /// The bound is caused by a dependency chain.
    Chain {
        before: Box<Bound>,
        mid_point: Point,
        after: Box<Bound>,
    },
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
            x @ Origin::Latency
            | x @ Origin::Bottleneck(..)
            | x @ Origin::HardwareEvaluation => (false, x, false),
            Origin::Loop { iterations: 0, .. } => (true, Origin::Latency, true),
            Origin::Loop {
                dims,
                iterations,
                inner,
            } => {
                let inner = Box::new(inner.simplify().1);
                (
                    true,
                    Origin::Loop {
                        dims,
                        iterations,
                        inner,
                    },
                    true,
                )
            }
            Origin::Scale { inner, factor } => {
                let inner = Box::new(inner.simplify().1);
                (false, Origin::Scale { inner, factor }, false)
            }
            Origin::Chain {
                before,
                mid_point,
                after,
            } => {
                let after = *after;
                let before = *before;
                let (trim_preds, before_origin, trim_after) = before.origin.simplify();
                let (trim_before, after_origin, trim_succs) = after.origin.simplify();
                let trim_preds = trim_preds || before.value == 0f64;
                let trim_succs = trim_succs || after.value == 0f64;
                if trim_before && before.value == 0f64 {
                    (true, after_origin, trim_succs)
                } else if trim_after && after.value == 0f64 {
                    (trim_preds, before_origin, true)
                } else {
                    let before = Bound {
                        value: before.value,
                        lol: before.lol,
                        origin: before_origin,
                        size: before.size,
                    };
                    let after = Bound {
                        value: after.value,
                        lol: after.lol,
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
            }
        }
    }
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Origin::Latency => write!(f, "a dependency"),
            Origin::Bottleneck(ref name, level) => {
                write!(f, "the pressure on {} at the {}", name, level)
            }
            Origin::HardwareEvaluation => write!(f, "the evaluation on the hardware"),
            Origin::Loop {
                ref dims,
                iterations,
                ref inner,
            } => write!(
                f,
                "{} iterations along dimensions [{}] of {{ {} }}",
                iterations,
                dims.iter().format(", "),
                inner
            ),
            Origin::Scale { ref inner, factor } => {
                write!(f, "scale by {:.2e}: {}", factor, inner)
            }
            Origin::Chain {
                ref before,
                ref mid_point,
                ref after,
            } => {
                display_inline_chain(before, f)?;
                write!(f, " to {}, then ", mid_point)?;
                display_inline_chain(after, f)?;
                Ok(())
            }
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
    latency: SymbolicFloat,
    bottlenecks: Vec<SymbolicFloat>,
}

impl HwPressure {
    /// Creates a new `Pressure`
    pub fn new<T, II>(latency: T, bottlenecks: II) -> Self
    where
        T: Into<SymbolicFloat>,
        II: IntoIterator<Item = T>,
    {
        HwPressure {
            latency: latency.into(),
            bottlenecks: bottlenecks.into_iter().map(T::into).collect(),
        }
    }

    /// Creates a null `Pressure` for the given device.
    pub fn zero(device: &dyn Device) -> Self {
        HwPressure::new(0f64, device.bottlenecks().iter().map(|_| 0f64))
    }

    /// Derive a bound on the execution time from the pressure on the hardware.
    pub fn bound(&self, level: BottleneckLevel, rates: &HwPressure) -> FastBound {
        let latency = FastBound::new(
            &self.latency / unwrap!(rates.latency.as_f64()),
            FastOrigin::Latency,
        );
        let bound = FastBound::max(
            self.bottlenecks
                .iter()
                .zip_eq(&rates.bottlenecks)
                .enumerate()
                .map(|(id, (value, rate))| {
                    FastBound::new(
                        value / unwrap!(rate.as_f64()),
                        FastOrigin::Bottleneck(id, level),
                    )
                })
                .chain(iter::once(latency)),
        );
        unwrap!(bound)
    }

    /// Adds the pressure of another computation, performed in parallel.
    pub fn add_parallel(&mut self, other: &HwPressure) {
        self.latency.max_assign(&other.latency);
        for (other, b) in other.bottlenecks.iter().zip_eq(&mut self.bottlenecks) {
            *b += other;
        }
    }

    /// Adds the pressure of another computation, performed sequentially.
    pub fn add_sequential(&mut self, other: &HwPressure) {
        self.latency += &other.latency;
        for (other, b) in other.bottlenecks.iter().zip_eq(&mut self.bottlenecks) {
            *b += other;
        }
    }

    /// Computes the pressure obtained by duplicating this one in parallel.
    pub fn repeat_parallel(&mut self, factor: size::SymbolicFloat) {
        for b in &mut self.bottlenecks {
            *b *= &factor;
        }
    }

    /// Adds the pressure of another computation, repeated in parallel. Ignores the latency.
    pub fn repeat_and_add_bottlenecks(
        &mut self,
        factor: &SymbolicFloat,
        other: &HwPressure,
    ) {
        for (other, b) in other.bottlenecks.iter().zip_eq(&mut self.bottlenecks) {
            *b += other * factor;
        }
    }

    /// Computes the pressure obtained by repeating this one sequentially.
    pub fn repeat_sequential(&mut self, factor: &size::SymbolicInt) {
        self.latency *= factor;
        for b in &mut self.bottlenecks {
            *b *= factor;
        }
    }

    /// Take the minimum of `self` and `other` for each bottleneck.
    pub fn minimize(&mut self, other: &HwPressure) {
        self.latency.min_assign(&other.latency);
        for (other, b) in other.bottlenecks.iter().zip_eq(&mut self.bottlenecks) {
            b.min_assign(other);
        }
    }

    /// Returns the pointwise minimum of a serie of `HwPressure`
    pub fn min<'a, IT>(mut it: IT) -> Option<Self>
    where
        IT: Iterator<Item = &'a Self>,
    {
        it.next().cloned().map(|mut pressure| {
            for other in it {
                pressure.minimize(other)
            }
            pressure
        })
    }

    /// Returns the pressure on a bottleneck.
    #[cfg(test)]
    pub fn get_bottleneck(&self, index: usize) -> &SymbolicFloat {
        &self.bottlenecks[index]
    }

    /// Pointwise multiplication of the pressure on each resource.
    pub fn multiply(&mut self, other: &HwPressure) {
        self.latency *= &other.latency;
        for (b, other) in self.bottlenecks.iter_mut().zip_eq(&other.bottlenecks) {
            *b *= other;
        }
    }

    /// Returns an object that implements [`Display`] for printing the hardware pressure in the
    /// corresponding device.
    ///
    /// [`Display`]: std::fmt::Display
    pub fn display<'a>(&'a self, device: &'a dyn Device) -> DisplayHwPressure<'a> {
        DisplayHwPressure {
            hw_pressure: self,
            device,
        }
    }
}

/// Helper struct for printing the hardware pressure with `format!` and `{}`.
///
/// The hardware pressure stores bottleneck information in a vector, but doesn't need to know about
/// the name of each bottleneck; that information is stored on the device.  This struct embeds a
/// reference to the [`Device`] to be able to print bottleneck names and rates.
///
/// [`Device`]: crate::device::Device
pub struct DisplayHwPressure<'a> {
    hw_pressure: &'a HwPressure,
    device: &'a dyn Device,
}

impl<'a> fmt::Debug for DisplayHwPressure<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.hw_pressure, fmt)
    }
}

impl<'a> fmt::Display for DisplayHwPressure<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let names = self.device.bottlenecks();
        let rates = self.device.total_rates();
        write!(
            fmt,
            "latency {} ({}ns)",
            self.hw_pressure.latency,
            &self.hw_pressure.latency / rates.latency.as_f64().unwrap()
        )?;

        let mut pairs = self
            .hw_pressure
            .bottlenecks
            .iter()
            .zip(&rates.bottlenecks)
            .enumerate()
            .map(|(id, (val, rate))| (names[id], val, val / rate.as_f64().unwrap()));
        if let Some((name, val, ratio)) = pairs.next() {
            write!(fmt, ", with usage {} on {} ({}ns)", val, name, ratio)?;
        }
        for (name, val, ratio) in pairs {
            write!(fmt, ", {} on {} ({}ns)", val, name, ratio)?;
        }
        Ok(())
    }
}
