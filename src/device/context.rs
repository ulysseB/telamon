//! Describes the context for which a function must be optimized.
use crate::codegen::{self, Function};
use crate::device::{ArrayArgument, Device, ScalarArgument};
use crate::explorer::Candidate;
use crate::ir;
use itertools::{process_results, Itertools};
use log::info;
use num;
use std::sync::Arc;
use std::{cmp, fmt};
use utils::{cmp_f64, unwrap};

/// A trait representing a kernel evaluator, i.e. an object which can run the kernel and return an
/// evaluated execution time.
///
/// Evaluators can be stabilized (see`Stabilizer`) by computing an average over several
/// evaluations; hence, implementers of this trait should make sure to have low overhead in the
/// `evaluate` function over the actual evaluation time.
pub trait KernelEvaluator: std::fmt::Display {
    /// Evaluate the kernel runtime, in nanoseconds.
    ///
    /// Repeated runs should return an identical value and hence calls to `evaluate` should not
    /// have side-effects visible from the kernel.
    fn evaluate(&mut self) -> Option<f64>;
}

pub trait AsyncCallbackFn<'a> {
    fn call(self: Box<Self>, candidate: Candidate<'a>, kernel: &mut dyn KernelEvaluator);
}

impl<'a, F> AsyncCallbackFn<'a> for F
where
    F: FnOnce(Candidate<'a>, &mut dyn KernelEvaluator),
{
    fn call(self: Box<Self>, candidate: Candidate<'a>, kernel: &mut dyn KernelEvaluator) {
        (*self)(candidate, kernel)
    }
}

pub type AsyncCallback<'a, 'b> = Box<dyn AsyncCallbackFn<'a> + Send + 'b>;

/// Describes the context for which a function must be optimized.
pub trait Context: Sync {
    /// Returns the description of the device the code runs on.
    fn device(&self) -> &dyn Device;
    /// Returns the execution time of a fully specified implementation in nanoseconds.
    ///
    /// This function should be called multiple times to obtain accurate execution time.
    /// Indeed, it only executes the code once, without warming the GPU first.
    fn evaluate(&self, space: &Function, mode: EvalMode) -> Result<f64, ()>;
    /// Compiles and benchmarks a functions. As opposed to `Self::evaluate`, the measured
    /// time contains potential startup times.
    fn benchmark(&self, space: &Function, num_samples: usize) -> Vec<f64>;
    /// Calls the `inner` closure in parallel, and gives it a pointer to an `AsyncEvaluator`
    /// to evaluate candidates in the context. `skip_bad_bounds` indicates than candidates
    /// whose bound is aboive the best candidate should be skiped.
    fn async_eval<'a, 'b>(
        &self,
        num_workers: usize,
        mode: EvalMode,
        inner: &(dyn Fn(&mut dyn AsyncEvaluator<'a, 'b>) + Sync),
    );

    /// Returns a parameter interpreted as a size, if possible.
    fn param_as_size(&self, name: &str) -> Option<u32>;

    /// Evaluate a size.
    fn eval_size(&self, size: &codegen::Size) -> u32 {
        let mut dividend: u32 = size.factor();
        for p in size.dividend() {
            dividend *= unwrap!(self.param_as_size(&p.name));
        }
        let (result, remider) = num::integer::div_rem(dividend, size.divisor());
        assert_eq!(
            remider, 0,
            "invalid size: {:?} (dividend = {})",
            size, dividend
        );
        result
    }

    /// Returns a default stabilizer configuration for use with this context.  By default, no
    /// stabilization is performed, so that stable devices don't need to override this.
    fn stabilizer(&self) -> Stabilizer {
        Stabilizer::default()
    }
}

/// Binds the argument names to their values.
pub trait ArgMap<'a>: Context + 'a {
    fn bind_erased_scalar(
        &mut self,
        param: &ir::Parameter,
        value: Box<dyn ScalarArgument>,
    );

    fn bind_erased_array(
        &mut self,
        param: &ir::Parameter,
        t: ir::Type,
        len: usize,
    ) -> Arc<dyn ArrayArgument + 'a>;
}

pub trait ArgMapExt<'a>: ArgMap<'a> {
    /// Binds a parameter to a given value.
    fn bind_scalar<S: ScalarArgument>(&mut self, param: &ir::Parameter, value: S) {
        self.bind_erased_scalar(param, Box::new(value))
    }

    /// Allocates an array of the given size in bytes.
    fn bind_array<S: ScalarArgument>(
        &mut self,
        param: &ir::Parameter,
        len: usize,
    ) -> Arc<dyn ArrayArgument + 'a> {
        self.bind_erased_array(param, S::t(), len)
    }
}

impl<'a, T: ?Sized> ArgMapExt<'a> for T where T: ArgMap<'a> {}

/// An evaluation context that runs kernels asynchronously on the target device.
pub trait AsyncEvaluator<'a, 'b> {
    /// Add a kernel to evaluate.
    fn add_any_kernel(
        &mut self,
        candidate: Candidate<'a>,
        callback: AsyncCallback<'a, 'b>,
    );
}

impl<'a, 'b, 'c> dyn AsyncEvaluator<'a, 'b> + 'c {
    /// Helper to add a kernel to an evaluator trait object.  Using `add_any_kernel` directly trips
    /// up type inference and requires explicitely typing closure arguments; using `add_kernel`
    /// instead allows a nicer syntax with closures.
    pub fn add_kernel<F>(&mut self, candidate: Candidate<'a>, callback: F)
    where
        F: FnOnce(Candidate<'a>, &mut dyn KernelEvaluator) + Send + 'b,
    {
        self.add_any_kernel(candidate, Box::new(callback))
    }
}

/// Indicates how evaluation should be performed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvalMode {
    /// Find the best candidate, skip bad candidates and allow optimizations.
    FindBest,
    /// Test the evaluation function, same as `FindBest` but do not skip candidates.
    TestEval,
    /// Test the performance model, do not skip candidates and do not optimize.
    TestBound,
}

impl EvalMode {
    /// Indicates if candidates with a bound above the cut can be skipped.
    pub fn skip_bad_candidates(self) -> bool {
        match self {
            EvalMode::FindBest => true,
            EvalMode::TestBound | EvalMode::TestEval => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Stabilizer {
    /// If true, the bad candidates will be skipped.  Bad candidates correspond to candidates whose
    /// bound is higher than the best provided.
    skip_bad_candidates: bool,
    /// Candidates with a runtime over `SKIP_THRESHOLD * best` are skipped after the first
    /// evaluation.   Ignored if `skip_bad_candidates` is `false`.
    skip_threshold: f64,
    /// Number of evaluation to perform on each candidate.
    num_evals: usize,
    /// Number of outlier evaluations to discard
    num_outliers: usize,
}

impl Default for Stabilizer {
    fn default() -> Self {
        Stabilizer {
            skip_bad_candidates: false,

            // FIXME: Tune values and add a second threshold after a few iterations
            skip_threshold: 3.,
            num_evals: 1,
            num_outliers: 0,
        }
    }
}

impl Stabilizer {
    pub fn skip_bad_candidates(mut self, skip_bad_candidates: bool) -> Self {
        self.skip_bad_candidates = skip_bad_candidates;
        self
    }

    pub fn skip_threshold(mut self, skip_threshold: f64) -> Self {
        self.skip_threshold = skip_threshold;
        self
    }

    pub fn num_evals(mut self, num_evals: usize) -> Self {
        self.num_evals = num_evals;
        self
    }

    pub fn num_outliers(mut self, num_outliers: usize) -> Self {
        self.num_outliers = num_outliers;
        self
    }
}

impl Stabilizer {
    pub fn wrap<'a>(
        &'a self,
        kernel: &'a mut dyn KernelEvaluator,
    ) -> StableEvaluator<'a> {
        StableEvaluator::new(self, kernel)
    }

    pub fn evaluate(&self, kernel: &mut dyn KernelEvaluator) -> Option<f64> {
        self.wrap(kernel).evaluate()
    }
}

pub struct StableEvaluator<'a> {
    stabilizer: &'a Stabilizer,
    kernel: &'a mut dyn KernelEvaluator,
    bound: Option<f64>,
    best: Option<f64>,
}

impl<'a> StableEvaluator<'a> {
    fn new(stabilizer: &'a Stabilizer, kernel: &'a mut dyn KernelEvaluator) -> Self {
        StableEvaluator {
            stabilizer,
            kernel,
            bound: None,
            best: None,
        }
    }

    pub fn bound(mut self, bound: Option<f64>) -> Self {
        self.bound = bound;
        self
    }

    pub fn best(mut self, best: Option<f64>) -> Self {
        self.best = best;
        self
    }

    pub fn evaluate(&mut self) -> Option<f64> {
        let mut num_evals = self.stabilizer.num_evals;
        if self.stabilizer.skip_bad_candidates {
            if let Some(bound) = self.bound {
                if let Some(best) = self.best {
                    if bound >= best {
                        info!("candidate skipped because of its bound");
                        return Some(std::f64::INFINITY);
                    }
                }

                let t0 = self.kernel.evaluate()?;

                if t0 * self.stabilizer.skip_threshold >= bound {
                    info!("candidate skipped after its first evaluation");
                    return Some(t0);
                }

                // Avoid spending too much time on very slow candidates.
                num_evals = cmp::max(
                    1,
                    cmp::min(self.stabilizer.num_evals, (1.0e9 / t0) as usize),
                );
            }
        }

        let num_samples =
            cmp::max(1, num_evals.saturating_sub(self.stabilizer.num_outliers));

        // TODO(cc_perf): becomes the limiting factor after a few hours. We should stop
        // earlier and make tests to know when (for example, measure the MAX delta between
        // min and median with N outliers).
        let runtimes = (0..num_evals).map(|_| self.kernel.evaluate().ok_or(()));
        let runtimes_by_value = process_results(runtimes, |iter| {
            iter.sorted_by(|lhs, rhs| cmp_f64(*lhs, *rhs))
        })
        .ok()?
        .collect::<Vec<_>>();
        let median = runtimes_by_value[num_evals / 2];
        let runtimes_by_delta = runtimes_by_value
            .into_iter()
            .sorted_by(|lhs, rhs| cmp_f64((lhs - median).abs(), (rhs - median).abs()))
            .collect::<Vec<_>>();
        let average = runtimes_by_delta[..num_samples]
            .iter()
            .cloned()
            .sum::<f64>()
            / num_samples as f64;

        Some(average)
    }
}

impl<'a> fmt::Display for StableEvaluator<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "stabilized evaluator for {}", self.kernel)
    }
}
