//! Describes the context for which a function must be optimized.
use crate::codegen::{self, Function};
use crate::device::{ArrayArgument, Device, ScalarArgument};
use crate::explorer::Candidate;
use crate::ir;
use boxfnonce::SendBoxFnOnce;
use num;
use std::sync::Arc;
use utils::unwrap;

/// A callback that is called after evaluating a kernel.
pub type AsyncCallback<'a, 'b> = SendBoxFnOnce<'b, (Candidate<'a>, f64)>;

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
    fn add_kernel(&mut self, candidate: Candidate<'a>, callback: AsyncCallback<'a, 'b>);
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
