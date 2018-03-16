//! Describes the context for which a function must be optimized.
use device::{Argument, Device};
use codegen::Function;
use explorer::Candidate;
use ir;
use num;

use std::boxed::FnBox;

/// A callback that is called after evaluating a kernel.
pub type AsyncCallback<'a, 'b> = Box<FnBox(Candidate<'a>, f64, usize) + Send + 'b>;

/// Describes the context for which a function must be optimized.
// FIXME: use an associated type for arguments to enable variance
pub trait Context<'a>: Sync {
    /// Returns the description of the device the code runs on.
    fn device(&self) -> &Device;
    /// Binds a parameter to a given value.
    fn bind_param(&mut self, param: &ir::Parameter, value: Box<Argument + 'a>);
    /// Allocates an array of the given size in bytes.
    fn allocate_array(&mut self, id: ir::mem::Id, size: usize) -> Box<Argument + 'a>;
    /// Returns a parameter given its name.
    fn get_param<'b>(&'b self, &str) -> &'b Argument;
    /// Returns the execution time of a fully specified implementation in nanoseconds.
    fn evaluate(&self, space: &Function) -> Result<f64, ()>;
    /// Calls the `inner` closure in parallel, and gives it a pointer to an `AsyncEvaluator`
    /// to evaluate candidates in the context.
    fn async_eval<'b, 'c>(&self, num_workers: usize,
                          inner: &(Fn(&mut AsyncEvaluator<'b, 'c>) + Sync)) ;

    /// Evaluate a size.
    fn eval_size(&self, size: &ir::Size) -> u32 {
        let mut result: u32 = size.factor();
        for p in size.dividend() {
            result *= unwrap!(self.get_param(&p.name).as_size());
        };
        let (result, remider) = num::integer::div_rem(result, size.divisor());
        assert_eq!(remider, 0);
        result
    }
}

/// An evaluation context that runs kernels asynchronously on the target device.
pub trait AsyncEvaluator<'a, 'b> {
    /// Add a kernel to evaluate.
    fn add_kernel(&mut self, candidate: Candidate<'a>, callback: AsyncCallback<'a, 'b>);
}

