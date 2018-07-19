#![allow(dead_code)]
//! Provides a fake implementations of device traits for testing.
use telamon::codegen;
use telamon::device::{self, ScalarArgument, ArrayArgument};
use telamon::ir::{self, Operator};
use telamon::explorer::Candidate;
use telamon::search_space::{SearchSpace, DimKind};
use telamon::model::{self, HwPressure};
use std::sync::Arc;
use std::f64;
use std::io::Write;
use utils::*;

use std::marker::PhantomData;
/// A fake device.
pub struct Device {
    pub shared_mem_size: u32,
}

impl Default for Device {
    fn default() -> Device { Device { shared_mem_size: 1 << 17 } }
}

impl device::Device for Device {
    fn name(&self) -> &str { "fake_device" }

    fn print(&self, _: &codegen::Function, _: &mut Write) { }

    fn check_type(&self, _: ir::Type) -> Result<(), ir::TypeError> { Ok(()) }

    fn max_unrolling(&self) -> u32 { 256 }

    fn vectorization_factors(&self, dim: &ir::Dimension, op: &ir::Operator) -> &[u32] {
        const LD_ST_FACTORS: [u32; 2] = [2, 4];
        const OTHER_FACTORS: [u32; 0] = [];
        match *op {
            Operator::TmpLd(..) | Operator::TmpSt(..) => &LD_ST_FACTORS,
            Operator::Ld(ref t, _, ref pattern) if pattern.is_consecutive(dim.id(), t) =>
                &LD_ST_FACTORS,
            Operator::St(_, ref operand, _, ref pattern)
                if pattern.is_consecutive(dim.id(), &operand.t()) => &LD_ST_FACTORS,
            _ => &OTHER_FACTORS,
        }
    }

    fn max_block_dims(&self) -> u32 { 3 }

    fn max_threads(&self) -> u32 { 1024 }

    fn shared_mem(&self) -> u32 { self.shared_mem_size }

    fn supports_nc_access(&self) -> bool { true }

    fn supports_l1_access(&self) -> bool { true }

    fn supports_l2_access(&self) -> bool { true }

    fn lower_type(&self, t: ir::Type, _: &SearchSpace) -> Option<ir::Type> { Some(t) }

    fn loop_iter_pressure(&self, _: DimKind) -> (HwPressure, HwPressure) {
        (HwPressure::zero(self), HwPressure::zero(self))
    }

    fn hw_pressure(&self, _: &SearchSpace,
                   _: &HashMap<ir::dim::Id, u32>,
                   _: &HashMap<ir::BBId, model::Nesting>,
                   _: &ir::BasicBlock,
                   _: &device::Context) -> HwPressure {
        HwPressure::zero(self)
    }

    fn bottlenecks(&self) -> &[&'static str] { &["issue", "alu", "mem"] }

    fn block_parallelism(&self, _: &SearchSpace) -> u32 { 16 }

    fn additive_indvar_pressure(&self, _: &ir::Type) -> HwPressure {
        HwPressure::zero(self)
    }

    fn multiplicative_indvar_pressure(&self, _: &ir::Type) -> HwPressure {
        HwPressure::zero(self)
    }

    fn thread_rates(&self) -> HwPressure { HwPressure::new(1.0, vec![1.0, 1.0, 1.0]) }

    fn block_rates(&self) -> HwPressure { HwPressure::new(1.0, vec![1.0, 1.0, 1.0]) }

    fn total_rates(&self) -> HwPressure { HwPressure::new(1.0, vec![1.0, 1.0, 1.0]) }

    fn add_block_overhead(&self, _: u64, _: u64, _: &mut HwPressure) { }
}

/// A fake context.
#[derive(Default)]
pub struct Context {
    pub device: Device,
}

impl device::Context for Context {
    fn device(&self) -> &device::Device { &self.device }

    fn evaluate(&self, _: &codegen::Function, _: device::EvalMode) -> Result<f64, ()> {
        Ok(1.0)
    }

    fn benchmark(&self, _: &codegen::Function, num_samples: usize) -> Vec<f64> {
        vec![1.0; num_samples]
    }

    fn param_as_size(&self, _: &str) -> Option<u32> { Some(1) }

    fn async_eval<'b, 'c>(&self, _: usize, _: device::EvalMode,
                          inner: &(Fn(&mut device::AsyncEvaluator<'b, 'c>) + Sync)) {
        inner(&mut Evaluator { phantom: PhantomData });
    }
}

impl device::ArgMap for Context {
    type Array = Array;

    fn bind_scalar<S: ScalarArgument>(&mut self, param: &ir::Parameter, _: S) {
        assert_eq!(param.t, S::t());
    }

    fn bind_array<S: ScalarArgument>(&mut self, _: &ir::Parameter, _: usize)
        -> Arc<Self::Array>
    {
        Arc::new(Array)
    }
}

pub struct Array;

impl ArrayArgument for Array {
    fn read_i8(&self) -> Vec<i8> { vec![] }

    fn write_i8(&self, _: &[i8]) { }
}

/// A fake asynchronous evaluator.
struct Evaluator<'a, 'b> {
    phantom: PhantomData<(&'a (), &'b ())>,
}

impl<'a, 'b, 'c > device::AsyncEvaluator<'a, 'c> for Evaluator<'a, 'b>
where 'a: 'b, 'c: 'b {
    fn add_kernel(&mut self, candidate: Candidate<'a>,
                  callback: device::AsyncCallback<'a, 'c>) {
        // Try to compile the function to check it works.
        codegen::Function::build(&candidate.space);
        callback.call(candidate, 1.0);
    }
}
