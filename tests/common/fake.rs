#![allow(dead_code)]
//! Provides a fake implementations of device traits for testing.
use libc;
use telamon::codegen;
use telamon::device;
use telamon::ir;
use telamon::explorer::Candidate;
use telamon::search_space::{SearchSpace, DimKind};
use telamon::model::{self, HwPressure};
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

    fn is_valid_type(&self, _: &ir::Type) -> bool { true }

    fn max_unrolling(&self) -> u32 { 256 }

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
                   _: &ir::BasicBlock) -> HwPressure {
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

    fn block_rates(&self, _: u64) -> HwPressure { HwPressure::new(1.0, vec![1.0, 1.0, 1.0]) }

    fn total_rates(&self, _: u64) -> HwPressure { HwPressure::new(1.0, vec![1.0, 1.0, 1.0]) }
}

/// A fake context.
#[derive(Default)]
pub struct Context {
    pub device: Device,
}

static ONE: i32 = 1;

impl<'a> device::Context<'a> for Context {
    fn device(&self) -> &device::Device { &self.device }

    fn evaluate(&self, _: &codegen::Function) -> Result<f64, ()> { Ok(1.0) }

    fn get_param(&self, _: &str) -> &device::Argument { &ONE }

    fn async_eval<'b, 'c>(&self, _: usize,
                          inner: &(Fn(&mut device::AsyncEvaluator<'b, 'c>) + Sync)) {
        inner(&mut Evaluator { phantom: PhantomData });
    }

    fn bind_param(&mut self, param: &ir::Parameter, value: Box<device::Argument + 'a>) {
        assert_eq!(param.t, value.t());
    }

    fn allocate_array(&mut self, id: ir::mem::Id, _: usize) -> Box<device::Argument> {
        Box::new(FakeArray { id: id })
    }
}

/// A fake array.
struct FakeArray { id: ir::mem::Id }

impl device::Argument for FakeArray {
    fn t(&self) -> ir::Type { ir::Type::PtrTo(self.id) }

    fn raw_ptr(&self) -> *const libc::c_void { panic!() }

    fn size_of(&self) -> usize { 4 }
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
        (callback)(candidate, 1.0, 1);
    }
}
