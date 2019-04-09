use std::collections::HashMap;
use std::io::Write;
use std::marker::PhantomData;
use std::sync::Arc;

use fnv::FnvHashMap;

use crate::codegen;
use crate::explorer::Candidate;
use crate::ir;
use crate::model::{self, HwPressure};
use crate::search_space::{DimKind, InstFlag, MemSpace, SearchSpace};

use super::{
    ArgMap, ArrayArgument, AsyncCallback, AsyncEvaluator, EvalMode, KernelEvaluator,
    ScalarArgument,
};

/// A fake device.
pub struct Device {
    pub shared_mem_size: u32,
}

impl Default for Device {
    fn default() -> Device {
        Device {
            shared_mem_size: 1 << 17,
        }
    }
}

impl super::Device for Device {
    fn name(&self) -> &str {
        "fake_device"
    }

    fn print(&self, _: &codegen::Function, _: &mut dyn Write) {}

    fn check_type(&self, _: ir::Type) -> Result<(), ir::TypeError> {
        Ok(())
    }

    fn max_unrolling(&self) -> u32 {
        256
    }

    /// Indicates which operators can be vectorized on a dimension. We only allow memory
    /// operators and `Add` to be vectorized (to be able to test both vectorizable and
    /// non-vectorizable operations).
    fn can_vectorize(&self, dim: &ir::Dimension, op: &ir::Operator) -> bool {
        match *op {
            ir::Operator::TmpLd(..)
            | ir::Operator::TmpSt(..)
            | ir::Operator::BinOp(ir::BinOp::Add, ..) => true,
            ir::Operator::Ld(t, _, ref pattern) => pattern.is_consecutive(dim.id(), t),
            ir::Operator::St(_, ref operand, _, ref pattern) => {
                pattern.is_consecutive(dim.id(), operand.t())
            }
            _ => false,
        }
    }

    fn max_vectorization(&self, _: &ir::Operator) -> [u32; 2] {
        // No need to discriminate on the operator since this is already handled by
        // `can_vectorize`.
        [4, 8]
    }

    fn has_vector_registers(&self) -> bool {
        true
    }

    fn max_block_dims(&self) -> u32 {
        3
    }

    fn max_threads(&self) -> u32 {
        1024
    }

    fn max_inner_block_size(&self) -> u32 {
        65535
    }

    fn shared_mem(&self) -> u32 {
        self.shared_mem_size
    }

    fn pointer_type(&self, _: MemSpace) -> ir::Type {
        ir::Type::I(32)
    }

    // Warning: this assumes only global memory accesses can use caches.
    fn supported_mem_flags(&self, op: &ir::Operator) -> InstFlag {
        match op {
            // Only accesses to external memory blocks can be non-coherent.
            ir::Operator::Ld(.., pat) if pat.mem_block().is_none() => InstFlag::ALL,
            ir::Operator::Ld(..)
            | ir::Operator::St(..)
            | ir::Operator::TmpLd(..)
            | ir::Operator::TmpSt(..) => InstFlag::COHERENT,
            _ => panic!("invalid memory access operator"),
        }
    }

    fn lower_type(&self, t: ir::Type, _: &SearchSpace) -> Option<ir::Type> {
        Some(t)
    }

    fn loop_iter_pressure(&self, _: DimKind) -> (HwPressure, HwPressure) {
        (HwPressure::zero(self), HwPressure::zero(self))
    }

    fn hw_pressure(
        &self,
        _: &SearchSpace,
        _: &FnvHashMap<ir::DimId, model::size::Range>,
        _: &FnvHashMap<ir::StmtId, model::Nesting>,
        _: &dyn ir::Statement,
        _: &dyn super::Context,
    ) -> HwPressure {
        HwPressure::zero(self)
    }

    fn bottlenecks(&self) -> &[&'static str] {
        &["issue", "alu", "mem"]
    }

    fn block_parallelism(&self, _: &SearchSpace) -> u32 {
        16
    }

    fn additive_indvar_pressure(&self, _: &ir::Type) -> HwPressure {
        HwPressure::zero(self)
    }

    fn multiplicative_indvar_pressure(&self, _: &ir::Type) -> HwPressure {
        HwPressure::zero(self)
    }

    fn thread_rates(&self) -> HwPressure {
        HwPressure::new(1.0, vec![1.0, 1.0, 1.0])
    }

    fn block_rates(&self) -> HwPressure {
        HwPressure::new(1.0, vec![1.0, 1.0, 1.0])
    }

    fn total_rates(&self) -> HwPressure {
        HwPressure::new(1.0, vec![1.0, 1.0, 1.0])
    }

    fn add_block_overhead(
        &self,
        _: model::size::FactorRange,
        _: model::size::FactorRange,
        _: model::size::Range,
        _: &mut HwPressure,
    ) {
    }
}

/// A fake context to use when we don't actually care about the
/// evaluation results.  This wraps any device for the performance
/// model, but always return 1 for all evaluation results.
#[derive(Debug, Default)]
pub struct Context<D = Device> {
    device: D,
    parameters: HashMap<String, Option<u32>>,
}

impl<D: super::Device> Context<D> {
    pub fn new(device: D) -> Self {
        Context {
            device,
            parameters: HashMap::new(),
        }
    }
}

impl<D: super::Device> super::Context for Context<D> {
    fn device(&self) -> &dyn super::Device {
        &self.device
    }

    fn evaluate(&self, _: &codegen::Function, _: EvalMode) -> Result<f64, ()> {
        Ok(1.0)
    }

    fn benchmark(&self, _: &codegen::Function, num_samples: usize) -> Vec<f64> {
        vec![1.0; num_samples]
    }

    fn param_as_size(&self, name: &str) -> Option<u32> {
        self.parameters[name]
    }

    fn async_eval<'b, 'c>(
        &self,
        _: usize,
        _: EvalMode,
        inner: &(dyn Fn(&mut dyn AsyncEvaluator<'b, 'c>) + Sync),
    ) {
        struct FakeEvaluator<'a, 'b> {
            phantom: PhantomData<(&'a (), &'b ())>,
        }

        impl<'a, 'b, 'c> AsyncEvaluator<'a, 'c> for FakeEvaluator<'a, 'b>
        where
            'a: 'b,
            'c: 'b,
        {
            fn add_dyn_kernel(
                &mut self,
                candidate: Candidate<'a>,
                callback: AsyncCallback<'a, 'c>,
            ) {
                use std::fmt;

                struct FakeCode;

                impl fmt::Display for FakeCode {
                    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                        write!(fmt, "<...>")
                    }
                }

                impl KernelEvaluator for FakeCode {
                    fn evaluate(&mut self) -> Option<f64> {
                        Some(1.)
                    }
                }

                codegen::Function::build(&candidate.space);
                callback.call(candidate, &mut FakeCode);
            }
        }

        inner(&mut FakeEvaluator {
            phantom: PhantomData,
        });
    }
}

impl<'a, D: super::Device + 'a> ArgMap<'a> for Context<D> {
    fn bind_erased_scalar(
        &mut self,
        param: &ir::Parameter,
        value: Box<dyn ScalarArgument>,
    ) {
        assert_eq!(param.t, value.get_type());

        self.parameters.insert(param.name.clone(), value.as_size());
    }

    fn bind_erased_array(
        &mut self,
        _: &ir::Parameter,
        _: ir::Type,
        _: usize,
    ) -> Arc<dyn ArrayArgument + 'a> {
        Arc::new(FakeArray)
    }
}

/// A fake array implementation which doesn't read or write anything.
pub struct FakeArray;

impl ArrayArgument for FakeArray {
    fn read_i8(&self) -> Vec<i8> {
        Vec::new()
    }

    fn write_i8(&self, _: &[i8]) {}
}
