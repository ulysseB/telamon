#![allow(dead_code)]
//! Provides a fake implementations of device traits for testing.
use std::io::Write;
use telamon::codegen;
use telamon::device::{self, fake::FakeContext};
use telamon::ir::{self, Operator};
use telamon::model::{self, HwPressure};
use telamon::search_space::*;
use utils::*;

pub type Context = FakeContext<Device>;

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

impl device::Device for Device {
    fn name(&self) -> &str {
        "fake_device"
    }

    fn print(&self, _: &codegen::Function, _: &mut Write) {}

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
            Operator::TmpLd(..)
            | Operator::TmpSt(..)
            | Operator::BinOp(ir::BinOp::Add, ..) => true,
            Operator::Ld(t, _, ref pattern) => pattern.is_consecutive(dim.id(), t),
            Operator::St(_, ref operand, _, ref pattern) => {
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
        _: &ir::Statement,
        _: &device::Context,
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
