use device;
use ir;
use search_space::SearchSpace;
use std;

/// Describes a MPPA chip.
#[derive(Default)]
pub struct MPPA;

impl device::Device for MPPA {
    fn print(&self, fun: &device::Function, out: &mut std::io::Write) {
        device::mppa::printer::print(fun, false, out).unwrap();
    }

    fn lower_type(&self, t: ir::Type, _: &SearchSpace) -> Option<ir::Type> {
        Some(t)
    }

    // TODO(mppa): allow cluster parallelism
    fn max_block_dims(&self) -> u32 {
        0
    }

    // TODO(mppa): allow auto-threading
    fn max_threads(&self) -> u32 {
        16
    }

    // TODO(mppa): tune the max unrolling factor
    fn max_unrolling(&self) -> u32 {
        256
    }

    // TODO(mppa): allow SMEM use and allow L1 as scratchpad
    fn shared_mem(&self) -> u32 {
        0
    }

    fn name(&self) -> &str {
        "mppa"
    }
}
