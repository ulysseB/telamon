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

    fn is_valid_type(&self, t: &ir::Type) -> bool {
        match *t {
            ir::Type::Void | ir::Type::PtrTo(..) => true,
            ir::Type::F(x) => x == 16 || x == 32 || x == 64,
            ir::Type::I(x) => x == 8 || x == 16 || x == 32 || x == 64,
        }
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

    fn supports_nc_access(&self) -> bool {
        false
    }

    fn supports_l1_access(&self) -> bool {
        false
    }

    fn supports_l2_access(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "mppa"
    }
}
