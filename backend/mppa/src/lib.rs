//! Defines the CPU target.
// TODO(mppa): improve the IR.
// - check privatisation between threads is correct
// - allow multiple threads groups in the same cluster, to overlap transfers
// - blocking async transfers
// - events + retiming
mod context;
#[cfg(not(feature = "real_mppa"))]
mod fake_telajax;
mod mppa;
pub mod printer;

pub use crate::context::Context;
pub use crate::mppa::Mppa;
pub use telamon_c::ValuePrinter;

