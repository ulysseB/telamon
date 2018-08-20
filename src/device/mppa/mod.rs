//! Backend for MPPA devices.
mod context;
mod device;
mod printer;
mod telajax;

pub use self::context::{Buffer, Context};
pub use self::device::MPPA;

// TODO(mppa): improve the IR.
// - check privatisation between threads is correct
// - allow multiple threads groups in the same cluster, to overlap transfers
// - blocking async transfers
// - events + retiming
