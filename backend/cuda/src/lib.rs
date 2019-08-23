//! Defines the CUDA target.
#![deny(bare_trait_objects)]
#[cfg(feature = "real_gpu")]
mod api;
#[cfg(not(feature = "real_gpu"))]
mod api {
    mod error;
    mod fake;
    pub use self::error::*;
    pub use self::fake::*;
}
mod context;
mod gpu;
mod kernel;
mod mem_model;
mod printer;

#[cfg(feature = "real_gpu")]
pub mod characterize;

// Constructs to retrieve information on the GPU, that are not needed for the regular
// operation of Telamon and thus only present if the cuda feature is.
pub use self::api::{Array, Executor, JITDaemon};
#[cfg(feature = "real_gpu")]
pub use self::api::{DeviceAttribute, PerfCounter, PerfCounterSet};
pub use self::context::Context;
pub use self::gpu::{Gpu, InstDesc};
pub use self::kernel::Kernel;

use fxhash::FxHashMap;
use telamon::codegen;
use telamon::ir;

#[derive(Default)]
pub(crate) struct NameGenerator {
    num_var: FxHashMap<ir::Type, usize>,
    num_sizes: usize,
}

impl NameGenerator {
    /// Generate a variable name prefix from a type.
    fn gen_prefix(t: ir::Type) -> &'static str {
        match t {
            ir::Type::I(1) => "p",
            ir::Type::I(8) => "c",
            ir::Type::I(16) => "s",
            ir::Type::I(32) => "r",
            ir::Type::I(64) => "rd",
            ir::Type::F(16) => "h",
            ir::Type::F(32) => "f",
            ir::Type::F(64) => "d",
            _ => panic!("invalid PTX type"),
        }
    }
}

impl codegen::NameGenerator for NameGenerator {
    fn name(&mut self, t: ir::Type) -> String {
        let prefix = NameGenerator::gen_prefix(t);
        let entry = self.num_var.entry(t).or_insert(0);
        let name = format!("%{}{}", prefix, *entry);
        *entry += 1;
        name
    }

    fn name_param(&mut self, p: codegen::ParamValKey) -> String {
        match p {
            codegen::ParamValKey::External(p) => p.name.clone(),
            codegen::ParamValKey::GlobalMem(mem) => format!("_gbl_mem_{}", mem.0),
            codegen::ParamValKey::Size(_) => {
                self.num_sizes += 1;
                format!("_size_{}", self.num_sizes - 1)
            }
        }
    }
}
