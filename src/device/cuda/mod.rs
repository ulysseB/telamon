//! Defines the CUDA target.
#[cfg(feature = "cuda")]
mod api;
#[cfg(not(feature = "cuda"))]
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

#[cfg(feature = "cuda")]
pub mod characterize;

// Constructs to retrieve information on the GPU, that are not needed for the regular
// operation of Telamon and thus only present if the cuda feature is.
pub use self::api::{Array, Executor, JITDaemon};
#[cfg(feature = "cuda")]
pub use self::api::{DeviceAttribute, PerfCounter, PerfCounterSet};
pub use self::context::Context;
pub use self::gpu::{Gpu, InstDesc};
pub use self::kernel::Kernel;
pub use self::printer::CudaPrinter;

use crate::codegen;
use crate::ir;
use num::bigint::BigInt;
use num::rational::Ratio;
use num::ToPrimitive;
use std;
use utils::*;

#[derive(Default)]
struct Namer {
    num_var: HashMap<ir::Type, usize>,
    num_sizes: usize,
}

impl Namer {
    /// Generate a variable name prefix from a type.
    fn gen_prefix(t: &codegen::DeclType) -> &'static str {
        match *t {
            codegen::DeclType::I(1) => "p",
            codegen::DeclType::I(8) => "c",
            codegen::DeclType::I(16) => "s",
            codegen::DeclType::I(32) => "r",
            codegen::DeclType::I(64) => "rd",
            codegen::DeclType::F(16) => "h",
            codegen::DeclType::F(32) => "f",
            codegen::DeclType::F(64) => "d",
            _ => panic!("invalid PTX type"),
        }
    }
}

impl codegen::Namer for Namer {
    fn name(&mut self, t: codegen::DeclType) -> String {
        let prefix = Namer::gen_prefix(&t);
        let entry = self.num_var.entry(t).or_insert(0);
        let name = format!("%{}{}", prefix, *entry);
        *entry += 1;
        name
    }

    fn name_float(&self, val: &Ratio<BigInt>, len: u16) -> String {
        assert!(len <= 64);
        let f = unwrap!(val.numer().to_f64()) / unwrap!(val.denom().to_f64());
        let binary = unsafe { std::mem::transmute::<f64, u64>(f) };
        format!("0D{:016X}", binary)
    }

    fn name_int(&self, val: &BigInt, len: u16) -> String {
        assert!(len <= 64);
        format!("{}", unwrap!(val.to_i64()))
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
    fn get_declared_variables(&self) -> Vec<(codegen::DeclType, usize)> {
        self.num_var.iter().map(|(&t, &n)| (t, n)).collect_vec()
    }
}
