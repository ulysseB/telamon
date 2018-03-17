//! Defines the CUDA target.
mod api;
mod context;
mod kernel;
mod gpu;
mod mem_model;
mod printer;

pub use self::api::{Array, ArrayArg, Executor, PerfCounter, PerfCounterSet, JITDaemon};
pub use self::api::DeviceAttribute;
pub use self::context::Context;
pub use self::gpu::{Gpu, InstDesc};
pub use self::kernel::Kernel;

use codegen;
use device::Device;
use ir;
use num::bigint::BigInt;
use num::rational::Ratio;
use num::ToPrimitive;
use std;
use utils::*;

struct Namer<'a> {
    gpu: &'a Gpu,
    function: &'a codegen::Function<'a>,
    num_var: HashMap<ir::Type, usize>,
    num_sizes: usize,
}

impl<'a> Namer<'a> {
    /// Creates a new `Namer`
    fn new(fun: &'a codegen::Function<'a>, gpu: &'a Gpu) -> Self {
        Namer { gpu, function: fun, num_var: HashMap::default(), num_sizes: 0 }
    }

    /// Generate a variable name prefix from a type.
    fn gen_prefix(t: &ir::Type) -> &'static str {
        match *t {
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

impl<'a> codegen::Namer for Namer<'a> {
    fn name(&mut self, t: ir::Type) -> String {
        let t = unwrap!(self.gpu.lower_type(t, self.function.space()));
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

    fn name_param(&mut self, p: &codegen::ParamVal) -> String {
        match *p {
            codegen::ParamVal::External(p) => p.name.clone(),
            codegen::ParamVal::GlobalMem(mem, _) => format!("_gbl_mem_{}", mem.0),
            codegen::ParamVal::Size(_) => {
                self.num_sizes += 1;
                format!("_size_{}", self.num_sizes-1)
            },
        }
    }
}
