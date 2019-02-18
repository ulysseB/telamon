//! Defines the CPU target.
mod context;
mod cpu;
//mod mem_model;
mod compile;
mod cpu_argument;
mod printer;

pub use self::context::Context;
pub use self::cpu::Cpu;
pub use self::printer::X86printer;

use crate::codegen;
use crate::ir;
use num::bigint::BigInt;
use num::rational::Ratio;
use num::ToPrimitive;
use utils::*;

#[derive(Default)]
struct Namer {
    num_var: HashMap<codegen::DeclType, usize>,
    num_sizes: usize,
    num_glob_ptr: usize,
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
            codegen::DeclType::PtrTo(..) => "ptr",
            _ => panic!("invalid CPU type"),
        }
    }
}

impl codegen::Namer for Namer {
    fn name(&mut self, t: codegen::DeclType) -> String {
        let prefix = Namer::gen_prefix(&t);
        match t {
            codegen::DeclType::Ptr => {
                let name = format!("{}{}", prefix, self.num_glob_ptr);
                self.num_glob_ptr += 1;
                name
            }
            _ => {
                let entry = self.num_var.entry(t).or_insert(0);
                let name = format!("{}{}", prefix, *entry);
                *entry += 1;
                name
            }
        }
    }

    fn name_float(&self, val: &Ratio<BigInt>, len: u16) -> String {
        assert!(len <= 64);
        let f = unwrap!(val.numer().to_f64()) / unwrap!(val.denom().to_f64());
        format!("{:.5e}", f)
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
