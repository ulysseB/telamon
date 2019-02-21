//! Defines the CPU target.
// TODO(mppa): improve the IR.
// - check privatisation between threads is correct
// - allow multiple threads groups in the same cluster, to overlap transfers
// - blocking async transfers
// - events + retiming
mod context;
mod mppa;
mod printer;

pub use self::context::Context;
pub use self::mppa::Mppa;
pub use self::printer::MppaPrinter;

use crate::codegen;
use itertools::Itertools;
use num::bigint::BigInt;
use num::rational::Ratio;
use num::ToPrimitive;
use utils::*;

#[derive(Default)]
pub struct Namer {
    num_var: HashMap<codegen::DeclType, usize>,
    num_sizes: usize,
}

impl Namer {
    pub fn gen_prefix(t: codegen::DeclType) -> &'static str {
        match t {
            codegen::DeclType::I(1) => "p",
            codegen::DeclType::I(8) => "c",
            codegen::DeclType::I(16) => "s",
            codegen::DeclType::I(32) => "r",
            codegen::DeclType::I(64) => "rd",
            codegen::DeclType::F(16) => "h",
            codegen::DeclType::F(32) => "f",
            codegen::DeclType::F(64) => "d",
            codegen::DeclType::Ptr => "ptr",
            _ => panic!("invalid CPU type"),
        }
    }

    pub fn get_string(t: codegen::DeclType) -> &'static str {
        match t {
            codegen::DeclType::Ptr => "uintptr_t",
            codegen::DeclType::F(32) => "float",
            codegen::DeclType::F(64) => "double",
            codegen::DeclType::I(1) => "uint8_t",
            codegen::DeclType::I(8) => "uint8_t",
            codegen::DeclType::I(16) => "uint16_t",
            codegen::DeclType::I(32) => "uint32_t",
            codegen::DeclType::I(64) => "uint64_t",
            _ => panic!("invalid CPU type"),
        }
    }
}

impl codegen::Namer for Namer {
    fn name(&mut self, t: codegen::DeclType) -> String {
        let prefix = Namer::gen_prefix(t);
        let entry = self.num_var.entry(t).or_insert(0);
        let name = format!("{}{}", prefix, *entry);
        *entry += 1;
        name
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
