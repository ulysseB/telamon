//! Defines the CPU target.
// TODO(mppa): improve the IR.
// - check privatisation between threads is correct
// - allow multiple threads groups in the same cluster, to overlap transfers
// - blocking async transfers
// - events + retiming
mod context;
mod mppa;
mod printer;
mod telajax;

pub use self::context::Context;
pub use self::mppa::Mppa;
pub use self::printer::MppaPrinter;

use codegen::{self, VarType};
use ir;
use num::bigint::BigInt;
use num::rational::Ratio;
use num::ToPrimitive;
use utils::*;
use itertools::Itertools;

#[derive(Default)]
pub struct Namer {
    num_var: HashMap<VarType, usize>,
    num_sizes: usize,
}

impl Namer {
    pub fn gen_prefix(t: VarType) -> &'static str {
        match t {
            VarType::I(1) => "p",
            VarType::I(8) => "c",
            VarType::I(16) => "s",
            VarType::I(32) => "r",
            VarType::I(64) => "rd",
            VarType::F(16) => "h",
            VarType::F(32) => "f",
            VarType::F(64) => "d",
            VarType::Ptr =>  "ptr",
            _ => panic!("invalid CPU type"),
        }
    }

    pub fn get_string(t: VarType) -> &'static str {
        match t {
            VarType::Ptr => "intptr_t",
            VarType::F(32) => "float",
            VarType::F(64) => "double",
            VarType::I(1) => "int8_t",
            VarType::I(8) => "int8_t",
            VarType::I(16) => "int16_t",
            VarType::I(32) => "int32_t",
            VarType::I(64) => "int64_t",
            _ => panic!("invalid CPU type"),
        }
    }
}

impl codegen::Namer for Namer {
    fn name(&mut self, t: ir::Type) -> String {
        let cpu_t = match t {
            ir::Type::Void => panic!("Should not get Void here"),
            ir::Type::I(size) => VarType::I(size),
            ir::Type::F(size) => VarType::F(size),
            ir::Type::PtrTo(..) => VarType::Ptr,
        };
        let prefix = Namer::gen_prefix(cpu_t);
        let entry = self.num_var.entry(cpu_t).or_insert(0);
        let name = format!("{}{}", prefix, *entry);
        *entry += 1;
        name
    }

    fn name_float(&self, val: &Ratio<BigInt>, len: u16) -> String {
        assert!(len <= 64);
        let f = unwrap!(val.numer().to_f64()) / unwrap!(val.denom().to_f64());
        format!("{:.5e}", f )
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
                format!("_size_{}", self.num_sizes-1)
            },
        }
    }
    fn get_declared_variables(&self) -> Vec<(VarType, usize)> {
        self.num_var.iter().map(|(&t, &n)| (t, n)).collect_vec()
    }
}
