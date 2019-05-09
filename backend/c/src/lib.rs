//! Defines the C target.
pub mod printer;

use fxhash::FxHashMap;
use num::bigint::BigInt;
use num::rational::Ratio;
use num::traits::Float;
use num::ToPrimitive;
use telamon::{codegen, ir};
use utils::unwrap;

#[derive(Default)]
pub struct ValuePrinter {
    pub num_var: FxHashMap<ir::Type, usize>,
    pub num_sizes: usize,
    pub num_glob_ptr: usize,
}

impl ValuePrinter {
    pub fn gen_prefix(t: ir::Type) -> &'static str {
        match t {
            ir::Type::I(1) => "p",
            ir::Type::I(8) => "c",
            ir::Type::I(16) => "s",
            ir::Type::I(32) => "r",
            ir::Type::I(64) => "rd",
            ir::Type::F(16) => "h",
            ir::Type::F(32) => "f",
            ir::Type::F(64) => "d",
            ir::Type::PtrTo(..) => "ptr",
            _ => panic!("Invalid C data type"),
        }
    }

    pub fn get_string(t: ir::Type) -> &'static str {
        match t {
            ir::Type::PtrTo(_) => "uintptr_t",
            ir::Type::F(32) => "float",
            ir::Type::F(64) => "double",
            ir::Type::I(1) => "uint8_t",
            ir::Type::I(8) => "uint8_t",
            ir::Type::I(16) => "uint16_t",
            ir::Type::I(32) => "uint32_t",
            ir::Type::I(64) => "uint64_t",
            _ => panic!("Invalid C data type"),
        }
    }
}

impl codegen::ValuePrinter for ValuePrinter {
    fn get_const_float(&self, val: &Ratio<BigInt>, len: u16) -> String {
        assert!(len <= 64);
        let f = unwrap!(val.numer().to_f64()) / unwrap!(val.denom().to_f64());

        // Print in C99 hexadecimal floating point representation
        let (mantissa, exponent, sign) = f.integer_decode();
        let signchar = if sign < 0 { "-" } else { "" };

        // Assume that floats and doubles in the C implementation have
        // 32 and 64 bits, respectively
        let floating_suffix = match len {
            32 => "f",
            64 => "",
            _ => panic!("Cannot print floating point value with {} bits", len),
        };

        format!(
            "{}0x{:x}p{}{}",
            signchar, mantissa, exponent, floating_suffix
        )
    }

    fn get_const_int(&self, val: &BigInt, len: u16) -> String {
        assert!(len <= 64);
        format!("{}", unwrap!(val.to_i64()))
    }

    fn name(&mut self, t: ir::Type) -> String {
        let prefix = ValuePrinter::gen_prefix(t);
        match t {
            ir::Type::PtrTo(..) => {
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
