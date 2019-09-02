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

use fxhash::FxHashMap;
use telamon::{codegen, ir};

#[derive(Default)]
pub struct NameGenerator {
    num_var: FxHashMap<ir::Type, usize>,
    num_glob_ptr: usize,
}

impl NameGenerator {
    /// Generate a variable name prefix from a type.
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
            _ => panic!("invalid CPU type"),
        }
    }
}

impl codegen::NameGenerator for NameGenerator {
    fn name(&mut self, t: ir::Type) -> String {
        let prefix = NameGenerator::gen_prefix(t);
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
}
