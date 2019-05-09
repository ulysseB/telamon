//! Defines the CPU target.
#![deny(bare_trait_objects)]

mod compile;
mod context;
mod cpu;
mod cpu_argument;

pub use crate::context::Context;
pub use crate::cpu::Cpu;
