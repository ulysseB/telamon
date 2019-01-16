//! Measures the memory used by an IR instance.
#[macro_use]
extern crate lazy_static;

use utils::unwrap;

mod common;

/// Reads the amount of resident memory.
fn resident_memory() -> usize {
    // many statistics are cached and only updated when the epoch is advanced.
    unwrap!(jemalloc_ctl::epoch());
    unwrap!(jemalloc_ctl::stats::resident())
}

fn main() {
    let mem_beg = resident_memory();
    let _space = common::MM.clone();
    let mem_one = resident_memory();
    println!("candidate size: {} bytes", mem_one - mem_beg);
}
