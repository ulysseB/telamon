#![feature(slice_patterns, conservative_impl_trait, box_syntax, box_patterns, type_ascription, fnbox)]

extern crate config;
extern crate crossbeam;
extern crate data_structure_traits;
#[cfg(test)]
extern crate env_logger;
extern crate getopts;
extern crate errno;
extern crate interval_heap;
extern crate immut_list;
extern crate itertools;
#[cfg(feature="cuda")]
extern crate ipc_channel;
#[macro_use]
extern crate lazy_static;
extern crate libc;
extern crate linked_list;
#[macro_use]
extern crate log;
extern crate num;
#[cfg(feature="mppa")]
extern crate parking_lot;
#[cfg(feature="cuda")]
extern crate prctl;
extern crate rand;
#[cfg(feature="cuda")]
extern crate rustc_serialize;
#[macro_use]
extern crate telamon_utils as utils;

pub mod codegen;
#[macro_use]
pub mod helper;
pub mod device;
// TODO(cleanup): remove warnings from the explorer
#[allow(dead_code,unused_variables,unused_imports,ignored_generic_bounds)]
#[cfg_attr(feature="cargo-clippy", allow(clippy))]
pub mod explorer;
pub mod ir;
pub mod model;
pub mod search_space;
