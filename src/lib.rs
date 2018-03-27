extern crate boxfnonce;
extern crate config;
extern crate crossbeam;
#[cfg(test)]
extern crate env_logger;
extern crate getopts;
extern crate errno;
extern crate interval_heap;
extern crate itertools;
#[cfg(feature="cuda")]
extern crate ipc_channel;
#[cfg(feature="cuda")]
#[macro_use]
extern crate lazy_static;
extern crate libc;
extern crate linked_list;
#[macro_use]
extern crate log;
#[macro_use]
extern crate matches;
extern crate ndarray;
extern crate num;
extern crate num_cpus;
#[cfg(feature="mppa")]
extern crate parking_lot;
#[cfg(feature="cuda")]
extern crate prctl;
extern crate rand;
extern crate rpds;
extern crate futures;
extern crate tokio_timer;
#[cfg(feature="cuda")]
extern crate rustc_serialize;
#[macro_use]
extern crate telamon_utils as utils;

pub mod codegen;
#[macro_use]
pub mod helper;
pub mod device;
pub mod explorer;
pub mod ir;
pub mod model;
pub mod search_space;

// FIXME: thread-mapping
// > test the model is corrent
//   - test local info
//   - test sum_pressure
// > test the codegen is correct
//   - thread mapping
//   - predicated store
// > fix store
