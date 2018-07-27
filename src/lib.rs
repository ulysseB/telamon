// Copyright 2018 Ulysse Beaugnon and Ecole Normale Superieure
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

extern crate binary_heap_plus;
extern crate boxfnonce;
extern crate config;
extern crate crossbeam;
#[cfg(test)]
extern crate env_logger;
extern crate getopts;
extern crate errno;
#[macro_use]
extern crate failure;
extern crate interval_heap;
extern crate itertools;
extern crate libloading;
extern crate tempfile;
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
extern crate prettytable;
#[cfg(feature="cuda")]
extern crate prctl;
extern crate rand;
extern crate rpds;
extern crate futures;
extern crate tokio_timer;
#[macro_use]
extern crate serde;
extern crate serde_json;
extern crate bincode;
#[macro_use]
extern crate telamon_utils as utils;
extern crate xdg;

pub mod codegen;
#[macro_use]
pub mod helper;
pub mod device;
pub mod explorer;
pub mod ir;
pub mod model;
pub mod search_space;
