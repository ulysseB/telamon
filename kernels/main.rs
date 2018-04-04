#![allow(dead_code)]
// TODO(cleanup): remove the cuda/non-cuda switch and put in in Cargo.toml instead once
// the required features will have landed.
extern crate env_logger;
extern crate telamon;
#[macro_use]
extern crate log;
extern crate itertools;
extern crate rayon;

mod common;

use common::*;
use rayon::prelude::*;
use telamon::helper;
use telamon::helper::tensor::{Tensor, VirtualTensor};
use telamon::device::cuda;
use telamon::device::Context;
use telamon::{ir, search_space};
use telamon::search_space::{Action, DimKind, Order, InstFlag};

use ir::DimMapScope::Global as GlobalScope;

// FIXME: generic doesn't works on mv

// Finds the best implementation of the saxpy function. saxpy computes y = a.x + y, where
// "a" is a scalar and "x" and "y" are vectors of size "n".
fn main() {
    // Enable logging.
    env_logger::init();
    let executor = cuda::Executor::init();
    // axpy(1 << 26, ir::Type::F(32), true, &executor);
     mv(1 << 18, 1 << 10, ir::Type::F(32), true, &executor); 
    //gesummv(1 << 18, 1 << 10, ir::Type::F(32), true, &executor);
    // mm(1024, 1024, 1024, ir::Type::F(32), true, &executor);
    // doitgen(256, 256, 256, ir::Type::F(32), true, &executor);
}

/*
fn tensor_reduction() { unimplemented!() } // FIXME

fn floyd_warshall(n: i32, data_type: ir::Type, generic: bool, executor: &cuda::Executor) {
    // TODO(search): express the outer loop.
    unimplemented!() // FIXME, for a fixed k
}

fn n_bodies() { unimplemented!() } // FIXME, in n dimensions, need sqrt

fn cell() {
    // FIXME: FC + relu
    unimplemented!()
}

fn pooling() { unimplemented!() } // FIXME: FC + relu + polling

fn lstm_cell() { unimplemented!() } // FIXME: too complex for now*/

// FIXME: Polybench
// backpropagation ?
// gemver > might want to load twice separately
// atax, CNN > need global broadcast
// dicgi, mvt, dot > need global reduction
// 2mm, two-level NN > need global bcast or global reduction
