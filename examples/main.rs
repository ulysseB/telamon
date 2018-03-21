#![feature(conservative_impl_trait)]
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

use ir::DimMapScope::Global as GlobalScope;

// Finds the best implementation of the saxpy function. saxpy computes y = a.x + y, where
// "a" is a scalar and "x" and "y" are vectors of size "n".
fn main() {
    // Enable logging.
    env_logger::init();
    let executor = cuda::Executor::init();
    //axpy(1 << 26, ir::Type::F(32), false, &executor);
    mv(1 << 18, 1 << 10, ir::Type::F(32), false, &executor); 
}

/// Generates code for `y = _a*x+y`.
fn axpy(n: i32, data_type: ir::Type,  generic: bool, executor: &cuda::Executor) {
    // Initializes the evaluation context.
    let mut context = cuda::Context::new(&executor);
    // Declares the function signature and the arguments to use for the evaluation.
    let (x, y);
    let signature = {
        let mut builder = helper::SignatureBuilder::new("axpy", &mut context);
        // Create two scalar parameters, with the values N and A used for the evaluation.
        let n = create_size(n, "n", generic, &mut builder);
        builder.param("a", 0.0);
        // Allocates two arrays of size N.
        x = Tensor::new("x", vec![n], data_type, true, &mut builder);
        y = Tensor::new("y", vec![n], data_type, false, &mut builder);
        builder.get()
    };
    // Declares the bofy of the function.
    let function = {
        let mut builder = helper::Builder::new(&signature, context.device());
        assert!(n >= 1024*4);
        let tiling = &[1024, 4]; // FIXME: try more tile sizes

        let ld_x = x.load(&[tiling], &mut builder);
        let ld_y = y.load(&[tiling], &mut builder);
        let mad_dim = builder.open_mapped_dim(&ld_x[0]);
        let x_op = ld_x.dim_map(&[&mad_dim], GlobalScope, &mut builder);
        let y_op = ld_y.dim_map(&[&mad_dim], GlobalScope, &mut builder);
        let mad = VirtualTensor::new(builder.mad(&x_op, &"a", &y_op), vec![mad_dim]);
        mad.store(&y, &mut builder);
        builder.get()
    };
    // Explore the search space.
    gen_best(vec![function], &context, &file_name("axpy", data_type, &[n], generic));
}

/// Generates code for `y = A.x`.
fn mv(m: i32, n: i32, data_type: ir::Type, generic: bool, executor: &cuda::Executor) {
    let (a, x, y);
    let mut context = cuda::Context::new(&executor);
    let signature = {
        let mut builder = helper::SignatureBuilder::new("mv", &mut context);
        let m = create_size(m, "m", generic, &mut builder);
        let n = create_size(n, "n", generic, &mut builder);
        a = Tensor::new("a", vec![m, n], data_type, true, &mut builder);
        x = Tensor::new("x", vec![n], data_type, true, &mut builder);
        y = Tensor::new("y", vec![m], data_type, false, &mut builder);
        builder.get()
    };
    assert!(m >= (1 << 13));
    assert!(n >= (1 << 4));
    // TODO(search_space): try independent tiling on `m` and `n`
    let tilings = par_iter_product(0i32..10, 0i32..5);//.cartesian_product(0..5);
    //let tilings = std::iter::once(((0, 0), 0));
    let candidates = tilings.map(|tile_m| {
        let m_tiling = &cleanup_tiling(&[1 << tile_m.0, 1 << tile_m.1]);
        let n_tiling = &cleanup_tiling(&[1 << tile_m.0]);

        let mut builder = helper::Builder::new(&signature, context.device());
        let ld_x = x.load(&[n_tiling], &mut builder);
        let ld_a = a.load(&[m_tiling, n_tiling], &mut builder);
        let init_dim_m = builder.open_mapped_dim(&ld_a[0]);
        let init = builder.mov(&0f32);
        let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
        let acc_dim_n = builder.open_mapped_dim(&ld_x[0]);
        let a_op = ld_a.dim_map(&[&acc_dim_m, &acc_dim_n], GlobalScope, &mut builder);
        let x_op = ld_x.dim_map(&[&acc_dim_n], GlobalScope, &mut builder);
        let acc = builder.mad(&a_op, &x_op, &helper::Reduce(init));
        builder.close_dim(&acc_dim_n);
        let sum = VirtualTensor::new(acc, vec![acc_dim_m]);
        let st_y = sum.store(&y, &mut builder);

        builder.order(&acc_dim_n, &st_y.inst(), search_space::Order::BEFORE);
        builder.get()
    }).collect();
    gen_best(candidates, &context, &file_name("mv", data_type, &[m, n], generic));
}


// FIXME: Polybench
// mm, 2mm, gemm, gemv
// gesummv, gemver, atax, bicg, doitgen, mvt > might want to load twice separately
// floyd-warshall, nbody single pass on K
// FC/FC+relu/FC+relu+Pooling/FC+relu+FC/FC+relu+polling+FC
