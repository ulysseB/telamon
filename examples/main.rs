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
    let tilings = par_iter_product(0i32..8, 0i32..5);
    //let tilings = std::iter::once((5, 2));
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
        // TODO(search_space): explore inst flags
        builder.action(Action::InstFlag(ld_x.inst(), search_space::InstFlag::MEM_CG));
        builder.action(Action::InstFlag(ld_a.inst(), search_space::InstFlag::MEM_CG));
        builder.action(Action::InstFlag(st_y.inst(), search_space::InstFlag::MEM_CS));
        builder.get()
    }).collect();
    gen_best(candidates, &context, &file_name("mv", data_type, &[m, n], generic));
}

/// Generates code for "y = (_alpha.A + _beta.B).x"
fn gesummv(m: i32, n: i32, data_type: ir::Type, generic: bool, executor: &cuda::Executor) {
    let (a, b, x, y);
    let mut context = cuda::Context::new(&executor);
    let signature = {
        let mut builder = helper::SignatureBuilder::new("gesummv", &mut context);
        builder.param("alpha", 0f32);
        builder.param("beta", 0f32);
        let m = create_size(m, "m", generic, &mut builder);
        let n = create_size(n, "n", generic, &mut builder);
        a = Tensor::new("a", vec![m, n], data_type, true, &mut builder);
        b = Tensor::new("b", vec![m, n], data_type, true, &mut builder);
        x = Tensor::new("x", vec![n], data_type, true, &mut builder);
        y = Tensor::new("y", vec![m], data_type, false, &mut builder);
        builder.get()
    };

    // TODO(search_space): try independent tiling on `m` and `n`
    assert!(m >= (1 << 13));
    assert!(n >= (1 << 4));
    let tilings = par_iter_product(0i32..8, 0i32..5);
    //let tilings = std::iter::once(((0, 0), 0));
    let candidates = tilings.map(|tile_m| {
        let m_tiling = &cleanup_tiling(&[1 << tile_m.0, 1 << tile_m.1]);
        let n_tiling = &cleanup_tiling(&[1 << tile_m.0]);

        let mut builder = helper::Builder::new(&signature, context.device());
        let ld_x = x.load(&[n_tiling], &mut builder);
        let ld_a = a.load(&[m_tiling, n_tiling], &mut builder);
        let ld_b = b.load(&[m_tiling, n_tiling], &mut builder);
        let init_dim_m = builder.open_mapped_dim(&ld_a[0]);
        let init_a = builder.mov(&0f32);
        let init_b = builder.mov(&0f32);
        let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
        let acc_dim_n = builder.open_mapped_dim(&ld_x[0]);
        let a_op = ld_a.dim_map(&[&acc_dim_m, &acc_dim_n], GlobalScope, &mut builder);
        let b_op = ld_b.dim_map(&[&acc_dim_m, &acc_dim_n], GlobalScope, &mut builder);
        let x_op = ld_x.dim_map(&[&acc_dim_n], GlobalScope, &mut builder);
        let acc_a = builder.mad(&a_op, &x_op, &helper::Reduce(init_a));
        let acc_b = builder.mad(&b_op, &x_op, &helper::Reduce(init_b));
        builder.close_dim(&acc_dim_n);
        let y_a = builder.mul(&acc_a, &"alpha");
        let sum = builder.mad(&acc_b, &"beta", &y_a);
        let sum = VirtualTensor::new(sum, vec![acc_dim_m]);
        let st_y = sum.store(&y, &mut builder);
        
        builder.order(&acc_dim_n, &y_a, search_space::Order::BEFORE);
        // TODO(search_space): explore inst flags
        builder.action(Action::InstFlag(ld_x.inst(), search_space::InstFlag::MEM_CG));
        builder.action(Action::InstFlag(ld_a.inst(), search_space::InstFlag::MEM_CG));
        builder.action(Action::InstFlag(ld_b.inst(), search_space::InstFlag::MEM_CG));
        builder.action(Action::InstFlag(st_y.inst(), search_space::InstFlag::MEM_CS));
        builder.get()
    }).collect();
    gen_best(candidates, &context, &file_name("gesummv", data_type, &[m, n], generic));
}

fn mm(m: i32, n: i32, k: i32,
      data_type: ir::Type,
      generic: bool,
      executor: &cuda::Executor) {
    let mut context = cuda::Context::new(&executor);
    let (a, b, c);
    let signature = &{
        let mut builder = helper::SignatureBuilder::new("mm", &mut context);
        let m = create_size(m, "m", generic, &mut builder);
        let n = create_size(n, "n", generic, &mut builder);
        let k = create_size(k, "k", generic, &mut builder);
        a = Tensor::new("a", vec![m, k], data_type, true, &mut builder);
        b = Tensor::new("b", vec![k, n], data_type, true, &mut builder);
        c = Tensor::new("c", vec![m, n], data_type, false, &mut builder);
        builder.get()
    };

    // TODO(search_space): more tilings
    let tilings = (0..6).into_par_iter().flat_map(|t1| {
        (0..std::cmp::min(8-t1, 5)).into_par_iter()
            .map(move |t2| (1u32 << t1, 1u32 << t2))
    });
    //let tilings = std::iter::once((32, 4));
    let candidates = tilings.map(|(tile_1, tile_2)| {
        let full_tiling = cleanup_tiling(&[tile_1, tile_2]);
        let small_tiling = cleanup_tiling(&[tile_1]); 
        let mut builder = helper::Builder::new(signature, context.device());

        let ld_a = a.load(&[&full_tiling, &small_tiling], &mut builder);
        let ld_b = b.load(&[&small_tiling, &full_tiling], &mut builder);

        let init_dim_m = builder.open_mapped_dim(&ld_a[0]);
        let init_dim_n = builder.open_mapped_dim(&ld_b[1]);
            let acc_init = builder.mov(&0f32);
        let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
        let acc_dim_n = builder.open_mapped_dim(&init_dim_n);
        let acc_dim_k = builder.open_mapped_dim(&ld_a[1]);
            let a_op = ld_a.dim_map(&[&acc_dim_m, &acc_dim_k], GlobalScope, &mut builder);
            let b_op = ld_b.dim_map(&[&acc_dim_k, &acc_dim_n], GlobalScope, &mut builder);
            let acc = builder.mad(&a_op, &b_op, &helper::Reduce(acc_init));
        builder.close_dim(&acc_dim_k);

        let acc = VirtualTensor::new(acc, vec![acc_dim_n, acc_dim_m]);
        let st_c = acc.store(&c, &mut builder);

        // Order for correctness.
        builder.order(&st_c.inst(), &acc_dim_k, Order::AFTER);
        // Arbitrary constrains to reduce the search space
        // TODO(search_space): remove arbitrary decisions.
        builder.action(Action::InstFlag(ld_a.inst(), InstFlag::MEM_CG | InstFlag::MEM_NC));
        builder.action(Action::InstFlag(ld_b.inst(), InstFlag::MEM_CG | InstFlag::MEM_NC));
        builder.action(Action::InstFlag(st_c.inst(), InstFlag::MEM_CS));

        builder.action(Action::DimKind(init_dim_n[0], DimKind::BLOCK));
        builder.action(Action::DimKind(init_dim_m[0], DimKind::BLOCK));
        builder.get()
    }).collect();
    gen_best(candidates, &context, &file_name("mm", data_type, &[m, n, k], generic));
}

/// Computes "B[r] = A[r].transpose(X)" where A, B are 3D tensors.
fn doitgen(p: i32, q: i32, r: i32,
           data_type: ir::Type,
           generic: bool,
           executor: &cuda::Executor) {
    let mut context = cuda::Context::new(&executor);
    let (a, b, x);
    let signature = &{
        let mut builder = helper::SignatureBuilder::new("mm", &mut context);
        let p = create_size(p, "p", generic, &mut builder);
        let q = create_size(q, "q", generic, &mut builder);
        let r = create_size(r, "r", generic, &mut builder);
        a = Tensor::new("a", vec![r, q, p], data_type, true, &mut builder);
        b = Tensor::new("b", vec![r, q, p], data_type, false, &mut builder);
        x = Tensor::new("x", vec![p, p], data_type, true, &mut builder);
        builder.get()
    };
    // TODO(search_space): explore more tilings
    let tilings = par_iter_product(0..5, 0..5);
    //let tilings = std::iter::once((4, 4));
    let candidates = tilings.map(|(tile_s, tile_r)| {
        let mut builder = helper::Builder::new(signature, context.device());
        let tile_p = 2; // TODO(search_space): explore tile_p
        let p_tiling = &cleanup_tiling(&[1u32 << tile_s, 1u32 << tile_p]);
        let q_tiling = &cleanup_tiling(&[1u32 << tile_s, 1u32 << tile_p]);
        let r_tiling = &cleanup_tiling(&[1u32 << tile_r]);
        let s_tiling = &cleanup_tiling(&[1u32 << tile_s]);
        let ld_a = a.load(&[r_tiling, q_tiling, s_tiling], &mut builder);
        let ld_x = x.load(&[p_tiling, s_tiling], &mut builder);

        let init_dim_r = builder.open_mapped_dim(&ld_a[0]);
        let init_dim_q = builder.open_mapped_dim(&ld_a[1]);
        let init_dim_p = builder.open_mapped_dim(&ld_x[0]);
        let init = builder.mov(&0f32);
        let acc_dim_r = builder.open_mapped_dim(&init_dim_r);
        let acc_dim_q = builder.open_mapped_dim(&init_dim_q);
        let acc_dim_p = builder.open_mapped_dim(&init_dim_p);
        let acc_dim_s = builder.open_mapped_dim(&ld_x[1]);
        let a_op = ld_a.dim_map(&[&acc_dim_r, &acc_dim_q, &acc_dim_s], GlobalScope, &mut builder);
        let x_op = ld_x.dim_map(&[&acc_dim_p, &acc_dim_s], GlobalScope, &mut builder);
        let acc = builder.mad(&a_op, &x_op, &helper::Reduce(init));
        builder.close_dim(&acc_dim_s);

        let res = VirtualTensor::new(acc, vec![acc_dim_r, acc_dim_q, acc_dim_p]);
        let st_b = res.store(&b, &mut builder);
        builder.order(&acc, &st_b.inst(), Order::BEFORE);
        // TODO(search_space): remove arbitrary decisions.
        builder.action(Action::InstFlag(ld_a.inst(), InstFlag::MEM_CG));
        builder.action(Action::InstFlag(ld_x.inst(), InstFlag::MEM_CG));
        builder.action(Action::InstFlag(st_b.inst(), InstFlag::MEM_CS));

        builder.get()
    }).collect();
    gen_best(candidates, &context, &file_name("doitgen", data_type, &[p, q, r], generic));
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
