#![cfg(feature = "mppa")]
extern crate env_logger;
extern crate libc;
extern crate telamon;
extern crate tempdir;

mod common;

use common::*;
use telamon::device::{self, Context};
use telamon::helper;
use telamon::ir;

/// Executes an empty kernel.
#[test]
fn execute_empty() {
    let ref context = device::mppa::Context::new();
    let signature = empty_signature(0);
    let builder = helper::Builder::new(&signature, context.device());
    gen_best(context, builder.get());
}

/// Compiles a kernel with only instructions.
#[test]
fn execute_instructions() {
    let ref mut context = device::mppa::Context::new();
    let mem_id;
    let signature = {
        let mut builder = helper::SignatureBuilder::new("instructions", context);
        mem_id = builder.array("a", 8);
        builder.param("b", 42i32);
        builder.get()
    };
    let mut builder = helper::Builder::new(&signature, context.device());
    let add = builder.add(&0i32, &"b");
    let _sub = builder.sub(&2f32, &-3.14f32);
    let mul = builder.mul(&add, &add);
    let mul_wide = builder.mul_ex(&mul, &add, ir::Type::I(64));
    let _mad = builder.mad(&add, &mul, &mul);
    let _mad_wide = builder.mad(&add, &mul, &1i64);
    let _div = builder.div(&2i8, &3i8);
    let mov = builder.mov(&42i16);
    let _ptr = builder.cast(&mov, ir::Type::PtrTo(mem_id));
    let pattern = builder.unknown_access_pattern(mem_id);
    builder.ld(ir::Type::F(64), &"a", pattern.clone());
    builder.st(&"a", &mul_wide, pattern);
    gen_best(context, builder.get());
}

/// Compiles a kernel with a plain loop.
#[test]
fn execute_loop() {
    let ref mut context = device::mppa::Context::new();
    let signature = {
        let mut builder = helper::SignatureBuilder::new("loops", context);
        builder.param("a", 42);
        builder.get()
    };
    let mut builder = helper::Builder::new(&signature, context.device());
    let d0_size = builder.param_size("a");
    let d0 = builder.open_dim_ex(d0_size, ir::dim::kind::LOOP);
    let d1 = builder.open_dim_ex(ir::Size::Constant(32), ir::dim::kind::UNROLL);
    let d2_size = builder.tile_size("a", 4);
    let d2 = builder.open_dim_ex(d2_size, ir::dim::kind::LOOP);
    let _ = builder.mad(&d0, &d1, &d2);
    builder.order(d0.into(), d1.into(), ir::choices::order::OUTER);
    builder.order(d1.into(), d2.into(), ir::choices::order::OUTER);
    gen_best(context, builder.get());
}

/// Compiles a kernel with induction variables.
#[test]
fn execute_induction_vars() {
    let ref context = device::mppa::Context::new();
    let signature = empty_signature(0);
    let mut builder = helper::Builder::new(&signature, context.device());
    let base = builder.mov(&0i32);
    let d0 = builder.open_dim_ex(ir::Size::Constant(10), ir::dim::kind::UNROLL);
    let d1 = builder.open_dim_ex(ir::Size::Constant(11), ir::dim::kind::LOOP);
    let _ = builder.add(&(d0, 2, base), &(d1, 3, base));
    builder.order(d0.into(), d1.into(), ir::choices::order::OUTER);
    gen_best(context, builder.get());
}

/// Compiles a kernel with a temporary, privatized memory block.
#[test]
#[ignore]
fn execute_tmp_mem() {
    let ref context = device::mppa::Context::new();
    let signature = empty_signature(0);
    let builder = helper::Builder::new(&signature, context.device());
    gen_best(context, builder.get());
    unimplemented!(); // FIXME: tread dim, 1 seq dim that store inside
}

/// Compiles a kernel with thread dimensions.
#[test]
fn execute_thread_dims() {
    let ref context = device::mppa::Context::new();
    let signature = empty_signature(0);
    let mut builder = helper::Builder::new(&signature, context.device());
    let base = builder.mov(&0i32);
    let d0 = builder.open_dim_ex(ir::Size::Constant(2), ir::dim::kind::THREAD);
    let d1 = builder.open_dim_ex(ir::Size::Constant(4), ir::dim::kind::THREAD);
    let _ = builder.add(&(d1, 2, base), &d1);
    builder.order(base.into(), d0.into(), ir::choices::order::BEFORE);
    builder.order(d0.into(), d1.into(), ir::choices::order::OUTER);
    gen_best(context, builder.get());
}

/// Compiles a kernel with a barrier.
#[test]
#[ignore]
fn execute_barrier() {
    let ref context = device::mppa::Context::new();
    let signature = empty_signature(0);
    let mut builder = helper::Builder::new(&signature, context.device());
    let d0 = builder.open_dim_ex(ir::Size::Constant(4), ir::dim::kind::THREAD);
    builder.add(&d0, &1i32);
    builder.close_dims(&[d0]);
    let d1 = builder.open_dim_ex(ir::Size::Constant(8), ir::dim::kind::THREAD);
    builder.add(&d1, &1i32);
    builder.order(d0.into(), d1.into(), ir::choices::order::BEFORE);
    gen_best(context, builder.get());
}
