//! Contains integration tests for Exhaust.
extern crate env_logger;
extern crate libc;
extern crate telamon;
extern crate telamon_utils as utils;
#[macro_use]
extern crate log;

mod common;

use common::*;
use telamon::device::Context;
use telamon::helper;
use telamon::ir::{self, Size, Type};
use telamon::search_space::*;

/// Obtains the best implementation for an empty function.
#[test]
fn empty() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    gen_best(
        &context,
        helper::Builder::new(&signature, context.device()).get(),
    );
}

/// Obtains the best implementation for two add instructions.
#[test]
fn two_add() {
    let _ = env_logger::try_init();
    let mut context = fake::Context::default();
    let signature = {
        let mut builder = helper::SignatureBuilder::new("two_add", &mut context);
        builder.scalar("a", 42);
        builder.get()
    };
    gen_best(&context, {
        let mut builder = helper::Builder::new(&signature, context.device());
        builder.add(&"a", &2);
        builder.add(&std::f32::consts::PI, &1.0f32);
        builder.get()
    });
}

/// Ensures the default order between instructions and dimensions is good.
#[test]
fn inst_dim_order() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    let dim0 = builder.open_dim(Size::new_const(64));
    let inst0 = builder.mov(&0i32);
    let pattern = ir::AccessPattern::Unknown(None);
    let addr = builder.cast(&0i64, pattern.pointer_type(context.device()));
    let inst1 = builder.st(&addr, &0i32, pattern);
    builder.close_dim(&dim0);
    let dim1 = builder.open_dim(Size::new_const(64));
    let _ = builder.mov(&0i32);
    let space = builder.get();
    assert_eq!(space.domain().get_dim_kind(dim0[0]), !DimKind::VECTOR);
    assert_eq!(space.domain().get_dim_kind(dim1[0]), !DimKind::VECTOR);
    assert_eq!(
        space.domain().get_is_iteration_dim(inst0, dim0[0]),
        Bool::TRUE
    );
    assert_eq!(
        space.domain().get_is_iteration_dim(inst0, dim0[0]),
        Bool::TRUE
    );
    assert_eq!(
        space.domain().get_order(inst0.into(), dim1[0].into()),
        Order::INNER | Order::ORDERED
    );
    assert_eq!(
        space.domain().get_is_iteration_dim(inst1, dim1[0]),
        Bool::FALSE
    );
    assert_eq!(
        space.domain().get_order(inst1.into(), dim1[0].into()),
        Order::INNER | Order::ORDERED
    );
    gen_best(&context, space);
}

#[test]
/// Ensures oredering contraints for `ir::VarDef::Inst` are respected.
fn inst_variable_order() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    let src = builder.mov(&1f32);
    let var = builder.get_inst_variable(src);
    let dst = builder.mov(&var);
    let space = builder.get();
    assert_eq!(
        space.domain().get_order(src.into(), dst.into()),
        Order::BEFORE
    );
    gen_best(&context, space);
}

/// Ensures oredering contraints for `ir::VarDef::DimMap` are respected.
#[test]
fn dim_map_variable_order() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature(0);
    let mut builder = helper::Builder::new(&signature, context.device());

    let src_dim = builder.open_dim(ir::Size::new_const(16));
    let src = builder.mov(&1f32);
    let src_var = builder.get_inst_variable(src);
    let dst_dim = builder.open_mapped_dim(&src_dim);
    let dst_var = builder.create_dim_map_variable(src_var, &[(&src_dim, &dst_dim)]);
    let dst = builder.mov(&dst_var);
    // Ensure ordering constraints are respected.
    let space = builder.get_clone();
    assert_eq!(
        space.domain().get_order(src.into(), dst.into()),
        Order::BEFORE
    );
    assert_eq!(
        space
            .domain()
            .get_order(src_dim[0].into(), dst_dim[0].into()),
        Order::BEFORE | Order::MERGED
    );
    // Ensure point-to-point communication is enforced by merging dimensions if it cannot
    // use different register names along the dimensions.
    builder.action(Action::DimKind(src_dim[0], DimKind::LOOP));
    let space = builder.get();
    assert_eq!(
        space
            .domain()
            .get_order(src_dim[0].into(), dst_dim[0].into()),
        Order::MERGED
    );

    gen_best(&context, space);
}

/// Ensures oredering contraints for `ir::VarDef::Last` are respected.
#[test]
fn last_variable_order() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature(0);
    let mut builder = helper::Builder::new(&signature, context.device());

    let dim = builder.open_dim(ir::Size::new_const(16));
    let src = builder.mov(&1f32);
    let src_var = builder.get_inst_variable(src);
    builder.close_dim(&dim);
    let last_var = builder.create_last_variable(src_var, &[&dim]);
    let dst = builder.mov(&last_var);

    let space = builder.get();
    assert_eq!(
        space.domain().get_order(src.into(), dst.into()),
        Order::BEFORE
    );
    assert_eq!(
        space.domain().get_order(dim[0].into(), dst.into()),
        Order::BEFORE
    );
}

/// Ensures nested thread dimensions are packed and that their number is limited.
#[test]
fn nested_thread_dims() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    let size4 = builder.cst_size(4);
    let d0 = builder.open_dim_ex(size4.clone(), DimKind::THREAD);
    let d1 = builder.open_dim_ex(size4.clone(), DimKind::THREAD);
    let d2 = builder.open_dim_ex(size4, DimKind::THREAD);
    let size512 = builder.cst_size(512);
    let d3 = builder.open_dim(size512);
    builder.mov(&0i32);
    builder.order(&d0, &d3, Order::INNER);
    let space = builder.get();
    assert!(
        !space
            .domain()
            .get_dim_kind(d3[0])
            .intersects(DimKind::THREAD)
    );
    assert_eq!(
        space.domain().get_order(d0[0].into(), d3[0].into()),
        Order::INNER
    );
    assert_eq!(
        space.domain().get_order(d1[0].into(), d3[0].into()),
        Order::INNER
    );
    assert_eq!(
        space.domain().get_order(d2[0].into(), d3[0].into()),
        Order::INNER
    );
    gen_best(&context, space);
}

/// Ensures the maximal number of threads is respected when adding an instruction.
#[test]
fn max_thread_on_addinst() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    builder.open_dim_ex(Size::new_const(1024), DimKind::THREAD);
    let d1 = builder.open_dim(Size::new_const(2));
    builder.mov(&0i32);
    let space = builder.get();
    assert!(
        !space
            .domain()
            .get_dim_kind(d1[0])
            .intersects(DimKind::THREAD)
    );
    gen_best(&context, space);
}

/// Ensures the maximal number of thread is respected when setting a kind.
#[test]
fn max_thread_on_setkind() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    let d0 = builder.open_dim(Size::new_const(1024));
    let d1 = builder.open_dim(Size::new_const(2));
    builder.mov(&0i32);
    builder.action(Action::DimKind(d0[0], DimKind::THREAD));
    let space = builder.get();
    assert!(
        !space
            .domain()
            .get_dim_kind(d1[0])
            .intersects(DimKind::THREAD)
    );
    gen_best(&context, space);
}

/// Ensures block dimensions are nested under every other dimension.
#[test]
fn block_dims() {
    let _ = env_logger::try_init();
    let mut context = fake::Context::default();
    let n;
    let signature = {
        let mut builder = helper::SignatureBuilder::new("block_dims", &mut context);
        n = builder.max_size("n", 64);
        builder.get()
    };
    let mut builder = helper::Builder::new(&signature, context.device());
    let d0 = builder.open_dim(Size::new_const(4));
    let inst = builder.mov(&0i32);
    let s1 = n.into_ir_size(&builder);
    let d1 = builder.open_dim_ex(s1, DimKind::BLOCK);
    let d2 = builder.open_dim_ex(Size::new_const(2), DimKind::BLOCK);
    let d3 = builder.open_dim_ex(Size::new_const(3), DimKind::BLOCK);
    let space = builder.get();
    assert_eq!(
        space.domain().get_is_iteration_dim(inst.into(), d0[0]),
        Bool::TRUE
    );
    assert_eq!(
        space.domain().get_is_iteration_dim(inst.into(), d1[0]),
        Bool::TRUE
    );
    assert_eq!(
        space.domain().get_is_iteration_dim(inst.into(), d2[0]),
        Bool::TRUE
    );
    assert_eq!(
        space.domain().get_is_iteration_dim(inst.into(), d3[0]),
        Bool::TRUE
    );
    assert_eq!(
        space.domain().get_dim_kind(d0[0]),
        DimKind::LOOP | DimKind::THREAD | DimKind::UNROLL
    );
    assert_eq!(
        space.domain().get_order(d1[0].into(), d2[0].into()),
        Order::NESTED
    );
    assert_eq!(
        space.domain().get_order(d1[0].into(), d3[0].into()),
        Order::NESTED
    );
    assert_eq!(
        space.domain().get_order(d2[0].into(), d3[0].into()),
        Order::NESTED
    );
    gen_best(&context, space);
}

/// Ensures vector dimensions have the correct restrictions.
#[test]
fn vector_dims() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    let base_addr = builder.cast(&0i64, context.device().pointer_type(MemSpace::GLOBAL));
    let d0 = builder.open_dim(Size::new_const(4));
    // Test with one vectorizable instruction
    let (addr, pattern) = builder.tensor_access(&base_addr, None, Type::I(8), &[&d0]);
    builder.ld(Type::I(8), &addr, pattern.clone());
    assert!(
        builder
            .get_clone()
            .domain()
            .get_dim_kind(d0[0])
            .intersects(DimKind::VECTOR)
    );
    // Test with two insts and a non-vectorizable inst.
    builder.ld(Type::I(8), &addr, pattern);
    builder.close_dim(&d0);
    let d1 = builder.open_dim(Size::new_const(4));
    builder.mul(&0i32, &0i32);
    builder.close_dim(&d1);
    let space = builder.get();
    assert!(
        !space
            .domain()
            .get_dim_kind(d0[0])
            .intersects(DimKind::VECTOR)
    );
    assert!(
        !space
            .domain()
            .get_dim_kind(d1[0])
            .intersects(DimKind::VECTOR)
    );
    gen_best(&context, space);
}

/// Ensure restrictions are applied to unrolled dimensions.
#[test]
fn unroll_dims() {
    let _ = env_logger::try_init();
    let mut context = fake::Context::default();
    let n;
    let signature = {
        let mut builder = helper::SignatureBuilder::new("unroll_dims", &mut context);
        n = builder.max_size("n", 64);
        builder.get()
    };
    let mut builder = helper::Builder::new(&signature, context.device());
    let d0 = builder.open_dim(Size::new_const(64));
    let d1 = builder.open_dim(Size::new_const(4096));
    let s2 = n.into_ir_size(&builder);
    let d2 = builder.open_dim(s2);
    builder.mov(&0i32);
    let space = builder.get();
    assert!(space.domain().get_dim_kind(d0[0]).contains(DimKind::UNROLL));
    assert!(!space.domain().get_dim_kind(d1[0]).contains(DimKind::UNROLL));
    assert!(!space.domain().get_dim_kind(d2[0]).contains(DimKind::UNROLL));
    gen_best(&context, space);
}

/// Ensures the invariants on dimensions that carry reductions are respected.
#[test]
fn reduce_dim_invariants() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    let init = builder.cast(&0i64, context.device().pointer_type(MemSpace::GLOBAL));
    let d0 = builder.open_dim(Size::new_const(4));
    let pattern = ir::AccessPattern::Unknown(None);
    let reduce = builder.ld(Type::I(64), &helper::Reduce(init), pattern);
    builder.close_dim(&d0);

    let d1 = builder.open_dim(Size::new_const(4));
    builder.action(Action::IsIterationDim(reduce.into(), d1[0], Bool::TRUE));
    builder.close_dim(&d1);
    let d2 = builder.open_dim(Size::new_const(4));
    builder.mov(&0i32);

    builder.order(&d2, &init, !Order::OUTER);
    let space = builder.get();
    assert_eq!(
        space.domain().get_dim_kind(d0[0]),
        DimKind::LOOP | DimKind::UNROLL
    );
    assert_eq!(
        space.domain().get_order(d0[0].into(), init.into()),
        Order::AFTER
    );
    assert!(Order::OUTER.contains(space.domain().get_order(d1[0].into(), init.into())));
    assert_eq!(
        space.domain().get_is_iteration_dim(reduce.into(), d2[0]),
        Bool::FALSE
    );
    gen_best(&context, space);
}

/// Ensures the renaming triggered by a thread creation work properly.
#[test]
fn rename_thread() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    let d_n_1 = &builder.open_dim_ex(Size::new_const(8), DimKind::THREAD);
    builder.mov(&0i32);
    builder.mov(d_n_1);
    gen_best(&context, builder.get());
}

/// Ensures dimension merging occurs corectly.
#[test]
fn dim_merge() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    let d0 = builder.open_dim_ex(Size::new_const(4), DimKind::LOOP);
    builder.mov(&0i32);
    let d1 = builder.open_dim_ex(Size::new_const(4), DimKind::LOOP);
    builder.order(&d0, &d1, Order::MERGED);
    gen_best(&context, builder.get());
}

/// Ensure loop fusion, with dependent instructions, occurs correctly.
#[test]
fn loop_fusion() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    let d0 = builder.open_dim_ex(Size::new_const(4), DimKind::LOOP);
    let inst0 = builder.mov(&0i32);
    let d1 = builder.open_mapped_dim(&d0);
    builder.mov(&inst0);
    builder.order(&d0, &d1, Order::MERGED);
    // Ensure no temporary memory has been generated.
    let instance = builder.get();
    assert_eq!(instance.ir_instance().insts().count(), 2);
    gen_best(&context, instance);
}

/// Ensure un-fused unrolled loops are correctly handled.
#[test]
fn unrolled_loop_unfused_simple() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    let d0 = builder.open_dim_ex(Size::new_const(4), DimKind::UNROLL);
    let inst0 = builder.mov(&0i32);
    let d1 = builder.open_mapped_dim(&d0);
    builder.mov(&inst0);
    builder.order(&d0, &d1, !Order::MERGED);
    // Ensure no temporary memory has been generated.
    let instance = builder.get();
    assert_eq!(instance.ir_instance().insts().count(), 2);
    gen_best(&context, instance);
}

/// Ensure temporary memory is generated when needed.
#[test]
fn temporary_memory_gen_simple() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    let d0 = builder.open_dim_ex(Size::new_const(4), DimKind::LOOP);
    let inst0 = builder.mov(&0i32);
    let d1 = builder.open_mapped_dim(&d0);
    builder.mov(&helper::TmpArray(inst0));
    builder.order(&d0, &d1, !Order::MERGED);
    // Ensure load and store instruction have been generated.
    let instance = builder.get();
    assert!(instance.ir_instance().insts().count() >= 4);
    gen_best(&context, instance);
}

/// Ensures un-fused loops are correctly handle in persence of reduction.
#[test]
fn unrolled_loop_unfused_reduction() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    let d0 = builder.open_dim_ex(ir::Size::new_const(4), DimKind::UNROLL);
    let inst0 = builder.mov(&0i32);
    builder.open_mapped_dim(&d0);
    let d1 = builder.open_dim_ex(ir::Size::new_const(1024), DimKind::LOOP);
    builder.mov(&helper::Reduce(inst0));

    builder.order(&d0, &d1, Order::BEFORE);
    // Ensure not temporary memory has been generated.
    let instance = builder.get();
    assert_eq!(instance.ir_instance().insts().count(), 2);
    gen_best(&context, instance);
}

#[test]
fn two_thread_dim_map() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    // Generate a variable in each thread.
    let dim0_0 = builder.open_dim_ex(ir::Size::new_const(32), DimKind::THREAD);
    let dim0_1 = builder.open_dim_ex(ir::Size::new_const(32), DimKind::THREAD);
    let x = builder.mov(&0i32);
    // Transpose twice the variable using temporary memory.
    let dim1_0 = builder.open_mapped_dim(&dim0_1);
    let dim1_1 = builder.open_mapped_dim(&dim0_0);
    builder.mov(&helper::TmpArray(x));
    // Set the nesting order.
    builder.order(&dim0_0, &dim0_1, Order::OUTER);
    builder.order(&dim1_0, &dim1_1, Order::OUTER);
    gen_best(&context, builder.get());
}

#[test]
fn double_dim_map() {
    // FIXME: investigate Failed lowering that should be cut earlier
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();

    let mut builder = helper::Builder::new(&signature, context.device());
    // Load from a and b.
    let dim0_0 = builder.open_dim_ex(ir::Size::new_const(32), DimKind::THREAD);
    let dim0_1 = builder.open_dim_ex(ir::Size::new_const(32), DimKind::THREAD);
    let dim0_2 = builder.open_dim_ex(ir::Size::new_const(4), DimKind::UNROLL);
    let x = builder.mov(&0i32);
    // Transpose and add a and b. Store the result in a.
    let dim1_0 = builder.open_mapped_dim(&dim0_1);
    let dim1_1 = builder.open_mapped_dim(&dim0_0);
    let dim1_2 = builder.open_mapped_dim(&dim0_2);
    builder.mov(&x);
    builder.mov(&x);
    // Fix the nesting order.
    builder.order(&dim0_0, &dim0_1, Order::OUTER);
    builder.order(&dim0_1, &dim0_2, Order::OUTER);
    builder.order(&dim1_0, &dim1_1, Order::OUTER);
    builder.order(&dim1_1, &dim1_2, Order::OUTER);

    //gen_best(&context, builder.get());
}

/// Ensures mapping multiple dimensions to the same vectorization level works.
#[test]
fn multi_dim_to_same_vector_level() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    builder.open_dim_ex(ir::Size::new_const(2), DimKind::INNER_VECTOR);
    builder.open_dim_ex(ir::Size::new_const(4), DimKind::INNER_VECTOR);
    let inner = builder.open_dim_ex(ir::Size::new_const(4), DimKind::VECTOR);
    builder.add(&0i32, &0i32);

    // Ensure the search space is valid.
    let space = builder.get();
    // Ensure the total vector size constraint is respected.
    assert_eq!(space.domain().get_dim_kind(inner[0]), DimKind::OUTER_VECTOR);
    // Try to generate a specfied candidate.
    gen_best(&context, space);
}

/// Ensures the two levels of vectorization work correctly.
#[test]
fn two_level_vectorization() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    let inner_vec = builder.open_dim_ex(ir::Size::new_const(2), DimKind::INNER_VECTOR);
    let outer_vec = builder.open_dim_ex(ir::Size::new_const(2), DimKind::OUTER_VECTOR);
    let not_a_vec = builder.open_dim_ex(ir::Size::new_const(2), DimKind::LOOP);
    builder.add(&0i32, &0i32);
    // Ensure the search space is valid.
    let space = builder.get();
    // Ensure nesting constraints are enforced.
    let not_a_vec_id = not_a_vec[0].into();
    let outer_vec_id = outer_vec[0].into();
    let inner_vec_id = inner_vec[0].into();
    assert_eq!(
        space.domain().get_order(not_a_vec_id, outer_vec_id),
        Order::OUTER
    );
    assert_eq!(
        space.domain().get_order(outer_vec_id, inner_vec_id),
        Order::OUTER
    );

    // Try to generate a fully specified candidate.
    gen_best(&context, space);
}
