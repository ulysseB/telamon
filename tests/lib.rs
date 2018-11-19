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
use telamon::ir::{self, Size, Type};
use telamon::search_space::*;
use telamon::{codegen, helper};

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
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    let src_dim = builder.open_dim(ir::Size::new_const(16));
    let src = builder.mov(&1f32);
    let src_var = builder.get_inst_variable(src);
    let dst_dim = builder.open_mapped_dim(&src_dim);
    let def_mode = ir::VarDefMode::InPlace { allow_sync: false };
    let use_mode = ir::VarUseMode::FromRegisters;
    let dst_var = builder.create_dim_map_variable(
        src_var,
        &[(&src_dim, &dst_dim)],
        def_mode,
        use_mode,
    );
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
    let signature = empty_signature();
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
    let ptr_type = context.device().pointer_type(ir::MemorySpace::Global);
    let base_addr = builder.cast(&0i64, ptr_type);
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
    let ptr_type = context.device().pointer_type(ir::MemorySpace::Global);
    let init = builder.cast(&032, ptr_type);
    let d0 = builder.open_dim(Size::new_const(4));
    let pattern = ir::AccessPattern::Unknown(None);
    let fby = builder.create_fby_variable(init, &[&d0]);
    let reduce = builder.ld(Type::I(32), &fby, pattern);
    builder.set_loop_carried_variable(fby, reduce);
    builder.close_dim(&d0);

    let d1 = builder.open_dim(Size::new_const(4));
    builder.action(Action::IsIterationDim(reduce.into(), d1[0], Bool::TRUE));
    builder.close_dim(&d1);

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
    let var = builder.map_instruction(
        inst0,
        &[(&d0, &d1)],
        true,
        vec![ir::MemorySpace::Shared],
    );
    builder.mov(&var);
    builder.order(&d0, &d1, !Order::MERGED);
    // Ensure load and store instruction have been generated.
    let space = builder.get();
    assert_eq!(space.ir_instance().insts().count(), 4);
    // The variable has a single instantiated dimension, so its rank must be 1.
    let layout_dim = space.ir_instance().variable(var).layout()[0];
    assert_eq!(space.domain().get_is_instantiated(layout_dim), IsInstantiated::TRUE);
    const RANK_1: NumericSet = NumericSet { enabled_values: 0b10 };
    assert_eq!(space.domain().get_rank(layout_dim), RANK_1);
    gen_best(&context, space);
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
    let fby = builder.create_fby_variable(inst0, &[&d1]);
    let inst1 = builder.mov(&fby);
    builder.set_loop_carried_variable(fby, inst1);

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
    // Transpose the variable using temporary memory.
    let dim1_0 = builder.open_mapped_dim(&dim0_1);
    let dim1_1 = builder.open_mapped_dim(&dim0_0);
    let mapping = [(&dim0_1, &dim1_0), (&dim0_0, &dim1_1)];
    let var = builder.map_instruction(x, &mapping, true, vec![ir::MemorySpace::Shared]);
    builder.mov(&var);
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

    builder.open_dim_ex(ir::Size::new_const(2), DimKind::OUTER_VECTOR);
    builder.open_dim_ex(ir::Size::new_const(4), DimKind::OUTER_VECTOR);
    let inner = builder.open_dim_ex(ir::Size::new_const(4), DimKind::VECTOR);
    builder.add(&0i32, &0i32);

    // Ensure the search space is valid.
    let space = builder.get();
    // Ensure the total vector size constraint is respected.
    assert_eq!(space.domain().get_dim_kind(inner[0]), DimKind::INNER_VECTOR);
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

/// Ensure we can create `fby` variables.
#[test]
fn simple_fby() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    let init = builder.mov(&0f32);
    let init_var = builder.get_inst_variable(init);
    let dim = builder.open_dim_ex(ir::Size::new_const(16), DimKind::LOOP);
    let fby = builder.create_fby_variable(init_var, &[&dim]);
    let acc = builder.add(&fby, &fby);
    let acc_var = builder.get_inst_variable(acc);
    builder.set_loop_carried_variable(fby, acc_var);
    builder.close_dim(&dim);
    let res = builder.create_last_variable(acc_var, &[&dim]);
    builder.mov(&res);

    let space = builder.get();
    assert_eq!(
        space.domain().get_order(init.into(), dim[0].into()),
        Order::BEFORE
    );
    // Try to generate a fully specified candidate.
    gen_best(&context, space);
}

/// Ensure the search cannot add additional fby dimensions. This would be incorrect as it
/// would apply the loop-carried computation more than necessary to the variable.
#[test]
fn no_additional_fby_dimension() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    let init = builder.mov(&0f32);
    let init_var = builder.get_inst_variable(init);
    let dim = builder.open_dim_ex(ir::Size::new_const(16), DimKind::LOOP);
    let fby = builder.create_fby_variable(init_var, &[&dim]);
    let other_dim = builder.open_dim_ex(ir::Size::new_const(16), DimKind::LOOP);
    let acc = builder.add(&fby, &fby);
    let acc_var = builder.get_inst_variable(acc);
    builder.set_loop_carried_variable(fby, acc_var);
    let space = builder.get();
    // Additional dimensions outside `acc` must also be outside `init`.
    assert_eq!(
        space.domain().get_order(init.into(), other_dim[0].into()),
        Order::INNER
    );
    // Try to generate a fully specified candidate.
    gen_best(&context, space);
}

/// Ensure we can chain `DimMap` and `Fby` variables.
#[test]
fn chained_dim_map_fby() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    // for i in 0..4:
    // . init[i] = 2.0
    let init_dim = builder.open_dim(ir::Size::new_const(4));
    let init = builder.mov(&2f32);
    let init_var = builder.get_inst_variable(init);

    // for j in 0..16:
    // . for i in 0..4:
    // . .  mul[i] = phi(init[i], sub[i]) * 2.0
    let def_mode = ir::VarDefMode::InPlace { allow_sync: true };
    let use_mode = ir::VarUseMode::FromRegisters;
    let reduction_dim = builder.open_dim_ex(ir::Size::new_const(16), DimKind::LOOP);
    let mul_dim = builder.open_mapped_dim(&init_dim);
    let mapped_init_var = builder.create_dim_map_variable(
        init_var,
        &[(&init_dim, &mul_dim)],
        def_mode,
        use_mode.clone(),
    );
    let fby = builder.create_fby_variable(mapped_init_var, &[&reduction_dim]);
    let mul = builder.mul(&fby, &2f32);
    let mul_var = builder.get_inst_variable(mul);

    // . for i in 0..4:
    // . . sub[i] = mul[i] - 1
    let sub_dim = builder.open_mapped_dim(&mul_dim);
    let mapped_mul_var = builder.create_dim_map_variable(
        mul_var,
        &[(&mul_dim, &sub_dim)],
        def_mode,
        use_mode.clone(),
    );
    let sub = builder.add(&mapped_mul_var, &1f32);
    let sub_var = builder.get_inst_variable(sub);
    let mapped_sub_var = builder.create_dim_map_variable(
        sub_var,
        &[(&sub_dim, &mul_dim)],
        def_mode,
        use_mode,
    );
    builder.set_loop_carried_variable(fby, mapped_sub_var);

    builder.order(&reduction_dim, &mul_dim, Order::OUTER);
    builder.order(&reduction_dim, &sub_dim, Order::OUTER);
    let space = builder.get();
    // Try to generate a fully specified candidate.
    gen_best(&context, space);
}

/// Ensure we can map multiple dimensions to the same vectorization level.
#[test]
fn two_dims_to_same_level_vectorization() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    let array = builder.allocate_shared(ir::Type::F(32), 1024);
    let outer_dim = builder.open_dim(ir::Size::new_const(2));
    let inner_dim = builder.open_dim(ir::Size::new_const(2));
    let (ptr, pattern) = builder.tensor_access(
        &array,
        Some(array),
        ir::Type::F(32),
        &[&outer_dim, &inner_dim],
    );
    builder.ld(ir::Type::F(32), &ptr, pattern);
    // Ensure vectorization is possible on both dimensions.
    let space = builder.get_clone();
    assert_eq!(space.domain().get_dim_kind(inner_dim[0]), DimKind::ALL);
    assert_eq!(space.domain().get_dim_kind(outer_dim[0]), DimKind::ALL);
    // Ensure ranks and strided flags are correctly set.
    let fun = space.ir_instance();
    let outer_layout_dim = fun.layout_dimension(fun.dim(outer_dim[0]).layout_dims()[0]);
    let inner_layout_dim = fun.layout_dimension(fun.dim(inner_dim[0]).layout_dims()[0]);
    assert!(!outer_layout_dim.is_strided());
    assert!(!inner_layout_dim.is_strided());
    assert_eq!(&outer_layout_dim.possible_ranks().unwrap()[..], &[2]);
    assert_eq!(&inner_layout_dim.possible_ranks().unwrap()[..], &[1]);
    // Ensure dimensions are correctly ordered if we vectorize the outer one.
    let mut builder = builder.clone();
    builder.action(Action::DimKind(outer_dim[0], DimKind::INNER_VECTOR));
    let space = builder.get();
    assert_eq!(
        space.domain().get_dim_kind(inner_dim[0]),
        DimKind::INNER_VECTOR
    );
    assert_eq!(
        space
            .domain()
            .get_order(inner_dim[0].into(), outer_dim[0].into()),
        Order::INNER
    );
}

/// Ensure we cannot map non-contiguous dimension to the same vectorization level
#[test]
fn non_contiguous_vector_dims() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    let array = builder.allocate_shared(ir::Type::F(32), 1024);
    let outer_dim = builder.open_dim_ex(ir::Size::new_const(2), DimKind::OUTER_VECTOR);
    let mid_dim = builder.open_dim(ir::Size::new_const(2));
    let inner_dim = builder.open_dim_ex(ir::Size::new_const(2), DimKind::OUTER_VECTOR);
    builder.order(&inner_dim, &mid_dim, Order::INNER);
    builder.order(&mid_dim, &outer_dim, Order::INNER);
    let (ptr, pattern) = builder.tensor_access(
        &array,
        Some(array),
        ir::Type::F(32),
        &[&outer_dim, &mid_dim, &inner_dim],
    );
    builder.ld(ir::Type::F(32), &ptr, pattern);

    let space = builder.get();
    assert_eq!(
        space.domain().get_dim_kind(mid_dim[0]),
        DimKind::OUTER_VECTOR
    );
}

/// Ensures constraints are respected for DMA instructions.
#[test]
fn simple_dma() {
    const DATA_TYPE: ir::Type = ir::Type::F(32);

    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    let buffer = builder.allocate_shared(DATA_TYPE, 256);
    let start_dim = builder.open_dim_ex(ir::Size::new_const(256), DimKind::VECTOR);
    let (src_ptr, src_pattern) =
        builder.tensor_access(&0i32, None, DATA_TYPE, &[&start_dim]);
    let (dst_ptr, _) =
        builder.tensor_access(&buffer, Some(buffer), DATA_TYPE, &[&start_dim]);
    let start = builder.dma_start(&src_ptr, src_pattern, &dst_ptr, false);
    let wait_dim = builder.open_mapped_dim(&start_dim);
    let (_, dst_pattern) =
        builder.tensor_access(&buffer, Some(buffer), DATA_TYPE, &[&wait_dim]);
    builder.dma_wait(start, dst_pattern);

    let sync_var = builder.get_inst_variable(start);
    let space = builder.get();
    // Ensures start and wait have the same vectorization pattern.
    assert_eq!(space.domain().get_dim_kind(wait_dim[0]), DimKind::VECTOR);
    // Ensure the synchronisation flag has the correct type.
    assert_eq!(
        space.ir_instance().variable(sync_var).t(),
        ir::Type::SyncFlag
    );
    // Ensure the synchronisation flag is handled as a vector register.
    assert_eq!(
        space.domain().get_memory_space(sync_var),
        MemorySpace::VECTOR_REGISTER
    );
    // Try to generate a fully specified candidate.
    gen_best(&context, space);
}

/// Ensures DMA start and stop instructions have the same nesting.
#[test]
fn dma_start_wait_nesting() {
    const DATA_TYPE: ir::Type = ir::Type::F(32);

    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    let buffer = builder.allocate_shared(DATA_TYPE, 256);
    let start_dim = builder.open_dim_ex(ir::Size::new_const(256), DimKind::VECTOR);
    let (src_ptr, src_pattern) =
        builder.tensor_access(&0i32, None, DATA_TYPE, &[&start_dim]);
    let (dst_ptr, _) =
        builder.tensor_access(&buffer, Some(buffer), DATA_TYPE, &[&start_dim]);
    let start = builder.dma_start(&src_ptr, src_pattern, &dst_ptr, false);
    let wait_dim = builder.open_mapped_dim(&start_dim);
    let (_, dst_pattern) =
        builder.tensor_access(&buffer, Some(buffer), DATA_TYPE, &[&wait_dim]);
    let wait = builder.dma_wait(start, dst_pattern);

    let extra_dim = builder.open_dim(ir::Size::new_const(42));
    builder.order(&start, &extra_dim, Order::INNER);
    let space = builder.get();
    assert_eq!(
        space.domain().get_order(wait.into(), extra_dim[0].into()),
        Order::INNER
    );
}

/// Ensure we can generate DMA with strided accesses and multiple requests in-flight.
#[test]
fn complex_dma() {
    const DATA_TYPE: ir::Type = ir::Type::F(32);

    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    let buffer = builder.allocate_shared(DATA_TYPE, 256);
    let start_dim_0 = builder.open_dim_ex(ir::Size::new_const(4), DimKind::OUTER_VECTOR);
    let start_dim_1 = builder.open_dim_ex(ir::Size::new_const(4), DimKind::UNROLL);
    let start_dim_2 = builder.open_dim_ex(ir::Size::new_const(4), DimKind::INNER_VECTOR);
    let (src_ptr, src_pattern) = builder.tensor_access(
        &0i32,
        None,
        DATA_TYPE,
        &[&start_dim_0, &start_dim_1, &start_dim_2],
    );
    let (dst_ptr, _) = builder.tensor_access(
        &buffer,
        Some(buffer),
        DATA_TYPE,
        &[&start_dim_0, &start_dim_1, &start_dim_2],
    );
    let start = builder.dma_start(&src_ptr, src_pattern, &dst_ptr, false);

    let wait_dim_0 = builder.open_mapped_dim(&start_dim_0);
    let wait_dim_1 = builder.open_mapped_dim(&start_dim_1);
    let wait_dim_2 = builder.open_mapped_dim(&start_dim_2);
    let (_, dst_pattern) = builder.tensor_access(
        &buffer,
        Some(buffer),
        DATA_TYPE,
        &[&wait_dim_0, &wait_dim_1, &wait_dim_2],
    );
    builder.dma_wait(start, dst_pattern);
    builder.order(&start_dim_1, &wait_dim_1, Order::BEFORE);
    let space = builder.get();
    // Try to generate a fully specified candidate.
    gen_best(&context, space);
}

/// Ensures `is_instantited` is correctly set when needed.
#[test]
fn is_instantiated() {
    let _ = env_logger::try_init();
    let mut context = fake::Context::default();
    let n;
    let signature = {
        let mut builder = helper::SignatureBuilder::new("unroll_dims", &mut context);
        n = builder.max_size("n", 4);
        builder.get()
    };
    let mut builder = helper::Builder::new(&signature, context.device());
    let n_size = n.into_ir_size(&builder);
    let dyn_dim = builder.open_dim(n_size);
    let thread_dim = builder.open_dim_ex(ir::Size::new_const(4), DimKind::THREAD);
    let d1_0 = builder.open_dim(ir::Size::new_const(4));
    let d2_0 = builder.open_dim(ir::Size::new_const(4));
    let i0 = builder.mov(&0f32);
    let v0 = builder.get_inst_variable(i0);
    let d1_1 = builder.open_mapped_dim(&d1_0);
    let d2_1 = builder.open_mapped_dim(&d2_0);
    let v1_def_mode = ir::VarDefMode::InPlace { allow_sync: true };
    let v1_use_mode = ir::VarUseMode::FromRegisters;
    let v1 = builder.create_dim_map_variable(
        v0,
        &[(&d1_0, &d1_1), (&d2_0, &d2_1)],
        v1_def_mode,
        v1_use_mode,
    );
    builder.mov(&v1);
    builder.order(&d1_0, &d1_1, Order::BEFORE);
    // Retrieve the layout dimensions corresponding to each dimension.
    let layout_dyn_dim = builder.function().dim(dyn_dim[0]).layout_dims()[0];
    let layout_thread = builder.function().dim(thread_dim[0]).layout_dims()[0];
    let layout_d1_1 = builder.function().dim(d1_1[0]).layout_dims()[0];
    let layout_d2_0 = builder.function().dim(d2_0[0]).layout_dims()[0];
    let layout_d2_1 = builder.function().dim(d2_1[0]).layout_dims()[0];

    builder.action(Action::IsInstantiated(layout_d2_0, IsInstantiated::FALSE));
    let space = builder.get();
    // Dimensions with a dynamic size are not instantiated.
    assert_eq!(
        space.domain().get_is_instantiated(layout_dyn_dim),
        IsInstantiated::FALSE
    );
    // Thread dimensions are instantiated.
    assert_eq!(
        space.domain().get_is_instantiated(layout_thread),
        IsInstantiated::TRUE
    );
    // Non-merged dimensions are instantiated.
    assert_eq!(
        space.domain().get_is_instantiated(layout_d1_1),
        IsInstantiated::TRUE
    );
    // Non-instantiated dimensions are merged.
    assert_eq!(
        space.domain().get_order(d2_0[0].into(), d2_1[0].into()),
        Order::MERGED
    );
    // Instantiation is transmitted to successors.
    assert_eq!(
        space.domain().get_is_instantiated(layout_d2_1),
        IsInstantiated::FALSE
    );
}

/// Ensures the is_instantiated flag is not transmitted if the value is coppied.
#[test]
fn is_instantiated_on_copy() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    // Produce `v0` in a loop nest.
    let d0 = builder.open_dim(ir::Size::new_const(8));
    let i0 = builder.mov(&0f32);
    let v0 = builder.get_inst_variable(i0);
    let v0_layout = builder.function().variable(v0).layout()[0];
    // Store `v0` in an in-memory variable `v1`.
    let d1 = builder.open_mapped_dim(&d0);
    let fake_copy = ir::VarDefMode::Copy(None); // Don't materialize the copy.
    let no_uses = ir::VarUseMode::NoUses; // Don't place a copy in registers.
    let v1 = builder.create_dim_map_variable(v0, &[(&d0, &d1)], fake_copy, no_uses);
    let v1_layout = builder.function().variable(v1).layout()[0];
    builder.action(Action::MemorySpace(v1, MemorySpace::SHARED));
    builder.action(Action::IsInstantiated(v1_layout, IsInstantiated::TRUE));
    // Alias `v1` in `v2`.
    let fake_copy = ir::VarDefMode::Copy(None); // Don't materialize the copy.
    let no_uses = ir::VarUseMode::NoUses; // Don't place a copy in registers.
    let v2 = builder.create_dim_map_variable(v1, &[], fake_copy, no_uses);
    let v2_layout = builder.function().variable(v2).layout()[0];
    // Create a dummy instruction in d1 so the code is valid.
    builder.mov(&0f32);

    let space = builder.get();
    // v1 should be a copy of v0 and v2 alias with v1.
    assert_eq!(space.domain().get_var_def_mode(v1), VarDefMode::COPY);
    assert_eq!(space.domain().get_var_def_mode(v2), VarDefMode::IN_PLACE);
    assert_eq!(space.domain().get_memory_space(v2), MemorySpace::SHARED);
    assert_eq!(
        space.domain().get_is_instantiated(v0_layout),
        IsInstantiated::ALL
    );
    assert_eq!(
        space.domain().get_is_instantiated(v2_layout),
        IsInstantiated::TRUE
    );
}

/// Ensures the is_instantited flag is transmitted to and from memory accesses.
#[test]
fn is_instantiated_to_mem_accesses() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    // Produce `v0` in a loop nest.
    let d0 = builder.open_dim(ir::Size::new_const(8));
    let i0 = builder.mov(&0f32);
    let v0 = builder.get_inst_variable(i0);
    // Create an in-memory copy of `v0` in a separate loop nest.
    let d1 = builder.open_mapped_dim(&d0);
    let fake_copy = ir::VarDefMode::Copy(None); // Don't materialize the copy.
    let no_uses = ir::VarUseMode::NoUses; // Don't place a copy in registers.
    let v1 = builder.create_dim_map_variable(v0, &[(&d0, &d1)], fake_copy, no_uses);
    let v1_layout = builder.function().variable(v1).layout()[0];
    // Create a dummy instruction in d1 so the code is valid.
    builder.mov(&0f32);
    builder.close_dim(&d1);
    // Manually create an instruction to copy `v0` in `v1`. Use a fake address.
    builder.reopen_dim(&d0);
    let pattern = ir::AccessPattern::Variable {
        id: v1,
        dims: std::iter::once((d0[0], d1[0])).collect(),
    };
    let st = builder.st(&ir::Operand::Addr(ir::ArrayId::Variable(v1)), &v0, pattern);
    let st_layout = builder.function().inst(st).mem_access_layout()[0];

    // Try to set the layout dimension of the variable as instantiated.
    let mut builder_clone = builder.clone();
    builder_clone.action(Action::IsInstantiated(v1_layout, IsInstantiated::TRUE));
    let space = builder_clone.get();
    assert_eq!(
        space.domain().get_is_instantiated(st_layout),
        IsInstantiated::TRUE
    );
    // Try to set the layout dimension of the variable as not instantiated.
    let mut builder_clone = builder.clone();
    builder_clone.action(Action::IsInstantiated(v1_layout, IsInstantiated::FALSE));
    let space = builder_clone.get();
    assert_eq!(
        space.domain().get_is_instantiated(st_layout),
        IsInstantiated::FALSE
    );
    // Try to set the layout dimension of the memory_access as instantiated.
    builder.action(Action::IsInstantiated(st_layout, IsInstantiated::TRUE));
    let space = builder.get();
    assert_eq!(
        space.domain().get_is_instantiated(v1_layout),
        IsInstantiated::TRUE
    );
}

/// Ensures the number of registers is limited
#[test]
fn limit_registers() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    builder.open_dim_ex(ir::Size::new_const(1024), DimKind::THREAD);
    // Create a variable in a loop nest.
    let d1 = builder.open_dim(ir::Size::new_const(64));
    let i0 = builder.mov(&0f32);
    // Consume it in another loop nest.
    let d2 = builder.open_mapped_dim(&d1);
    let v1 = builder.map_instruction(i0, &[(&d1, &d2)], false, vec![]);
    let v1_layout = builder.function().dim(d2[0]).layout_dims()[0];
    builder.mov(&v1);
    builder.action(Action::MemorySpace(v1, !MemorySpace::VECTOR_REGISTER));

    let space = builder.get();
    // Because the number of registers is limited to 32 * 1024, d1 and d2 cannot
    // communicate through registers. Instead, they must be merged.
    assert_eq!(
        space.domain().get_order(d1[0].into(), d2[0].into()),
        Order::MERGED
    );
    assert_eq!(
        space.domain().get_is_instantiated(v1_layout),
        IsInstantiated::FALSE
    );
}

/// Ensures the number of synchronisation registers is limited, but not too much.
#[test]
fn limit_synchronisation_registers() {
    const DATA_TYPE: ir::Type = ir::Type::F(32);

    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    builder.open_dim_ex(ir::Size::new_const(1024), DimKind::THREAD);
    // Create a DMA start in a loop nest. Use fake addresses.
    let d1 = builder.open_dim_ex(ir::Size::new_const(8), !DimKind::VECTOR);
    let (src_ptr, src_pattern) = builder.tensor_access(&0i32, None, DATA_TYPE, &[&d1]);
    let (dst_ptr, _) = builder.tensor_access(&0i32, None, DATA_TYPE, &[&d1]);
    let start = builder.dma_start(&src_ptr, src_pattern, &dst_ptr, false);
    // Create a DMA wait in a separate loop nest.
    let d2 = builder.open_mapped_dim(&d1);
    let (_, dst_pattern) = builder.tensor_access(&0i32, None, DATA_TYPE, &[&d2]);
    builder.dma_wait(start, dst_pattern);

    let space = builder.get();
    // Because the number of registers is limited to 4 * 1024, d1 and d2 cannot
    // communicate through registers. Instead, they must be merged.
    assert_eq!(
        space.domain().get_order(d1[0].into(), d2[0].into()),
        Order::MERGED
    );
}

/// Ensures ranks are unique and limited by the number of instantiated decisions.
#[test]
fn ranks_unique_and_bounded() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());

    // Produce `v0` in a loop nest.
    let d0_0 = builder.open_dim(ir::Size::new_const(8));
    let d1_0 = builder.open_dim(ir::Size::new_const(8));
    let d2_0 = builder.open_dim(ir::Size::new_const(8));
    let i0 = builder.mov(&0f32);
    let v0 = builder.get_inst_variable(i0);
    // Store `v0` in an in-memory variable `v1`.
    let d0_1 = builder.open_mapped_dim(&d0_0);
    let d1_1 = builder.open_mapped_dim(&d1_0);
    let d2_1 = builder.open_mapped_dim(&d1_0);
    let fake_copy = ir::VarDefMode::Copy(None); // Don't materialize the copy.
    let no_uses = ir::VarUseMode::NoUses; // Don't place a copy in registers.
    let mapping = [(&d0_0, &d0_1), (&d1_0, &d1_1), (&d2_0, &d2_1)];
    let v1 = builder.create_dim_map_variable(v0, &mapping, fake_copy, no_uses);
    let d0_layout = builder.function().dim(d0_1[0]).layout_dims()[0];
    let d1_layout = builder.function().dim(d1_1[0]).layout_dims()[0];
    let d2_layout = builder.function().dim(d2_1[0]).layout_dims()[0];
    builder.action(Action::MemorySpace(v1, MemorySpace::SHARED));
    builder.action(Action::IsInstantiated(d2_layout, IsInstantiated::FALSE));

    let mut space = builder.get();
    // We can't apply the decision before generating the search space because we need
    // layout dimensions to be registered as memory layout dimensions.
    let d0_rank = Action::Rank(
        d0_layout,
        NumericSet {
            enabled_values: 0b010,
        },
    );
    assert!(space.apply_decisions(vec![d0_rank]).is_ok());
    // Only two dimensions may be instantiated so the rank is less that 3 (represented by
    // 0b1000). It cannot be 1 (represented by 0b010) as it is already taken but can be 0
    // (represented by 0b0001).
    assert_eq!(
        space.domain().get_rank(d1_layout),
        NumericSet {
            enabled_values: 0b101
        }
    );
}

/// Ranks are correctly transmitted to aliasing variables.
#[test]
fn ranks_aliasing_variables() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    // Produce `v0` in a loop nest.
    let d0_0 = builder.open_dim(ir::Size::new_const(8));
    let d1_0 = builder.open_dim(ir::Size::new_const(8));
    let i0 = builder.mov(&0f32);
    let v0 = builder.get_inst_variable(i0);
    // Store `v0` in an in-memory variable `v1.
    let d0_1 = builder.open_mapped_dim(&d0_0);
    let d1_1 = builder.open_mapped_dim(&d1_0);
    let fake_copy = ir::VarDefMode::Copy(None); // Don't materialize the copy.
    let no_uses = ir::VarUseMode::NoUses; // Don't place a copy in registers.
    let mapping = [(&d0_0, &d0_1), (&d1_0, &d1_1)];
    let v1 = builder.create_dim_map_variable(v0, &mapping, fake_copy, no_uses);
    let v1_d0_layout = builder.function().dim(d0_1[0]).layout_dims()[0];
    let v1_d1_layout = builder.function().dim(d1_1[0]).layout_dims()[0];
    builder.action(Action::MemorySpace(v1, MemorySpace::SHARED));
    // Alias `v1` in `v2`.
    let fake_copy = ir::VarDefMode::Copy(None); // Don't materialize the copy.
    let no_uses = ir::VarUseMode::NoUses; // Don't place a copy in registers.
    let _v2 = builder.create_dim_map_variable(v1, &[], fake_copy, no_uses);
    let v2_d0_layout = builder.function().dim(d0_1[0]).layout_dims()[1];
    let v2_d1_layout = builder.function().dim(d1_1[0]).layout_dims()[1];
    // Create a dummy instruction in d1 so the code is valid.
    builder.mov(&0f32);

    let mut space = builder.get();
    // We can't apply the decision before generating the search space because we need
    // layout dimensions to be registered as memory layout dimensions.
    const RANK_1: NumericSet = NumericSet {
        enabled_values: 0b010,
    };
    const RANK_2: NumericSet = NumericSet {
        enabled_values: 0b100,
    };
    let v1_d0_rank = Action::Rank(v1_d0_layout, RANK_1);
    let v2_d1_rank = Action::Rank(v2_d1_layout, RANK_2);
    assert!(space.apply_decisions(vec![v1_d0_rank, v2_d1_rank]).is_ok());
    assert_eq!(space.domain().get_rank(v1_d1_layout), RANK_2);
    assert_eq!(space.domain().get_rank(v2_d0_layout), RANK_1);
}

/// Ranks are correctly transmitted to memory accesses.
#[test]
fn ranks_to_memory_access() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    // Produce `v0` in a loop nest.
    let d0_0 = builder.open_dim(ir::Size::new_const(8));
    let d1_0 = builder.open_dim(ir::Size::new_const(8));
    let i0 = builder.mov(&0f32);
    let v0 = builder.get_inst_variable(i0);
    // Store `v0` in an in-memory variable `v1.
    let d0_1 = builder.open_mapped_dim(&d0_0);
    let d1_1 = builder.open_mapped_dim(&d1_0);
    let fake_copy = ir::VarDefMode::Copy(None); // Don't materialize the copy.
    let no_uses = ir::VarUseMode::NoUses; // Don't place a copy in registers.
    let mapping = [(&d0_0, &d0_1), (&d1_0, &d1_1)];
    let v1 = builder.create_dim_map_variable(v0, &mapping, fake_copy, no_uses);
    let v1_d0_layout = builder.function().dim(d0_1[0]).layout_dims()[0];
    let v1_d1_layout = builder.function().dim(d1_1[0]).layout_dims()[0];
    builder.action(Action::MemorySpace(v1, MemorySpace::SHARED));
    // Create a dummy instruction in d1 so the code is valid.
    builder.mov(&0f32);
    builder.close_dim(&d0_1);
    builder.close_dim(&d1_1);
    // Manually create an instruction to copy `v0` in `v1`. Use a fake address.
    builder.reopen_dim(&d0_0);
    builder.reopen_dim(&d1_0);
    let pattern = ir::AccessPattern::Variable {
        id: v1,
        dims: vec![(d0_0[0], d0_1[0]), (d1_0[0], d1_1[0])]
            .into_iter()
            .collect(),
    };
    let _st = builder.st(&ir::Operand::Addr(ir::ArrayId::Variable(v1)), &v0, pattern);
    let st_d0_layout = builder.function().dim(d0_0[0]).layout_dims()[1];
    let st_d1_layout = builder.function().dim(d1_0[0]).layout_dims()[1];

    let mut space = builder.get();
    // We can't apply the decision before generating the search space because we need
    // layout dimensions to be registered as memory layout dimensions.
    const RANK_1: NumericSet = NumericSet {
        enabled_values: 0b010,
    };
    const RANK_2: NumericSet = NumericSet {
        enabled_values: 0b100,
    };
    let v1_d0_rank = Action::Rank(v1_d0_layout, RANK_1);
    let st_d1_rank = Action::Rank(st_d1_layout, RANK_2);
    assert!(space.apply_decisions(vec![v1_d0_rank, st_d1_rank]).is_ok());
    assert_eq!(space.domain().get_rank(v1_d1_layout), RANK_2);
    assert_eq!(space.domain().get_rank(st_d0_layout), RANK_1);
}

/// Ensures we correctly compute the amount of shared memory used.
#[test]
fn total_memory_use() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    // Produce `v0` in a loop nest.
    let d0_0 = builder.open_dim(ir::Size::new_const(8));
    let d1_0 = builder.open_dim(ir::Size::new_const(16));
    let i0 = builder.mov(&0f32);
    let v0 = builder.get_inst_variable(i0);
    // Store `v0` in an in-memory variable `v1`.
    let d0_1 = builder.open_mapped_dim(&d0_0);
    let d1_1 = builder.open_mapped_dim(&d1_0);
    let fake_copy = ir::VarDefMode::Copy(None); // Don't materialize the copy.
    let no_uses = ir::VarUseMode::NoUses; // Don't place a copy in registers.
    let mapping = [(&d0_0, &d0_1), (&d1_0, &d1_1)];
    let v1 = builder.create_dim_map_variable(v0, &mapping, fake_copy, no_uses);
    let v1_d0_layout = builder.function().dim(d0_1[0]).layout_dims()[0];
    let v1_d1_layout = builder.function().dim(d1_1[0]).layout_dims()[0];
    builder.action(Action::MemorySpace(v1, MemorySpace::SHARED));
    // Instantiate `v1_d0_layout` but not `v2_d0_layout`.
    builder.action(Action::IsInstantiated(v1_d0_layout, IsInstantiated::TRUE));
    builder.action(Action::IsInstantiated(v1_d1_layout, IsInstantiated::FALSE));
    // Create a dummy instruction in d1 so the code is valid.
    builder.mov(&0f32);

    let space = builder.get();
    assert_eq!(space.domain().get_shared_mem_used(), HalfRange { min: 8 });
}

/// Ensures the correct strides are generated for dynamic layouts.
#[test]
fn codegen_strides() {
    let _ = env_logger::try_init();
    let context = fake::Context::default();
    let signature = empty_signature();
    let mut builder = helper::Builder::new(&signature, context.device());
    // Produce `v0` in a loop nest.
    let d0_0 = builder.open_dim_ex(ir::Size::new_const(8), DimKind::LOOP);
    let d1_0 = builder.open_dim_ex(ir::Size::new_const(16), DimKind::LOOP);
    let d2_0 = builder.open_dim_ex(ir::Size::new_const(32), DimKind::LOOP);
    builder.order(&d0_0, &d1_0, Order::OUTER);
    builder.order(&d1_0, &d2_0, Order::OUTER);
    let i0 = builder.mov(&0f32);
    let v0 = builder.get_inst_variable(i0);
    // Store `v0` in an in-memory variable `v1`.
    let d0_1 = builder.open_mapped_dim(&d0_0);
    let d1_1 = builder.open_mapped_dim(&d1_0);
    let d2_1 = builder.open_mapped_dim(&d2_0);
    builder.action(Action::DimKind(d0_1[0], DimKind::LOOP));
    builder.action(Action::DimKind(d1_1[0], DimKind::LOOP));
    builder.action(Action::DimKind(d2_1[0], DimKind::LOOP));
    builder.order(&d0_1, &d1_1, Order::OUTER);
    builder.order(&d1_1, &d2_1, Order::OUTER);
    builder.order(&d1_0, &d1_1, Order::BEFORE);
    builder.order(&d2_0, &d2_1, Order::BEFORE);
    let fake_copy = ir::VarDefMode::Copy(None); // Don't materialize the copy.
    let no_uses = ir::VarUseMode::NoUses; // Don't place a copy in registers.
    let mapping = [(&d0_0, &d0_1), (&d1_0, &d1_1), (&d2_0, &d2_1)];
    let v1 = builder.create_dim_map_variable(v0, &mapping, fake_copy, no_uses);
    let d0_layout = builder.function().dim(d0_1[0]).layout_dims()[0];
    let d1_layout = builder.function().dim(d1_1[0]).layout_dims()[0];
    let d2_layout = builder.function().dim(d2_1[0]).layout_dims()[0];
    builder.action(Action::MemorySpace(v1, MemorySpace::SHARED));
    // Constrain ranks. d0 -> not instantiated, d1 -> 1, d2 -> 2
    builder.action(Action::IsInstantiated(d0_layout, IsInstantiated::FALSE));
    let mut space = builder.get();
    const RANK_1: NumericSet = NumericSet {
        enabled_values: 0b010,
    };
    const RANK_2: NumericSet = NumericSet {
        enabled_values: 0b100,
    };
    let d1_rank = Action::Rank(d1_layout, RANK_1);
    let d2_rank = Action::Rank(d2_layout, RANK_2);
    assert!(space.apply_decisions(vec![d1_rank, d2_rank]).is_ok());
    // Check strides.
    let code = codegen::Function::build(&space);
    assert_eq!(
        code.layout_dim_stride(d0_layout),
        &codegen::Size::new(0, vec![], 1)
    );
    assert_eq!(
        code.layout_dim_stride(d1_layout),
        &codegen::Size::new(1, vec![], 1)
    );
    assert_eq!(
        code.layout_dim_stride(d2_layout),
        &codegen::Size::new(16, vec![], 1)
    );
}
