#![cfg(all(feature="cuda", test))]
use codegen;
use device::{Context, cuda, EvalMode};
use env_logger;
use helper::*;
use model;
use search_space::*;
use super::*;

#[test]
fn partial_bound_0() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor);
    let z;
    let signature = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        z = builder.array::<f32>("z", 16).0;
        builder.get()
    };

    let mut builder = Builder::new(&signature, context.device());
    let size = builder.cst_size(4);

    let dim_x = builder.open_dim_ex(size.clone(), DimKind::THREAD);
    let dim_y = builder.open_dim_ex(size.clone(), DimKind::THREAD);
    builder.mov(&0f32);
    builder.close_dim(&dim_y);
    builder.close_dim(&dim_x);

    let dim_z = builder.open_dim_ex(size, DimKind::THREAD);
    let (addr, pattern) = builder.tensor_access(&"z", z, &ir::Type::F(32), &[&dim_z]);
    let st_z = builder.st(&addr, &0f32, pattern);

    builder.order(&dim_x, &dim_z, Order::BEFORE);
    builder.order(&dim_x, &dim_y, Order::OUTER);

    let partial_pressure = {
        let space = builder.get_clone();
        let local_info = LocalInfo::compute(&space, &context);
        trace!("partial nesting: {:?}", local_info.nesting[&st_z.into()]);
        sum_pressure(&context, &space, &local_info,
                     BottleneckLevel::Global, &[], &ir::Size::one())
    }.get_bottleneck(3);

    builder.action(Action::ThreadMapping(dim_z, dim_x, ThreadMapping::MAPPED_OUT));
    let final_pressure = {
        let space = builder.get_clone();
        let local_info = LocalInfo::compute(&space, &context);
        trace!("final nesting: {:?}", local_info.nesting[&st_z.into()]);
        sum_pressure(&context, &space, &local_info,
                     BottleneckLevel::Global, &[], &ir::Size::one())
    }.get_bottleneck(3);

    assert!(final_pressure*1.001 >= partial_pressure, "{} < {}",
            final_pressure, partial_pressure);
}

#[test]
fn partial_bound_1() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor); 
    let z;
    let signature = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        z = builder.array::<f32>("z", 256).0;
        builder.get()
    };

    let mut builder = Builder::new(&signature, context.device());
    let size = builder.cst_size(256);
    let dim_x = builder.open_dim_ex(size.clone(), DimKind::THREAD);
    builder.mov(&0f32);
    builder.close_dim(&dim_x);

    let dim_z = builder.open_dim(size);
    let (addr, pattern) = builder.tensor_access(&"z", z, &ir::Type::F(32), &[&dim_z]);
    let st_z = builder.st(&addr, &0f32, pattern);

    let partial_pressure = {
        let space = builder.get_clone();
        let local_info = LocalInfo::compute(&space, &context);
        trace!("partial nesting: {:?}", local_info.nesting[&st_z.into()]);
        sum_pressure(&context, &space, &local_info,
                     BottleneckLevel::Global, &[], &ir::Size::one())
    }.get_bottleneck(5);

    builder.action(Action::DimKind(dim_z, DimKind::THREAD));
    let final_pressure = {
        let space = builder.get();
        let local_info = LocalInfo::compute(&space, &context);
        trace!("final nesting: {:?}", local_info.nesting[&st_z.into()]);
        sum_pressure(&context, &space, &local_info,
                     BottleneckLevel::Global, &[], &ir::Size::one())
    }.get_bottleneck(5);

    assert!(final_pressure*1.001 >= partial_pressure, "{} < {}",
            final_pressure, partial_pressure);
}


#[test]
fn partial_bound_2() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor); 

    let (x, y, a);
    let signature = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        builder.scalar("m", 1i32 << 13);
        builder.scalar("n", 1i32 << 13);
        let m_size: tensor::DimSize = "m".into();
        let n_size: tensor::DimSize = "n".into();
        x = builder.tensor::<f32>("x", vec![n_size.clone()], true);
        a = builder.tensor::<f32>("a", vec![m_size.clone(), n_size], true);
        y = builder.tensor::<f32>("y", vec![m_size], false);
        builder.get()
    };

    let m_tiling = &[2];

    let mut builder = Builder::new(&signature, context.device());
    let ld_x = x.load(&[&[]], &mut builder);
    let ld_a = a.load(&[m_tiling, &[]], &mut builder);

    let init_dim_m = builder.open_mapped_dim(&ld_a[0]);
    let init = builder.mov(&0f32);
    let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
    let acc_dim_n = builder.open_mapped_dim(&ld_x[0]);
    let a_op = ld_a.dim_map(&[&acc_dim_m, &acc_dim_n], ir::DimMapScope::Global,
                            &mut builder);
    let x_op = ld_x.dim_map(&[&acc_dim_n], ir::DimMapScope::Global, &mut builder);
    let acc = builder.mad(&a_op, &x_op, &Reduce(init));
    builder.close_dim(&acc_dim_n);

    let sum = tensor::VirtualTensor::new(acc, vec![acc_dim_m.clone()]);
    let st_y = sum.store(&y, &mut builder);

    builder.action(Action::DimKind(ld_a[0][1], DimKind::UNROLL));
    builder.action(Action::DimKind(init_dim_m[1], DimKind::UNROLL));
    builder.action(Action::DimKind(st_y[0][1], DimKind::UNROLL));

    builder.order(&acc_dim_n, &st_y.inst(), Order::BEFORE);
    builder.order(&ld_a[0][1], &ld_x.inst(), Order::BEFORE);
    builder.order(&acc_dim_m[1], &ld_x.inst(), Order::AFTER);

    builder.action(Action::InstFlag(ld_x.inst(), InstFlag::MEM_CG));
    builder.action(Action::InstFlag(ld_a.inst(), InstFlag::MEM_CG));
    builder.action(Action::InstFlag(st_y.inst(), InstFlag::MEM_CS));

    let partial_bound = model::bound(&builder.get_clone(), &context);
    builder.action(Action::DimKind(ld_a[0][0], DimKind::BLOCK));

    let final_bound = model::bound(&builder.get(), &context);

    assert!(final_bound.value()*1.001 >= partial_bound.value(), "{} < {}",
    final_bound, partial_bound);

}

#[test]
fn partial_bound_3() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor); 

    let a;
    let signature = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        a = builder.array::<f32>("a", 256).0;
        builder.get()
    };

    let mut builder = Builder::new(&signature, context.device());

    let size_m = builder.cst_size(256);
    let ld_a_dim = builder.open_tiled_dim(size_m, &[4]);
    let (addr, patt) = builder.tensor_access(&"a", a, &ir::Type::F(32), &[&ld_a_dim]);
    builder.ld(ir::Type::F(32), &addr, patt);
    builder.close_dim(&ld_a_dim);

    let size_n = builder.cst_size(4);
    let init_dim_m = builder.open_mapped_dim(&ld_a_dim);
    let init_dim_n = builder.open_dim(size_n);
    builder.mov(&0f32);

    builder.action(Action::DimKind(ld_a_dim[0], DimKind::THREAD));
    builder.action(Action::DimKind(ld_a_dim[1], DimKind::UNROLL));

    builder.action(Action::DimKind(init_dim_m[0], DimKind::THREAD));
    //builder.action(Action::DimKind(init_dim_m[1], DimKind::THREAD));
    builder.action(Action::ThreadMapping(ld_a_dim[0], init_dim_m[0],
                                         ThreadMapping::MAPPED));
    builder.action(Action::ThreadMapping(
            init_dim_m[0], init_dim_m[1], ThreadMapping::MAPPED_IN));
    builder.action(Action::DimKind(init_dim_n, DimKind::THREAD));

    let partial_pressure = {
        let space = builder.get_clone();
        let local_info = LocalInfo::compute(&space, &context);
        sum_pressure(&context, &space, &local_info,
                     BottleneckLevel::Global, &[], &ir::Size::one())
    }.get_bottleneck(4);

    builder.action(Action::ThreadMapping(init_dim_n, ld_a_dim[0],
                                         ThreadMapping::MAPPED_IN));

    let final_pressure = {
        let space = builder.get();
        let local_info = LocalInfo::compute(&space, &context);
        sum_pressure(&context, &space, &local_info,
                     BottleneckLevel::Global, &[], &ir::Size::one())
    }.get_bottleneck(4);

    assert!(final_pressure*1.001 >= partial_pressure, "{} < {}",
            final_pressure, partial_pressure);
}

#[test]
fn partial_bound_4() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor); 

    let a: tensor::Tensor<f32>;
    let signature = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        a = builder.tensor::<f32>("a", vec![25.into(), 26.into(), 32.into()], true);
        builder.get()
    };

    let mut builder = Builder::new(&signature, context.device());
    let ld_a = a.load(&[&[], &[], &[]], &mut builder);

    builder.action(Action::DimKind(ld_a[0][0], DimKind::THREAD));
    builder.action(Action::DimKind(ld_a[1][0], DimKind::THREAD));
    builder.action(Action::DimKind(ld_a[2][0], DimKind::LOOP));

    builder.action(Action::InstFlag(ld_a.inst(), InstFlag::MEM_CS));

    let partial_pressure = {
        let space = builder.get_clone();
        let local_info = LocalInfo::compute(&space, &context);
        sum_pressure(&context, &space, &local_info,
                     BottleneckLevel::Global, &[], &ir::Size::one())
    }.get_bottleneck(3);

    builder.action(Action::ThreadMapping(ld_a[0][0], ld_a[1][0],
                                         ThreadMapping::MAPPED_IN));

    let final_pressure = {
        let space = builder.get();
        let local_info = LocalInfo::compute(&space, &context);
        sum_pressure(&context, &space, &local_info,
                     BottleneckLevel::Global, &[], &ir::Size::one())
    }.get_bottleneck(3);

    assert!(final_pressure*1.001 >= partial_pressure, "{} < {}",
            final_pressure, partial_pressure);
}

#[test]
fn partial_bound_5() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor); 

    let (signature, a) = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        let a = tensor::TensorBuilder::new("a", vec![13.into(), 32.into()])
            .stride_dim(1).finish::<f32, _>(&mut builder);
        (builder.get(), a)
    };

    let mut builder = Builder::new(&signature, context.device());

    let ld_a = a.load(&[&[]], &mut builder);
    let dim1 = builder.open_dim_ex(ir::Size::new_const(26), DimKind::THREAD);
    let _ = builder.mov(&0f32);

    builder.order(&ld_a.inst(), &dim1, Order::AFTER);

    let partial_pressure = {
        let space = builder.get_clone();
        let local_info = LocalInfo::compute(&space, &context);
        sum_pressure(&context, &space, &local_info,
                     BottleneckLevel::Global, &[], &ir::Size::one())
    }.get_bottleneck(4);

    builder.action(Action::DimKind(ld_a[0][0], DimKind::UNROLL));

    let final_pressure = {
        let space = builder.get();
        let local_info = LocalInfo::compute(&space, &context);
        sum_pressure(&context, &space, &local_info,
                     BottleneckLevel::Global, &[], &ir::Size::one())
    }.get_bottleneck(4);

    assert!(final_pressure*1.001 >= partial_pressure, "{} < {}",
            final_pressure, partial_pressure);
}

#[test]
fn final_bound_0() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor);

    let (x, y, z);
    let signature = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        builder.scalar("n", 1 << 25);
        let n_size: tensor::DimSize = "n".into();
        x = builder.tensor::<f32>("x", vec![n_size.clone()], true);
        y = builder.tensor::<f32>("y", vec![n_size.clone()], true);
        z = builder.tensor::<f32>("z", vec![n_size], false);
        builder.get()
    };

    let tiling = &[1024, 4];
    let mut builder = Builder::new(&signature, context.device());

    let ld_x = x.load(&[tiling], &mut builder);
    let ld_y = y.load(&[tiling], &mut builder);
    let mad_dim = builder.open_mapped_dim(&ld_x[0]);
    let x_op = ld_x.dim_map(&[&mad_dim], ir::DimMapScope::Global, &mut builder);
    let y_op = ld_y.dim_map(&[&mad_dim], ir::DimMapScope::Global, &mut builder);
    let mad = tensor::VirtualTensor::new(
        builder.mad(&x_op, &4.33f32, &y_op), vec![mad_dim.clone()]);
    let st_z = mad.store(&z, &mut builder);

    builder.action(Action::DimKind(ld_x[0][2], DimKind::VECTOR));
    builder.action(Action::DimKind(ld_x[0][1], DimKind::THREAD));
    builder.action(Action::DimKind(ld_x[0][0], DimKind::BLOCK));
    builder.action(Action::DimKind(ld_y[0][2], DimKind::VECTOR));
    builder.action(Action::DimKind(ld_y[0][1], DimKind::THREAD));
    builder.action(Action::DimKind(mad_dim[1], DimKind::THREAD));
    builder.action(Action::DimKind(mad_dim[2], DimKind::UNROLL));
    builder.action(Action::DimKind(st_z[0][1], DimKind::THREAD));
    builder.action(Action::DimKind(st_z[0][2], DimKind::VECTOR));
    builder.order(&ld_x[0][2], &ld_y.inst(), Order::BEFORE);
    builder.order(&ld_x[0][1], &ld_y.inst(), Order::BEFORE);
    builder.order(&ld_y[0][1], &mad.inst(), Order::BEFORE);
    builder.order(&mad_dim[1], &st_z.inst(), Order::OUTER);
    builder.action(Action::InstFlag(ld_x.inst(), InstFlag::MEM_CS));
    builder.action(Action::InstFlag(ld_y.inst(), InstFlag::MEM_CG));
    builder.action(Action::InstFlag(st_z.inst(), InstFlag::MEM_CG));
    let space = builder.get();
    let bound = model::bound(&space, &context);
    let kernel = codegen::Function::build(&space);
    let eval = unwrap!(context.evaluate(&kernel, EvalMode::TestBound));
    assert!(eval * 1.001 >= bound.value(), "{:.2e} < {}", eval, bound);
}
