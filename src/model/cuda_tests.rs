#![cfg(all(feature = "cuda", test))]
/* Disable tests to avoi circular dependencies, until we have a dedicated model crate.
use super::*;
use crate::codegen;
use crate::context::{Context, EvalMode};
use env_logger;
use crate::helper::*;
use crate::model;
use crate::search_space::*;
use telamon_cuda as cuda;

#[test]
fn partial_bound_0() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor);
    let signature = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        builder.array::<f32>("z", 16);
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
    let (addr, pattern) = builder.tensor_access(&"z", None, ir::Type::F(32), &[&dim_z]);
    let st_z = builder.st(&addr, &0f32, pattern);

    builder.order(&dim_x, &dim_z, Order::BEFORE);
    builder.order(&dim_x, &dim_y, Order::OUTER);

    let partial_pressure = {
        let space = builder.get_clone();
        let local_info = LocalInfo::compute(&space, context.params(), context.device());
        trace!("partial nesting: {:?}", local_info.nesting[&st_z.into()]);
        sum_pressure(
            context.params(), context.device(),
            &space,
            &local_info,
            BottleneckLevel::Global,
            &[],
            &ir::PartialSize::default(),
        )
    }
    .get_bottleneck(3);

    builder.action(Action::ThreadMapping(
        dim_z[0],
        dim_x[0],
        ThreadMapping::MAPPED_OUT,
    ));
    let final_pressure = {
        let space = builder.get_clone();
        let local_info = LocalInfo::compute(&space, context.params(), context.device());
        trace!("final nesting: {:?}", local_info.nesting[&st_z.into()]);
        sum_pressure(
            context.params(), context.device(),
            &space,
            &local_info,
            BottleneckLevel::Global,
            &[],
            &ir::PartialSize::default(),
        )
    }
    .get_bottleneck(3);

    assert!(
        final_pressure * 1.001 >= partial_pressure,
        "{} < {}",
        final_pressure,
        partial_pressure
    );
}

#[test]
fn partial_bound_1() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor);
    let signature = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        builder.array::<f32>("z", 256);
        builder.get()
    };

    let mut builder = Builder::new(&signature, context.device());
    let size = builder.cst_size(256);
    let dim_x = builder.open_dim_ex(size.clone(), DimKind::THREAD);
    builder.mov(&0f32);
    builder.close_dim(&dim_x);

    let dim_z = builder.open_dim(size);
    let (addr, pattern) = builder.tensor_access(&"z", None, ir::Type::F(32), &[&dim_z]);
    let st_z = builder.st(&addr, &0f32, pattern);

    let partial_pressure = {
        let space = builder.get_clone();
        let local_info = LocalInfo::compute(&space, context.params(), context.device());
        trace!("partial nesting: {:?}", local_info.nesting[&st_z.into()]);
        sum_pressure(
            context.params(), context.device(),
            &space,
            &local_info,
            BottleneckLevel::Global,
            &[],
            &ir::PartialSize::default(),
        )
    }
    .get_bottleneck(5);

    builder.action(Action::DimKind(dim_z[0], DimKind::THREAD));
    let final_pressure = {
        let space = builder.get();
        let local_info = LocalInfo::compute(&space, context.params(), context.device());
        trace!("final nesting: {:?}", local_info.nesting[&st_z.into()]);
        sum_pressure(
            context.params(), context.device(),
            &space,
            &local_info,
            BottleneckLevel::Global,
            &[],
            &ir::PartialSize::default(),
        )
    }
    .get_bottleneck(5);

    assert!(
        final_pressure * 1.001 >= partial_pressure,
        "{} < {}",
        final_pressure,
        partial_pressure
    );
}

#[test]
fn partial_bound_2() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor);

    let (x, y, a);
    let signature = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        let m_size = builder.max_size("m", 1 << 13);
        let n_size = builder.max_size("n", 1 << 13);
        x = builder.tensor::<f32>("x", vec![n_size.clone()], true);
        a = builder.tensor::<f32>("a", vec![m_size.clone(), n_size], true);
        y = builder.tensor::<f32>("y", vec![m_size], false);
        builder.get()
    };

    let m_tiling = TilingPattern::new_fixed(&[2]);

    let mut builder = Builder::new(&signature, context.device());
    let ld_x = x.load(vec![TilingPattern::default()], &mut builder);
    let ld_a = a.load(vec![m_tiling, TilingPattern::default()], &mut builder);

    let init_dim_m = builder.open_mapped_dim(&ld_a[0]);
    let init = builder.mov(&0f32);
    let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
    let acc_dim_n = builder.open_mapped_dim(&ld_x[0]);
    let a_op = ld_a.dim_map(
        &[&acc_dim_m, &acc_dim_n],
        ir::DimMapScope::Global(()),
        &mut builder,
    );
    let x_op = ld_x.dim_map(&[&acc_dim_n], ir::DimMapScope::Global(()), &mut builder);
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

    builder.action(Action::InstFlag(ld_x.inst(), InstFlag::CACHE_GLOBAL));
    builder.action(Action::InstFlag(ld_a.inst(), InstFlag::CACHE_GLOBAL));
    builder.action(Action::InstFlag(st_y.inst(), InstFlag::NO_CACHE));

    let partial_bound = model::bound(&builder.get_clone(), &context);
    builder.action(Action::DimKind(ld_a[0][0], DimKind::BLOCK));

    let final_bound = model::bound(&builder.get(), &context);

    assert!(
        final_bound.value() * 1.001 >= partial_bound.value(),
        "{} < {}",
        final_bound,
        partial_bound
    );
}

#[test]
fn partial_bound_3() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor);

    let signature = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        builder.array::<f32>("a", 256);
        builder.get()
    };

    let mut builder = Builder::new(&signature, context.device());

    let size_m = builder.cst_size(256);
    let ld_a_dim = builder.open_tiled_dim(size_m, TilingPattern::new_fixed(&[4]));
    let (addr, patt) = builder.tensor_access(&"a", None, ir::Type::F(32), &[&ld_a_dim]);
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
    builder.action(Action::ThreadMapping(
        ld_a_dim[0],
        init_dim_m[0],
        ThreadMapping::MAPPED,
    ));
    builder.action(Action::ThreadMapping(
        init_dim_m[0],
        init_dim_m[1],
        ThreadMapping::MAPPED_IN,
    ));
    builder.action(Action::DimKind(init_dim_n[0], DimKind::THREAD));

    let partial_pressure = {
        let space = builder.get_clone();
        let local_info = LocalInfo::compute(&space, context.params(), context.device());
        sum_pressure(
            context.params(), context.device(),
            &space,
            &local_info,
            BottleneckLevel::Global,
            &[],
            &ir::PartialSize::default(),
        )
    }
    .get_bottleneck(4);

    builder.action(Action::ThreadMapping(
        init_dim_n[0],
        ld_a_dim[0],
        ThreadMapping::MAPPED_IN,
    ));

    let final_pressure = {
        let space = builder.get();
        let local_info = LocalInfo::compute(&space, context.params(), context.device());
        sum_pressure(
            context.params(), context.device(),
            &space,
            &local_info,
            BottleneckLevel::Global,
            &[],
            &ir::PartialSize::default(),
        )
    }
    .get_bottleneck(4);

    assert!(
        final_pressure * 1.001 >= partial_pressure,
        "{} < {}",
        final_pressure,
        partial_pressure
    );
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
    let ld_a = a.load(vec![TilingPattern::default(); 3], &mut builder);

    builder.action(Action::DimKind(ld_a[0][0], DimKind::THREAD));
    builder.action(Action::DimKind(ld_a[1][0], DimKind::THREAD));
    builder.action(Action::DimKind(ld_a[2][0], DimKind::LOOP));

    builder.action(Action::InstFlag(ld_a.inst(), InstFlag::NO_CACHE));

    let partial_pressure = {
        let space = builder.get_clone();
        let local_info = LocalInfo::compute(&space, context.params(), context.device());
        sum_pressure(
            context.params(), context.device(),
            &space,
            &local_info,
            BottleneckLevel::Global,
            &[],
            &ir::PartialSize::default(),
        )
    }
    .get_bottleneck(3);

    builder.action(Action::ThreadMapping(
        ld_a[0][0],
        ld_a[1][0],
        ThreadMapping::MAPPED_IN,
    ));

    let final_pressure = {
        let space = builder.get();
        let local_info = LocalInfo::compute(&space, context.params(), context.device());
        sum_pressure(
            context.params(), context.device(),
            &space,
            &local_info,
            BottleneckLevel::Global,
            &[],
            &ir::PartialSize::default(),
        )
    }
    .get_bottleneck(3);

    assert!(
        final_pressure * 1.001 >= partial_pressure,
        "{} < {}",
        final_pressure,
        partial_pressure
    );
}

#[test]
fn partial_bound_5() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor);

    let (signature, a) = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        let a = tensor::TensorBuilder::new("a", vec![13.into(), 32.into()])
            .stride_dim(1)
            .finish::<f32, _>(&mut builder);
        (builder.get(), a)
    };

    let mut builder = Builder::new(&signature, context.device());

    let ld_a = a.load(vec![TilingPattern::default()], &mut builder);
    let dim1 = builder.open_dim_ex(ir::Size::new_const(26), DimKind::THREAD);
    let _ = builder.mov(&0f32);

    builder.order(&ld_a.inst(), &dim1, Order::AFTER);

    let partial_pressure = {
        let space = builder.get_clone();
        let local_info = LocalInfo::compute(&space, context.params(), context.device());
        sum_pressure(
            context.params(), context.device(),
            &space,
            &local_info,
            BottleneckLevel::Global,
            &[],
            &ir::PartialSize::default(),
        )
    }
    .get_bottleneck(4);

    builder.action(Action::DimKind(ld_a[0][0], DimKind::UNROLL));

    let final_pressure = {
        let space = builder.get();
        let local_info = LocalInfo::compute(&space, context.params(), context.device());
        sum_pressure(
            context.params(), context.device(),
            &space,
            &local_info,
            BottleneckLevel::Global,
            &[],
            &ir::PartialSize::default(),
        )
    }
    .get_bottleneck(4);

    assert!(
        final_pressure * 1.001 >= partial_pressure,
        "{} < {}",
        final_pressure,
        partial_pressure
    );
}

#[test]
fn final_bound_0() {
    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor);

    let (x, y, z);
    let signature = {
        let mut builder = SignatureBuilder::new("test", &mut context);
        let n_size = builder.max_size("n", 1 << 25);
        x = builder.tensor::<f32>("x", vec![n_size.clone()], true);
        y = builder.tensor::<f32>("y", vec![n_size.clone()], true);
        z = builder.tensor::<f32>("z", vec![n_size], false);
        builder.get()
    };

    let tiling = TilingPattern::new_fixed(&[1024, 4]);
    let mut builder = Builder::new(&signature, context.device());

    let ld_x = x.load(vec![tiling.clone()], &mut builder);
    let ld_y = y.load(vec![tiling], &mut builder);
    let mad_dim = builder.open_mapped_dim(&ld_x[0]);
    let x_op = ld_x.dim_map(&[&mad_dim], ir::DimMapScope::Global(()), &mut builder);
    let y_op = ld_y.dim_map(&[&mad_dim], ir::DimMapScope::Global(()), &mut builder);
    let mad = tensor::VirtualTensor::new(
        builder.mad(&x_op, &4.33f32, &y_op),
        vec![mad_dim.clone()],
    );
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
    builder.action(Action::InstFlag(ld_x.inst(), InstFlag::NO_CACHE));
    builder.action(Action::InstFlag(ld_y.inst(), InstFlag::CACHE_GLOBAL));
    builder.action(Action::InstFlag(st_z.inst(), InstFlag::CACHE_GLOBAL));
    let space = builder.get();
    let bound = model::bound(&space, &context);
    let kernel = codegen::Function::build(&space);
    let eval = unwrap!(context.evaluate(&kernel, EvalMode::TestBound));
    assert!(eval * 1.001 >= bound.value(), "{:.2e} < {}", eval, bound);
}
*/
