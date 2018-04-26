#![cfg(feature="cuda")]
extern crate env_logger;
extern crate telamon;
extern crate telamon_kernels;

use telamon::device::cuda;
use telamon_kernels::{Kernel, linalg};

macro_rules! test_output {
    ($name:ident, $kernel:ty, $num_tests:expr, $params:expr) => {
        #[test]
        fn $name() {
            let _ = env_logger::try_init();
            let executor = cuda::Executor::init();
            let mut context = cuda::Context::new(&executor);
            <$kernel>::test_correctness($params, $num_tests, &mut context);
        }
    }
}

test_output!(axpy, linalg::Axpy<f32>, 100, 1 << 15);
test_output!(mv, linalg::MatVec<f32>, 100, (1<<4, 1<<2));
test_output!(gesummv, linalg::Gesummv<f32>, 100, (1<<4, 1<<4));
test_output!(matmul, linalg::MatMul<f32>, 100, (1<<4, 1<<4, 1<<4));
test_output!(doitgen, linalg::Doitgen<f32>, 100, (1<<4, 1<<4, 1<<4));

#[test]
fn inner_bound_0() {
    use telamon::device::Context;
    use telamon::ir::{self, dim, InstId, mem};
    use telamon::ir::DimMapScope::Global as GlobalScope;
    use telamon::{helper, model};
    use telamon::search_space::{Action, DimKind, InstFlag, Order};
    use telamon_kernels::create_size;

    let _ = env_logger::try_init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor);
    let (a, b, c);
    let signature = {
        let mut builder = helper::SignatureBuilder::new("bound_error", &mut context);
        let m_size = create_size(1024, "m", true, &mut builder);
        let n_size = create_size(1024, "n", true, &mut builder);
        let k_size = create_size(1024, "k", true, &mut builder);
        a = builder.tensor::<f32>("a", vec![m_size, k_size], true);
        b = builder.tensor::<f32>("b", vec![k_size, n_size], true);
        c = builder.tensor::<f32>("c", vec![m_size, n_size], false);
        builder.get()
    };

    let full_tiling = vec![4, 2];
    let small_tiling = vec![4];
    let mut builder = helper::Builder::new(&signature, context.device());

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

    let acc = helper::tensor::VirtualTensor::new(acc, vec![acc_dim_m, acc_dim_n]);
    let st_c = acc.store(&c, &mut builder);

    builder.order(&st_c.inst(), &acc_dim_k, Order::AFTER);
    builder.action(Action::InstFlag(ld_a.inst(), InstFlag::MEM_CG | InstFlag::MEM_NC));
    builder.action(Action::InstFlag(ld_b.inst(), InstFlag::MEM_CG | InstFlag::MEM_NC));
    builder.action(Action::InstFlag(st_c.inst(), InstFlag::MEM_CS));

    builder.action(Action::DimKind(init_dim_n[0], DimKind::BLOCK));
    builder.action(Action::DimKind(init_dim_m[0], DimKind::BLOCK));
    builder.action(Action::DimKind(ld_a[0][2], DimKind::UNROLL));
    builder.action(Action::DimKind(ld_a[0][1], DimKind::THREAD_Z));
    builder.action(Action::DimKind(ld_a[1][1], DimKind::THREAD_Y));
    builder.action(Action::DimKind(ld_b[0][1], DimKind::UNROLL));
    builder.action(Action::DimKind(ld_b[1][2], DimKind::THREAD_X));

    let mut space = builder.get();
    let partial_bound = model::bound(&space, &context);
    let actions = vec![
        Action::DimKind(dim::Id(8), DimKind::THREAD_Z),
        Action::DimKind(dim::Id(11), DimKind::THREAD_Z)];
    assert!(space.apply_decisions(actions).is_ok());
    assert!(space.lower_layout(mem::InternalId(0),
                               vec![dim::Id(30), dim::Id(32), dim::Id(31)],
                               vec![dim::Id(33), dim::Id(35), dim::Id(34)]).is_ok());
    assert!(space.lower_layout(mem::InternalId(1),
                               vec![dim::Id(37), dim::Id(38), dim::Id(36)],
                               vec![dim::Id(40), dim::Id(41), dim::Id(39)]).is_ok());
    let actions = vec![
        Action::DimKind(dim::Id(12), DimKind::THREAD_X),
        Action::DimKind(dim::Id(23), DimKind::UNROLL),
        Action::DimKind(dim::Id(29), DimKind::VECTOR),
        Action::DimKind(dim::Id(31), DimKind::VECTOR),
        Action::DimKind(dim::Id(35), DimKind::UNROLL),
        Action::DimKind(dim::Id(36), DimKind::VECTOR),
        Action::DimKind(dim::Id(39), DimKind::UNROLL),
        Action::DimKind(dim::Id(41), DimKind::UNROLL),
        Action::Order(ir::BBId::Dim(dim::Id(0)), ir::BBId::Inst(InstId(1)), Order::AFTER),
        Action::Order(ir::BBId::Dim(dim::Id(0)), ir::BBId::Inst(InstId(7)), Order::AFTER),
        Action::Order(ir::BBId::Dim(dim::Id(9)), ir::BBId::Dim(dim::Id(2)), Order::OUTER),
        Action::Order(ir::BBId::Dim(dim::Id(21)), ir::BBId::Inst(InstId(6)), Order::OUTER),
        Action::Order(ir::BBId::Dim(dim::Id(21)), ir::BBId::Inst(InstId(8)), Order::AFTER),
        Action::Order(ir::BBId::Dim(dim::Id(23)), ir::BBId::Dim(dim::Id(21)), Order::INNER),
        Action::Order(ir::BBId::Dim(dim::Id(23)), ir::BBId::Inst(InstId(6)), Order::OUTER),
        Action::Order(ir::BBId::Dim(dim::Id(41)), ir::BBId::Dim(dim::Id(39)), Order::OUTER)];
    assert!(space.apply_decisions(actions).is_ok());
    let final_bound = model::bound(&space, &context);
    assert!(partial_bound.value() <= final_bound.value() * 1.001,
            "{} > {}", partial_bound, final_bound);
}
