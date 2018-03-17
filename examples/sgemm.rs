extern crate env_logger;
extern crate telamon;
#[macro_use]
extern crate log;
extern crate rayon;

mod common;

#[allow(unused_imports)]
use telamon::{explorer, helper, ir};
use telamon::device::{Context, cuda};
use telamon::search_space::{Action, DimKind, InstFlag, SearchSpace, Order};
use rayon::prelude::*;

const M: i32 = 1024;
const K: i32 = 1024;
const N: i32 = 1024;

const DATA_TYPE: ir::Type = ir::Type::F(32);

// FIXME: allow global tmp mem
// FIXME: Attack plan
// * Other ideas:
// - try to vectorize A accesses
// - play with the number of register allocated per thread
// - allow more induction variable pattern: eg, for limiting 64bit computations or
//      improving register pressure
//      > eg. use the indv var as the loop index
//      > do not cast to i64 from the begining
//      > recompute to avoid storing
//      > Share induction variables for C
//      > cast to i64 on the host ?
//      > take cvt instructions into account in the perf model
//      move induction pattern computations around to reduce pressure/improve latency

fn main() {
    env_logger::init();
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor);
    // Declares the function signature and the arguments to use for the evaluation.
    let (a, b, c);
    let signature = &{
        let mut builder = helper::SignatureBuilder::new("sgemm", &mut context);
        builder.param("m", M);
        builder.param("n", N);
        builder.param("k", K);
        a = builder.array("a", 4*(M*K) as usize);
        b = builder.array("b", 4*(K*N) as usize);
        c = builder.array("c", 4*(M*N) as usize);
        builder.get()
    };
    let device = context.device();
    let candidates = (0..6).into_par_iter().flat_map(|tile_1| {
        (0..std::cmp::min(8-tile_1, 5)).into_par_iter().map(move |tile_2| {
            gen_gemm(signature, device, 1u32 << tile_1, 1u32 << tile_2, a, b, c)
        })
    }).collect();
    //let candidates = vec![gen_gemm(signature, device, 32, 4, a, b, c)];
    common::gen_best(candidates, &context);
}

fn gen_gemm<'a>(signature: &'a ir::Signature, device: &'a telamon::device::Device,
                tile_1: u32, tile_2: u32, a: ir::mem::Id, b: ir::mem::Id, c: ir::mem::Id)
        -> SearchSpace<'a>
{
    let mut full_tiling = vec![tile_1, tile_2];
    let mut reduction_tiling = vec![tile_1];
    full_tiling.retain(|&x| x > 1);
    reduction_tiling.retain(|&x| x > 1);

    let mut builder = helper::Builder::new(signature, device);
    let size_m = builder.param_size("m");
    let size_n = builder.param_size("n");
    let size_k = builder.param_size("k");

    // Load A from global memory.
    let a_ld_dim_m = builder.open_tiled_dim(size_m, &full_tiling);
    let a_ld_dim_k = builder.open_tiled_dim(size_k, &reduction_tiling);
        let (a_addr, a_pattern) = builder.tensor_access(
            &"a", a, &DATA_TYPE, &[&a_ld_dim_m, &a_ld_dim_k]);
        let a_ld = builder.ld_nc(DATA_TYPE, &a_addr, a_pattern);
    builder.close_dim(&a_ld_dim_k);
    builder.close_dim(&a_ld_dim_m);

    // Load B from global memory
    let b_ld_dim_k = builder.open_mapped_dim(&a_ld_dim_k);
    let b_ld_dim_n = builder.open_tiled_dim(size_n, &full_tiling);
        let (b_addr, b_pattern) = builder.tensor_access(
            &"b", b, &DATA_TYPE, &[&b_ld_dim_k, &b_ld_dim_n]);
        let b_ld = builder.ld_nc(DATA_TYPE, &b_addr, b_pattern);
    builder.close_dim(&b_ld_dim_n);
    builder.close_dim(&b_ld_dim_k);

    // Initialize the accumulator.
    let init_dim_m = builder.open_mapped_dim(&a_ld_dim_m);
    let init_dim_n = builder.open_mapped_dim(&b_ld_dim_n);
        let acc_init = builder.mov(&0f32);

    // Accumulate the product of A and B.
    let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
    let acc_dim_n = builder.open_mapped_dim(&init_dim_n);
    let acc_dim_k = builder.open_mapped_dim(&b_ld_dim_k);
        let a_op = builder.dim_map(
            a_ld, &[(&a_ld_dim_m, &acc_dim_m), (&a_ld_dim_k, &acc_dim_k)],
            ir::DimMapScope::Global);
        let b_op = builder.dim_map(
            b_ld, &[(&b_ld_dim_n, &acc_dim_n), (&b_ld_dim_k, &acc_dim_k)],
            ir::DimMapScope::Global);
        let acc = builder.mad(&a_op, &b_op, &helper::Reduce(acc_init));
    builder.close_dim(&acc_dim_k);

    // Store the result in C.
    let c_st_dim_m = builder.open_mapped_dim(&acc_dim_m);
    let c_st_dim_n = builder.open_mapped_dim(&acc_dim_n);
        let (c_addr, c_pattern) = builder.tensor_access(
            &"c", c, &DATA_TYPE, &[&c_st_dim_m, &c_st_dim_n]);
        let c_st = builder.st(&c_addr, &acc, c_pattern);

    // Order for correctness.
    builder.order(&c_st, &acc_dim_k, Order::AFTER);

    // Arbitrary constrains to reduce the search space
    builder.action(Action::InstFlag(a_ld, InstFlag::MEM_CG | InstFlag::MEM_NC));
    builder.action(Action::InstFlag(b_ld, InstFlag::MEM_CG | InstFlag::MEM_NC));
    builder.action(Action::InstFlag(c_st, InstFlag::MEM_CS));

    builder.action(Action::DimKind(init_dim_n[0], DimKind::BLOCK));
    builder.action(Action::DimKind(init_dim_m[0], DimKind::BLOCK));
    /*builder.action(Action::DimKind(thread_dim_0_n, DimKind::THREAD_Y));
    builder.action(Action::DimKind(thread_dim_0_m, DimKind::THREAD_X));
    builder.action(Action::DimKind(unroll_dim_0_n, DimKind::UNROLL));
    builder.action(Action::DimKind(unroll_dim_0_m, DimKind::UNROLL));
    builder.order(unroll_dim_0_n.into(), unroll_dim_0_m.into(), Order::OUTER);
    builder.order(unroll_dim_1_n.into(), unroll_dim_1_m.into(), Order::INNER);

    builder.action(Action::DimKind(k0_dim, DimKind::LOOP));
    builder.order(ld_k0_dim.into(), k0_dim.into(), Order::MERGED);
    builder.action(Action::DimKind(a_ld_thread_dim_0, DimKind::THREAD_Y));
    builder.action(Action::DimKind(a_ld_thread_dim_1, DimKind::THREAD_X));
    builder.action(Action::DimKind(a_ld_unroll_dim, DimKind::UNROLL));
    builder.action(Action::DimKind(b_ld_unroll_dim, DimKind::VECTOR));
    builder.order(a_ld_thread_dim_1.into(), b_ld_thread_dim_1.into(), Order::MERGED);
    builder.order(a_ld_thread_dim_0.into(), b_ld_thread_dim_0.into(), Order::MERGED);

    builder.action(Action::DimKind(k1_dim, DimKind::UNROLL));
    builder.action(Action::DimKind(unroll_dim_2_n, DimKind::VECTOR));

    let mut space = builder.get();
    let mem_0 = ir::mem::InternalId(0);
    let (d23, d24, d25) = (ir::dim::Id {id: 23}, ir::dim::Id {id: 24}, ir::dim::Id {id: 25});
    let (d26, d27, d28) = (ir::dim::Id {id: 26}, ir::dim::Id {id: 27}, ir::dim::Id {id: 28});
    assert!(space.lower_layout(mem_0, vec![d23, d24, d25], vec![d26, d27, d28]).is_ok());
    let mem_1 = ir::mem::InternalId(1);
    let (d29, d30, d31) = (ir::dim::Id {id: 29}, ir::dim::Id {id: 30}, ir::dim::Id {id: 31});
    let (d32, d33, d34) = (ir::dim::Id {id: 32}, ir::dim::Id {id: 33}, ir::dim::Id {id: 34});
    assert!(space.lower_layout(mem_1, vec![d29, d30, d31], vec![d32, d33, d34]).is_ok());
    let actions = vec![
        Action::DimKind(d25, DimKind::VECTOR),
        Action::DimKind(d28, DimKind::VECTOR),
        Action::DimKind(d31, DimKind::VECTOR),
        Action::DimKind(d34, DimKind::VECTOR),
        Action::Order(d27.into(), d32.into(), Order::MERGED),
        Action::Order(d32.into(), k1_dim.into(), Order::MERGED),
    ];
    assert!(space.apply_decisions(actions).is_ok());
    space*/
    builder.get()
}
