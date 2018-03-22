#![feature(conservative_impl_trait)]
extern crate env_logger;
extern crate itertools;
extern crate telamon;
#[macro_use]
extern crate log;
extern crate rayon;

mod common;

#[allow(unused_imports)]
use common::*;
use telamon::{helper, ir};
use telamon::device::{Context, cuda};
use telamon::helper::tensor::{Tensor, VirtualTensor};
use telamon::search_space::{Action, DimKind, InstFlag, Order};
use rayon::prelude::*;

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
    gen_mm(1024, 1024, 1024, ir::Type::F(32), false, &executor);
}

fn gen_mm(m: i32, n: i32, k: i32,
          data_type: ir::Type,
          instantiate: bool,
          executor: &cuda::Executor) {
    let mut context = cuda::Context::new(&executor);
    let (a, b, c);
    let signature = &{
        let mut builder = helper::SignatureBuilder::new("mm", &mut context);
        let m = create_size(m, "m", instantiate, &mut builder);
        let n = create_size(n, "n", instantiate, &mut builder);
        let k = create_size(k, "k", instantiate, &mut builder);
        a = Tensor::new("a", vec![m, k], data_type, true, &mut builder);
        b = Tensor::new("b", vec![k, n], data_type, true, &mut builder);
        c = Tensor::new("c", vec![m, n], data_type, false, &mut builder);
        builder.get()
    };

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
            let a_op = ld_a.dim_map(
                &[&acc_dim_m, &acc_dim_k], ir::DimMapScope::Global, &mut builder);
            let b_op = ld_b.dim_map(
                &[&acc_dim_k, &acc_dim_n], ir::DimMapScope::Global, &mut builder);
            let acc = builder.mad(&a_op, &b_op, &helper::Reduce(acc_init));
        builder.close_dim(&acc_dim_k);

        let acc = VirtualTensor::new(acc, vec![acc_dim_n, acc_dim_m]);
        let st_c = acc.store(&c, &mut builder);

        // Order for correctness.
        builder.order(&st_c.inst(), &acc_dim_k, Order::AFTER);
        // Arbitrary constrains to reduce the search space
        builder.action(Action::InstFlag(ld_a.inst(), InstFlag::MEM_CG | InstFlag::MEM_NC));
        builder.action(Action::InstFlag(ld_b.inst(), InstFlag::MEM_CG | InstFlag::MEM_NC));
        builder.action(Action::InstFlag(st_c.inst(), InstFlag::MEM_CS));

        builder.action(Action::DimKind(init_dim_n[0], DimKind::BLOCK));
        builder.action(Action::DimKind(init_dim_m[0], DimKind::BLOCK));
        builder.get()
    }).collect();
    gen_best(candidates, &context, &file_name("mm", data_type, &[m, n, k], instantiate));

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
}
