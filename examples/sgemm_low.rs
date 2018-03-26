#![feature(conservative_impl_trait)]
extern crate env_logger;
extern crate telamon;
extern crate itertools;
#[macro_use]
extern crate log;
extern crate rayon;

mod common;

#[allow(unused_imports)]
use telamon::{explorer, helper, ir};
use telamon::device::{Context, cuda};
use telamon::search_space::{Action, DimKind, InstFlag, SearchSpace, Order};
use rayon::prelude::*;
use common::*;

const M: i32 = 1024;
const K: i32 = 1024;
const N: i32 = 1024;

const DATA_TYPE: ir::Type = ir::Type::F(32);

// TODO(search_space):
// * explore unrolled dimensions order
// * explore inst-dim orders

// FIXME: Attack plan
// * Other ideas:
// - try to vectorize A accesses
// - move index computation around to reduce register pressure / improve latency
// - play with the number of register allocated per thread
// - allow more induction variable pattern: eg, for limiting 64bit computations or
//      improving register pressure
//      > eg. use the indv var as the loop index
//      > do not cast to i64 from the begining
//      > recompute to avoid storing
//      > Share induction variables for C
//      > cast to i64 on the host ?
//      > take cvt instructions into account in the perf model

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
        builder.param("alpha", 1f32);
        a = builder.array("a", 4*(M*K) as usize);
        b = builder.array("b", 4*(K*N) as usize);
        builder.param("beta", 1f32);
        c = builder.array("c", 4*(M*N) as usize);
        builder.get()
    };
    let device = context.device();
    let candidates = (1..6).into_par_iter().flat_map(|tile_1| {
        (1..std::cmp::min(8-tile_1, 5)).into_par_iter().map(move |tile_2| {
            gen_gemm(signature, device, 1u32 << tile_1, 1u32 << tile_2, a, b, c)
        })
    }).collect();
    //let candidate = gen_gemm(signature, device, 32, 4, a, b, c);
    common::gen_best(candidates, &context, &file_name("sgemm_low", DATA_TYPE, &[], true));
}

fn gen_gemm<'a>(signature: &'a ir::Signature, device: &'a telamon::device::Device,
                tile_1: u32, tile_2: u32, a: ir::mem::Id, b: ir::mem::Id, c: ir::mem::Id)
        -> SearchSpace<'a> {

    let mut builder = helper::Builder::new(signature, device);
    let tile_1_size = builder.cst_size(tile_1);
    let tile_2_size = builder.cst_size(tile_2);
    let data_len = DATA_TYPE.len_byte().unwrap();
    let data_size = builder.cst_size(DATA_TYPE.len_byte().unwrap());
    let tmp_mem_size = builder.cst_size(data_len*tile_1*tile_1*tile_2);
    //let a_tmp_mem = builder.allocate(tmp_mem_size.clone(), true);
    //let b_tmp_mem = builder.allocate(tmp_mem_size, true);
    let a_tmp_mem = builder.allocate_shared(tmp_mem_size.clone());
    let b_tmp_mem = builder.allocate_shared(tmp_mem_size);
    // Configure dimension sizes
    let m_tiled = builder.tile_size("m", tile_1 * tile_2);
    let n_tiled = builder.tile_size("n", tile_1 * tile_2);
    let k_tiled = builder.tile_size("k", tile_1);

    let n_t1_t2_incr = builder.size(&["k"], data_len*tile_1*tile_2, 1);
    let n_t2_incr = builder.size(&["k"], data_len*tile_2, 1);
    let n_incr = builder.size(&["k"], data_len, 1);
    let t2_incr = builder.cst_size(tile_2 * data_len);
    let t1_t2_incr = builder.cst_size(tile_1 * tile_2 * data_len);

    // Initialize the computations.
    let block_dim_n = builder.open_dim(n_tiled.clone());
    let block_dim_m = builder.open_dim(m_tiled.clone());
    let thread_dim_0_0 = builder.open_dim(tile_1_size.clone());
    let thread_dim_0_1 = builder.open_dim(tile_1_size.clone());
    let unroll_dim_0_0 = builder.open_dim(tile_2_size.clone());
    let unroll_dim_0_1 = builder.open_dim(tile_2_size.clone());
        let acc_init = builder.mov(&0f32);
    builder.close_dim(&unroll_dim_0_0);
    builder.close_dim(&unroll_dim_0_1);
    builder.close_dim(&thread_dim_0_0);
    builder.close_dim(&thread_dim_0_1);
    // Compute AxB in acc.
    let k0_dim = builder.open_dim(k_tiled);
    let ld_thread_dim_0 = builder.open_mapped_dim(&thread_dim_0_0);
    let ld_thread_dim_1 = builder.open_mapped_dim(&thread_dim_0_1);
    let a_ld_unroll_dim = builder.open_dim(tile_2_size.clone());
        // FIXME: vectorize loads from A ?
        let (a_addr, a_pattern) = builder.tensor_access(&"a", a, &DATA_TYPE,
            &[&block_dim_m, &ld_thread_dim_1, &a_ld_unroll_dim, &k0_dim, &ld_thread_dim_0]);
        let a_ld = builder.ld_nc(DATA_TYPE, &a_addr, a_pattern);
    builder.close_dim(&a_ld_unroll_dim);
    // Load B from global memory
    let b_ld_unroll_dim = builder.open_dim(tile_2_size.clone());
        let (b_addr, b_pattern) = builder.tensor_access(&"b", b, &DATA_TYPE,
            &[&k0_dim, &ld_thread_dim_1, &block_dim_n, &ld_thread_dim_0,
              &b_ld_unroll_dim]);
        let b_ld = builder.ld_nc(DATA_TYPE, &b_addr, b_pattern);
    builder.close_dim(&b_ld_unroll_dim);
    // Store A in shared memory.
    let a_st_tmp_unroll_dim = builder.open_mapped_dim(&a_ld_unroll_dim)[0];
        let (a_tmp_addr, a_tmp_st_pattern) = builder.tensor_access(
            &a_tmp_mem, a_tmp_mem.into(), &DATA_TYPE,
            &[&ld_thread_dim_1, &ld_thread_dim_0, &a_st_tmp_unroll_dim]);
        builder.st(&a_tmp_addr, &a_ld, a_tmp_st_pattern);
    builder.close_dim(&a_st_tmp_unroll_dim);
    // Store B in shared memory.
    let b_st_tmp_unroll_dim = builder.open_mapped_dim(&b_ld_unroll_dim)[0];
        let (b_tmp_addr, b_tmp_st_pattern) = builder.tensor_access(
                &b_tmp_mem, b_tmp_mem.into(), &DATA_TYPE,
                &[&ld_thread_dim_1, &ld_thread_dim_0, &b_st_tmp_unroll_dim]);
        builder.st(&b_tmp_addr, &b_ld, b_tmp_st_pattern);
    builder.close_dim(&b_st_tmp_unroll_dim);
    builder.close_dim(&ld_thread_dim_1);
    builder.close_dim(&ld_thread_dim_0);

    // Load from shared and multiply.
    let thread_dim_1_0 = builder.open_mapped_dim(&thread_dim_0_0);
    let thread_dim_1_1 = builder.open_mapped_dim(&thread_dim_0_1);
    let k1_dim = builder.open_dim(tile_1_size.clone());
        // FIXME: make sure their is no shared conflict
    // Load from a_tmp.
    let a_ld_tmp_unroll_dim = builder.open_dim(tile_2_size.clone());
        let (a_tmp_ld_addr, a_tmp_ld_pattern) = builder.tensor_access(
            &a_tmp_mem, a_tmp_mem.into(), &DATA_TYPE,
            &[&thread_dim_1_1, &k1_dim, &a_ld_tmp_unroll_dim]);
        let a_val = builder.ld(DATA_TYPE, &a_tmp_ld_addr, a_tmp_ld_pattern);
    builder.close_dim(&a_ld_tmp_unroll_dim);
    // Load from b_tmp.
    let b_ld_tmp_unroll_dim = builder.open_dim(tile_2_size.clone());
        let (b_tmp_ld_addr, b_tmp_ld_pattern) = builder.tensor_access(
            &b_tmp_mem, b_tmp_mem.into(), &DATA_TYPE,
            &[&k1_dim, &thread_dim_1_0, &b_ld_tmp_unroll_dim]);
        let b_val = builder.ld(DATA_TYPE, &b_tmp_ld_addr, b_tmp_ld_pattern);
    builder.close_dim(&b_ld_tmp_unroll_dim);
    // Multiply.
    let unroll_dim_1_0 = builder.open_mapped_dim(&unroll_dim_0_0);
    let unroll_dim_1_1 = builder.open_mapped_dim(&unroll_dim_0_1);
        let a_op = builder.dim_map(a_val, &[(&a_ld_tmp_unroll_dim, &unroll_dim_1_1)],
                                   ir::DimMapScope::Local);
        let b_op = builder.dim_map(b_val, &[(&b_ld_tmp_unroll_dim, &unroll_dim_1_0)],
                                   ir::DimMapScope::Local);
        let acc = builder.mad(&a_op, &b_op, &helper::Reduce(acc_init));
    builder.close_dim(&unroll_dim_1_0);
    builder.close_dim(&unroll_dim_1_1);
    builder.close_dim(&k1_dim);
    builder.close_dim(&k0_dim);
    builder.close_dim(&thread_dim_1_1);
    builder.close_dim(&thread_dim_1_0);

    // Store the result in C.
    let thread_dim_2_0 = builder.open_mapped_dim(&thread_dim_0_0)[0];
    let thread_dim_2_1 = builder.open_mapped_dim(&thread_dim_0_1)[0];
    let unroll_dim_2_0 = builder.open_mapped_dim(&unroll_dim_1_1)[0];
        // FIXME: improve indexes.
        let c_addr_0 = builder.induction_var(&"c", vec![
            (block_dim_m, n_t1_t2_incr),
            (thread_dim_2_1, n_t2_incr),
            (unroll_dim_2_0, n_incr),
            (block_dim_n, t1_t2_incr),
            (thread_dim_2_0, t2_incr),
        ]);
    builder.reopen_mapped_dim(&thread_dim_2_0, &thread_dim_1_0);
    builder.reopen_mapped_dim(&thread_dim_2_1, &thread_dim_1_1);
    let c_ld_unroll_dim = builder.open_dim(tile_2_size.clone());
        let c_ld_pattern = builder.tensor_access_pattern(c, &DATA_TYPE,
            &[&block_dim_m, &thread_dim_2_1, &unroll_dim_2_0,
              &block_dim_n, &thread_dim_2_0, &c_ld_unroll_dim]);
        let c_ld_addr = builder.induction_var(
            &c_addr_0, vec![(c_ld_unroll_dim, data_size.clone())]);
        let c_ld = builder.ld(DATA_TYPE, &c_ld_addr, c_ld_pattern);
    builder.close_dim(&c_ld_unroll_dim);
    let unroll_dim_2_1 = builder.open_mapped_dim(&unroll_dim_1_0)[0];
        let acc_alpha = builder.mul(&acc, &"alpha");
        builder.reopen_mapped_dim(&unroll_dim_2_1, &c_ld_unroll_dim);
        let c_val = builder.mad(&c_ld, &"beta", &acc_alpha);
    builder.close_dim(&unroll_dim_2_1);
    let c_st_unroll_dim = builder.open_mapped_dim(&unroll_dim_2_1)[0];
        let c_st_pattern = builder.tensor_access_pattern(c, &DATA_TYPE,
            &[&block_dim_m, &thread_dim_2_1, &unroll_dim_2_0,
              &block_dim_n, &thread_dim_2_0, &c_st_unroll_dim]);
        let c_st_addr = builder.induction_var(&c_addr_0, vec![(c_st_unroll_dim, data_size)]);
        let c_st = builder.st(&c_st_addr, &c_val, c_st_pattern);

    // Order for correctness.
    builder.order(&acc_alpha, &k0_dim, Order::AFTER);
    builder.order(&acc_alpha, &k1_dim, Order::AFTER);
    builder.order(&a_val, &ld_thread_dim_0, Order::AFTER);
    builder.order(&a_val, &ld_thread_dim_1, Order::AFTER);
    builder.order(&a_val, &a_st_tmp_unroll_dim, Order::AFTER);
    builder.order(&b_val, &ld_thread_dim_0, Order::AFTER);
    builder.order(&b_val, &ld_thread_dim_1, Order::AFTER);
    builder.order(&b_val, &b_st_tmp_unroll_dim, Order::AFTER);
    builder.action(Action::DimKind(k0_dim, DimKind::SEQUENTIAL));
    builder.action(Action::DimKind(k1_dim, DimKind::SEQUENTIAL));

    // Arbitrary constrains to reduce the search space
    builder.action(Action::InstFlag(a_ld, InstFlag::MEM_CG | InstFlag::MEM_NC));
    builder.action(Action::InstFlag(b_ld, InstFlag::MEM_CG | InstFlag::MEM_NC));
    builder.action(Action::InstFlag(c_ld, InstFlag::MEM_CS));
    builder.action(Action::InstFlag(c_st, InstFlag::MEM_CS));

    // Best actions:
    // [Action(Order(dim 19, dim 18, ACTIVE_IN)), Action(Order(dim 4, dim 5, ACTIVE_OUT)),
    // Action(DimKind(10, VECTOR)), Action(DimKind(25, VECTOR)), Action(DimKind(12, VECTOR)),
    // Action(DimKind(0, BLOCK)), Action(DimKind(17, VECTOR)), Action(DimKind(23, VECTOR)),
    // Action(DimKind(11, VECTOR)), Action(DimKind(9, UNROLL)), Action(DimKind(15, UNROLL)),
    // Action(DimKind(16, VECTOR)), Action(DimKind(1, LOOP)), Action(DimKind(7, THREAD_Y)),
    // Action(DimKind(20, THREAD_Y)), Action(DimKind(5, UNROLL))]

    builder.get()
}
