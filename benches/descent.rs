//! Benchmarks how fast it is to decend in the search tree.
#[macro_use]
extern crate criterion;
extern crate env_logger;
extern crate telamon;
extern crate rand;
#[macro_use]
extern crate lazy_static;

use criterion::Criterion;
use telamon::{ir, device, explorer, helper};
use telamon::explorer::choice::ActionEx;
use telamon::search_space::*;
use rand::Rng;

/// Configure the bencher.
fn config_criterion() -> Criterion {
    Criterion::default()
        .sample_size(50)
        .configure_from_args()
}

lazy_static! {
    /// A fake GPU description, used only to know which candidates are valid.
    static ref DEVICE: device::cuda::Gpu = device::cuda::Gpu::dummy();

    static ref MM_SIGNATURE: MMSig = MMSig::signature();
    static ref MM: SearchSpace<'static> = MM_SIGNATURE.build_body();
}

/// Stores the signature and the external arrays IDs for matrix-matrix multiplication.
struct MMSig {
    signature: ir::Signature,
    a: ir::mem::Id,
    b: ir::mem::Id,
    c: ir::mem::Id,
}

impl MMSig {
    fn signature() -> Self {
        let mut signature = ir::Signature::new("mm".to_string());
        signature.add_scalar("m".to_string(), ir::Type::I(32));
        signature.add_scalar("n".to_string(), ir::Type::I(32));
        signature.add_scalar("k".to_string(), ir::Type::I(32));
        let a = signature.add_array("a".to_string());
        let b = signature.add_array("b".to_string());
        let c = signature.add_array("c".to_string());
        MMSig { signature, a, b, c }
    }

    fn build_body(&self) -> SearchSpace {
        const DATA_TYPE: ir::Type = ir::Type::F(32);
        let mut builder = helper::Builder::new(&self.signature, &*DEVICE);
        let m_size = builder.param_size("m");
        let n_size = builder.param_size("n");
        let k_size = builder.param_size("k");

        let ld_a_m = builder.open_tiled_dim(m_size, &[16, 4]);
        let ld_a_k = builder.open_tiled_dim(k_size.clone(), &[16]);
        let (ptr, pattern) = builder.tensor_access(
            &"a", self.a, &DATA_TYPE, &[&ld_a_m, &ld_a_k]);
        let ld_a =  builder.ld_nc(DATA_TYPE.clone(), &ptr, pattern);
        builder.close_dim(&ld_a_m);
        builder.close_dim(&ld_a_k);

        let ld_b_k = builder.open_tiled_dim(k_size, &[16]);
        let ld_b_n = builder.open_tiled_dim(n_size, &[16, 4]);
        let (ptr, pattern) = builder.tensor_access(
            &"b", self.b, &DATA_TYPE, &[&ld_b_k, &ld_b_n]);
        let ld_b = builder.ld_nc(DATA_TYPE, &ptr, pattern);
        builder.close_dim(&ld_b_k);
        builder.close_dim(&ld_b_n);

        let init_m = builder.open_mapped_dim(&ld_a_m);
        let init_n = builder.open_mapped_dim(&ld_b_n);
        let init = builder.mov(&0f32);

        let acc_m = builder.open_mapped_dim(&init_m);
        let acc_n = builder.open_mapped_dim(&init_n);
        let acc_k = builder.open_mapped_dim(&ld_b_k);
        let a_op = builder.dim_map(ld_a, &[(&ld_a_m, &acc_m), (&ld_a_k, &acc_k)],
        ir::DimMapScope::Global);
        let b_op = builder.dim_map(ld_b, &[(&ld_b_k, &acc_k), (&ld_b_n, &acc_n)],
        ir::DimMapScope::Global);
        let acc = builder.mad(&a_op, &b_op, &helper::Reduce(init));

        builder.close_dim(&acc_k);
        let st_m = builder.open_mapped_dim(&acc_m);
        let st_n = builder.open_mapped_dim(&acc_n);
        let (ptr, pattern) = builder.tensor_access(
            &"c", self.c, &DATA_TYPE, &[&st_m, &st_n]);
        let st = builder.st(&ptr, &acc, pattern);
        // order for correctness.
        builder.order(&st, &acc_k, Order::AFTER);
        builder.get()
    }
}

/// Benchmarks full descents in the search tree.
fn mm_descent(c: &mut Criterion) {
    let _ = env_logger::try_init();
    c.bench_function("mm descent", |b| b.iter(|| {
        let mut space = MM.clone();
        while let Some(mut choice) = {
            let choice = explorer::choice::list(&space).next();
            choice
        } {
           let id = rand::thread_rng().gen_range(0, choice.len());
           let res = match choice.swap_remove(id) {
                ActionEx::TileSizes(..) => panic!(),
                ActionEx::Action(action) => space.apply_decisions(vec![action]),
                ActionEx::LowerLayout { mem, ref st_dims, ref ld_dims } =>
                    space.lower_layout(mem, st_dims.clone(), ld_dims.clone()),
           };
           if res.is_err() { return; }
        }
    }));
}

/// Benchmarks full descents in the search tree, with a copy at each level.
fn mm_descent_copy(c: &mut Criterion) {
    let _ = env_logger::try_init();
    c.bench_function("mm descent with copy", |b| b.iter(|| {
        let mut spaces = vec![];
        let mut space = MM.clone();
        while let Some(mut choice) = {
            let choice = explorer::choice::list(&space).next();
            choice
        } {
            spaces.push(space.clone());
            let id = rand::thread_rng().gen_range(0, choice.len());
            let res = match choice.swap_remove(id) {
                ActionEx::TileSizes(..) => panic!(),
                ActionEx::Action(action) => space.apply_decisions(vec![action]),
                ActionEx::LowerLayout { mem, ref st_dims, ref ld_dims } =>
                    space.lower_layout(mem, st_dims.clone(), ld_dims.clone()),
            };
            if res.is_err() { return; }
        }
    }));
}

criterion_group! {
    name = benches;
    config = config_criterion();
    targets = mm_descent, mm_descent_copy
}

criterion_main!(benches);
