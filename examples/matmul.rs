//! Telamon's demo.
//!
//! To run the demo:
//! ```
//! cargo run --example=matmul --features=cuda
//! ```
extern crate env_logger;
extern crate telamon;

use telamon::device::{cuda, Context};
use telamon::{explorer, helper, ir, search_space};

// Define the problem size.
const M: u32 = 1024;
const N: u32 = 1024;
const K: u32 = 1024;

// Specifies how M, N and K dimensions should be tiled.
const M_TILING: &[u32] = &[32, 4];
const N_TILING: &[u32] = &[32, 4];
const K_TILING: &[u32] = &[32];

fn main() {
    // Step 0. Setup logging and the CUDA interface.
    env_logger::init();
    let executor = cuda::Executor::init();

    // Step 1. Define the kernel signature and the parameters we optimize for.
    let mut context = cuda::Context::new(&executor);
    let (a, b, c);
    let signature = {
        let mut builder = helper::SignatureBuilder::new("matmul", &mut context);
        // Declare 3 integer parameters `m`, `n` and `k`.
        builder.scalar("m", M as i32);
        builder.scalar("n", N as i32);
        builder.scalar("k", K as i32);
        // Declare 3 matricies of floats of size `m*k`, `k*n` and `m*n`.
        // `a` and `b` are read-only but not `c`.
        a = builder.tensor::<f32>("a", vec!["m".into(), "k".into()], true);
        b = builder.tensor::<f32>("b", vec!["k".into(), "n".into()], true);
        c = builder.tensor::<f32>("c", vec!["m".into(), "n".into()], false);
        // Build the signature.
        builder.get()
    };

    // Step 2. Define the kernel body
    let mut builder = helper::Builder::new(&signature, context.device());
    // Create two loop nests to load A and B
    // for i in 0..M:
    //   for k in 0..K:
    //     ld_a = load A[i][k]
    // for k in 0..K:
    //   for j in 0..N:
    //     ld_b = load B[k][j]
    // Creates a load instruction in a loop nest of the dimensionality of the tensor. Each
    // dimension is tilied with the given tiling pattern. Thus `M_TILING.len() + K_TILING.len() +
    // 2` iteration dimensions are actually created here.
    let ld_a = a.load(&[M_TILING, K_TILING], &mut builder);
    let ld_b = b.load(&[K_TILING, N_TILING], &mut builder);
    // Create a loop nest n*m to intialize c.
    // for i in 0..M:
    //   for j in 0..N:
    //     init = 0
    let init_dim_m = builder.open_mapped_dim(&ld_a[0]);
    let init_dim_n = builder.open_mapped_dim(&ld_b[1]);
    let init = builder.mov(&0f32);
    // Accumulate `a[i][.]*b[.][j]`.
    // for i in 0..M:
    //   for j in 0..N:
    //     for k in 0..K:
    //       acc = init += a * b
    let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
    let acc_dim_n = builder.open_mapped_dim(&init_dim_n);
    let acc_dim_k = builder.open_mapped_dim(&ld_a[1]);
    // Values are taken pointwise from the dimensions mapped to the currently open decisions.
    // Here, `acc_dim_m` is mapped to `init_dim_m` so iteration `i` on `acc_dim_m` uses the
    // value of `init` produced at iteration `i` of `init_dim_m`.
    //
    // For `ld_a` and `ld_b`, the dimensions are not mapped with `open_mapped_dim` so we manualy
    // specify the mapping. Additionaly, we set the mapping scope to global, which allows values
    // to be stored in memory between the two dimensions are executed.
    let global_scope = ir::DimMapScope::Global(());
    let op_a = ld_a.dim_map(&[&acc_dim_m, &acc_dim_k], global_scope, &mut builder);
    let op_b = ld_b.dim_map(&[&acc_dim_k, &acc_dim_n], global_scope, &mut builder);
    let acc = builder.mad(&op_a, &op_b, &helper::Reduce(init));
    builder.close_dim(&acc_dim_k);
    // Store the result in `C`.
    // for i in 0..M:
    //   for j in 0..N:
    //     store C[i][j] <- acc
    let acc = helper::tensor::VirtualTensor::new(acc, vec![acc_dim_m, acc_dim_n]);
    acc.store(&c, &mut builder);

    // Step 3. Apply manual decisions and retrieve the search space.
    // Don't use caches to load `A`.
    builder.action(search_space::Action::InstFlag(
        ld_a.inst(),
        search_space::InstFlag::MEM_CS,
    ));
    let space = builder.get();

    // Step 4. Launch a search.
    explorer::find_best(&Default::default(), &context, vec![space]).unwrap();
}
