// TODO(cleanup): remove the cuda/non-cuda switch and put in in Cargo.toml instead once
// the required features will have landed.
extern crate env_logger;
extern crate telamon;
#[macro_use]
extern crate log;

mod common;

use telamon::helper;
use telamon::device::cuda;
use telamon::device::Context;
use telamon::ir;

const N: i32 = 1024 * 1024 * 64;
const A: f32 = 10.0;
const DATA_TYPE: ir::Type = ir::Type::F(32);

// Finds the best implementation of the saxpy function. saxpy computes y = a.x + y, where
// "a" is a scalar and "x" and "y" are vectors of size "n".
fn main() {
    // Enable logging.
    env_logger::init();
    // Initializes the evaluation context.
    let executor = cuda::Executor::init();
    let mut context = cuda::Context::new(&executor);
    // Declares the function signature and the arguments to use for the evaluation.
    let (x, y);
    let signature = {
        let mut builder = helper::SignatureBuilder::new("saxpy", &mut context);
        // Create two scalar parameters, with the values N and A used for the evaluation.
        builder.param("n", N);
        builder.param("a", A);
        // Allocates two arrays of size N.
        x = builder.array("x", 4*N as usize);
        y = builder.array("y", 4*N as usize);
        builder.get()
    };
    // Declares the bofy of the function.
    let function = {
        let mut builder = helper::Builder::new(&signature, context.device());
        // Declares the size of iteration dimensions.
        let size0 = builder.tile_size("n", 1024 * 4);
        let size1 = builder.cst_size(1024);
        let size2 = builder.cst_size(4);
        // Declares three iteration dimension to iterate on "n" with four levels of tile.
        let d0 = builder.open_dim(size0);
        let d1 = builder.open_dim(size1);
        let d2_0 = builder.open_dim(size2);
        // Load from "x" without enforcing memory coherency.
        let (x_ptr, x_pat) = builder.tensor_access(
            &"x", x, &DATA_TYPE, &[&d0, &d1, &d2_0]);
        let x_val = builder.ld_nc(DATA_TYPE, &x_ptr, x_pat);
        // Load from "y
        let d2_1 = builder.open_mapped_dim(&d2_0)[0];
        let (y_ptr, y_pat) = builder.tensor_access(
            &"y", y, &DATA_TYPE, &[&d0, &d1, &d2_1]);
        let y_val = builder.ld(DATA_TYPE, &y_ptr, y_pat.clone());
        builder.close_dim(&d2_1);
        // Compute the new element and stores it back to "y".
        let d2_2 = builder.open_mapped_dim(&d2_0)[0];
        let y_val = builder.dim_map(y_val, &[(&d2_1, &d2_2)], ir::DimMapScope::Local);
        let res = builder.mad(&"a", &x_val, &y_val);
        let d2_3 = builder.open_mapped_dim(&d2_2)[0];
        let (y_ptr, y_pat) = builder.tensor_access(
            &"y", y, &DATA_TYPE, &[&d0, &d1, &d2_3]);
        builder.st(&y_ptr, &res, y_pat);
        builder.get()
    };

    // Explore the search space.
    common::gen_best(vec![function], &context);
    //let (total, leaf) = explorer::count_candidates(function);
    //println!("{} candidate and {} leafs", total, leaf);
}
