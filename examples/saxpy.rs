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
        let n_size = builder.param_size("n");
        // Load X.
        let ld_x_dim = builder.open_tiled_dim(n_size, &[1024, 4]);
        let (x_ptr, x_pat) = builder.tensor_access(&"x", x, &DATA_TYPE, &[&ld_x_dim]);
        let ld_x = builder.ld_nc(DATA_TYPE, &x_ptr, x_pat);
        // Load Y
        let ld_y_dim = builder.open_mapped_dim(&ld_x_dim);
        let (y_ptr, y_pat) = builder.tensor_access(&"y", y, &DATA_TYPE, &[&ld_y_dim]);
        let ld_y = builder.ld_nc(DATA_TYPE, &y_ptr, y_pat);
        // Multiply X and Y.
        let mad_dim = builder.open_mapped_dim(&ld_y_dim);
        let x_op = builder.dim_map(ld_x, &[(&ld_x_dim, &mad_dim)], ir::DimMapScope::Global);
        let y_op = builder.dim_map(ld_y, &[(&ld_y_dim, &mad_dim)], ir::DimMapScope::Global);
        let mad = builder.mad(&x_op, &"a", &y_op);
        // Store the result.
        let st_dim = builder.open_mapped_dim(&mad_dim);
        let (y_ptr, y_pat) = builder.tensor_access(&"y", y, &DATA_TYPE, &[&st_dim]);
        builder.st(&y_ptr, &mad, y_pat);

        builder.get()
    };

    // Explore the search space.
    common::gen_best(vec![function], &context);
}
