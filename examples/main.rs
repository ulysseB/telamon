// TODO(cleanup): remove the cuda/non-cuda switch and put in in Cargo.toml instead once
// the required features will have landed.
extern crate env_logger;
extern crate telamon;
#[macro_use]
extern crate log;
extern crate itertools;

mod common;

use common::*;
use telamon::helper;
use telamon::helper::tensor::{Tensor, VirtualTensor};
use telamon::device::cuda;
use telamon::device::Context;
use telamon::ir;

// Finds the best implementation of the saxpy function. saxpy computes y = a.x + y, where
// "a" is a scalar and "x" and "y" are vectors of size "n".
fn main() {
    // Enable logging.
    env_logger::init();
    let executor = cuda::Executor::init();
    gen_axpy(1 << 26, 10.0, ir::Type::F(32), false, &executor);
}

fn gen_axpy(n: i32, a: f32,
            data_type: ir::Type,
            instantiate: bool,
            executor: &cuda::Executor) {
    // Initializes the evaluation context.
    let mut context = cuda::Context::new(&executor);
    // Declares the function signature and the arguments to use for the evaluation.
    let (x, y);
    let signature = {
        let mut builder = helper::SignatureBuilder::new("saxpy", &mut context);
        // Create two scalar parameters, with the values N and A used for the evaluation.
        let n = create_size(n, "n", instantiate, &mut builder);
        builder.param("a", a);
        // Allocates two arrays of size N.
        x = Tensor::new("x", vec![n], data_type, true, &mut builder);
        y = Tensor::new("y", vec![n], data_type, false, &mut builder);
        builder.get()
    };
    // Declares the bofy of the function.
    let function = {
        let mut builder = helper::Builder::new(&signature, context.device());
        let tiling = &[1024, 4]; // FIXME: try more tile sizes

        let ld_x = x.load(&[tiling], &mut builder);
        let ld_y = y.load(&[tiling], &mut builder);
        let mad_dim = builder.open_mapped_dim(&ld_x[0]);
        let x_op = ld_x.dim_map(&[&mad_dim], ir::DimMapScope::Global, &mut builder);
        let y_op = ld_y.dim_map(&[&mad_dim], ir::DimMapScope::Global, &mut builder);
        let mad = VirtualTensor::new(builder.mad(&x_op, &"a", &y_op), vec![mad_dim]);
        mad.store(&y, &mut builder);
        builder.get()
    };
    // Explore the search space.
    gen_best(vec![function], &context, &file_name("axpy", data_type, &[n], instantiate));
}


// FIXME: Polybench
// dot, mm, 2mm, gemm
// gesummv, gemver, atax, bicg, doitgen, mvt > might want to load twice separately
// floyd-warshall, nbody single pass on K
