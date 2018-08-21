//! Defines common kernels used to test and benchmark Telamon.
#[cfg(feature = "cuda")]
extern crate cuda_sys;
extern crate itertools;
extern crate libc;
extern crate ndarray;
extern crate num;
extern crate num_cpus;
extern crate rand;
extern crate rayon;
extern crate telamon;
#[macro_use]
extern crate telamon_utils as utils;
#[macro_use]
extern crate log;

mod kernel;

pub mod linalg;
pub mod statistics;

pub use kernel::{analyze_bounds, Kernel};

use itertools::Itertools;
use telamon::device::{self, ArgMap, Context};
use telamon::helper::tensor::DimSize;
use telamon::helper::{Builder, LogicalDim, SignatureBuilder};
use telamon::search_space::*;
use telamon::{explorer, model};

// FIXME: update the kernel interface so build_body only return one candidates as this
// simplifies the code.

/// Creates a candidate from the search space and registers the tile sizes in it.
fn build_candidate<'a>(
    space: SearchSpace<'a>,
    ctx: &device::Context,
) -> explorer::Candidate<'a> {
    let bound = model::bound(&space, ctx);
    explorer::Candidate::new(space, bound)
}

/// Creates a `DimSize`. If the instantiate flag is true, it uses a constant size,
/// otherwise it creates a parameter with the given name.
fn create_size<'a, AM>(
    value: i32,
    name: &'a str,
    is_generic: bool,
    builder: &mut SignatureBuilder<AM>,
) -> DimSize<'a>
where
    AM: ArgMap + Context,
{
    if is_generic {
        builder.scalar(name, value);
        name.into()
    } else {
        (value as u32).into()
    }
}

/// Generates a list of possible tiling factors and set the number of tiling dimension
/// to a valid value.
fn generate_tile_sizes(
    size: u32,
    max_tiling: u32,
    max_tile_dims: usize,
) -> (Vec<u32>, usize) {
    let tile_sizes = (2..std::cmp::min(max_tiling, size / 2) + 1)
        .filter(|&t| max_tiling % t == 0)
        .take(NumericSet::MAX_LEN)
        .collect_vec();
    let num_tile_dims = tile_sizes
        .iter()
        .take(max_tile_dims)
        .scan(1, |state, t| {
            *state *= t;
            Some(*state)
        })
        .take_while(|t| tile_sizes.binary_search(t).is_ok())
        .count();
    (tile_sizes, num_tile_dims)
}

/// Limits the possible sizes of dimensions.
fn limit_tile_size(dim: &LogicalDim, max_size: &[u32], builder: &mut Builder) {
    if let LogicalDim::Composite {
        dims,
        tiling_factors,
        ..
    } = dim
    {
        for (id, &max_size) in dims[1..].iter().cloned().zip(max_size) {
            let factors = NumericSet::new_leq(tiling_factors, max_size);
            builder.action(Action::Size(id, factors));
        }
    }
}

/// A scalar that can be used as the data type for tests.
pub trait Scalar:
    device::ScalarArgument
    + ndarray::LinalgScalar
    + ndarray::ScalarOperand
    + PartialOrd
    + std::ops::Neg<Output = Self>
{
    /// Returns the amount of allowed error in tests.
    fn epsilon() -> Self {
        Self::zero()
    }

    /// Indicates if the scalar can be considered as zero in the context of error
    /// checking.
    fn is_err_ok(self) -> bool {
        self > Self::epsilon() || -self > Self::epsilon()
    }
}

impl Scalar for f32 {
    fn epsilon() -> Self {
        10e-6
    }
}

impl Scalar for f64 {
    fn epsilon() -> Self {
        10e-6
    }
}

// FIXME: implement kernels
// tensor reduction
// floyd warshall: for a fixed K
// n_bodies: in n dimensions, need sqrt. Only perform a single step
// cell: FC+relu
// pooling: cell + pooling
// FIXME: extend the IR to support the following kernels
// backpropagation ?
// gemver > might want to load twice separately
// atax, CNN > need global broadcast
// dicgi, mvt, dot > need global reduction
// 2mm, two-level NN > need global bcast or global reduction
// lstm: too complex for now
