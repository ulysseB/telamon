//! Defines common kernels used to test and benchmark Telamon.
#[cfg(feature = "cuda")]
extern crate cuda_sys;
extern crate itertools;
extern crate libc;
#[macro_use]
extern crate log;
extern crate ndarray;
extern crate num;
extern crate num_cpus;
extern crate rand;
extern crate rayon;
extern crate telamon;
#[macro_use]
extern crate telamon_utils as utils;

mod kernel;

pub mod linalg;
pub mod statistics;

pub use kernel::{analyze_bounds, Kernel};

use telamon::device::{self, ArgMap, Context};
use telamon::helper::tensor::DimSize;
use telamon::helper::{self, SignatureBuilder};
use telamon::{explorer, model, search_space};

/// Creates a candidate from the search space and registers the tile sizes in it.
fn build_candidate<'a>(
    space: search_space::SearchSpace<'a>,
    ctx: &device::Context,
) -> explorer::Candidate<'a> {
    let bound = model::bound(&space, ctx);
    explorer::Candidate::new(space, bound)
}

/// Creates a `DimSize`. If the instantiate flag is true, it uses a constant size,
/// otherwise it creates a parameter with the given name.
fn create_size<'a, AM: ?Sized>(
    value: i32,
    name: &'a str,
    is_generic: bool,
    builder: &mut SignatureBuilder<AM>,
) -> DimSize<'a>
where
    AM: ArgMap + Context,
{
    if is_generic {
        builder.max_size(name, value as u32)
    } else {
        (value as u32).into()
    }
}

/// Returns the given tiling pattern or infer one.
fn infer_tiling(
    size: i32,
    given_pattern: &Option<helper::TilingPattern>,
    max_sizes: &[u32],
) -> helper::TilingPattern {
    given_pattern
        .clone()
        .unwrap_or_else(|| helper::TilingPattern::infer_pattern(size as u32, max_sizes))
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
