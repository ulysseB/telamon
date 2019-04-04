//! Defines common kernels used to test and benchmark Telamon.
#![deny(bare_trait_objects)]

mod kernel;

pub mod linalg;
pub mod statistics;

pub use crate::kernel::{analyze_bounds, Kernel, KernelBuilder, MemInit};

use telamon::device::{self, ArgMap, Context};
use telamon::helper::tensor::DimSize;
use telamon::helper::{self, SignatureBuilder};
use telamon::{explorer, model, search_space};

use ::ndarray::{ArrayBase, Data, Dimension, FoldWhile, Zip};

/// Creates a candidate from the search space and registers the tile sizes in it.
fn build_candidate<'a>(
    space: search_space::SearchSpace<'a>,
    ctx: &dyn device::Context,
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
    AM: ArgMap<'a> + Context,
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

/// Returns `true` if two arrays are element-wise equal within a tolerance.
///
/// The tolerance values are defined by the absolute and relative offsets from the `Scalar` trait
/// for the corresponding type.
///
/// The relative difference (`rtol` * abs(`b`)) and the absolute difference `atol` are added
/// together and compared against the absolute difference between `a` and `b`.
///
/// # Panics
///
/// If broadcasting the arrays to the same shape is not possible.
fn allclose<A, S, D, S2, E>(a: &ArrayBase<S, D>, b: &ArrayBase<S2, E>) -> bool
where
    A: Scalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
    E: Dimension,
{
    !Zip::from(a)
        .and(b.broadcast(a.raw_dim()).unwrap())
        .fold_while((), |_, x, y| {
            if (*x - *y).abs() < A::atol() + A::rtol() * y.abs() {
                FoldWhile::Continue(())
            } else {
                FoldWhile::Done(())
            }
        })
        .is_done()
}

/// A scalar that can be used as the data type for tests.
pub trait Scalar: device::ScalarArgument + ndarray::NdFloat {
    /// Absolute tolerance for errors.
    fn atol() -> Self;

    /// Relative tolerance for errors.
    fn rtol() -> Self;
}

impl Scalar for f32 {
    fn atol() -> Self {
        1e-8
    }

    fn rtol() -> Self {
        1e-5
    }
}

impl Scalar for f64 {
    fn atol() -> Self {
        1e-8
    }

    fn rtol() -> Self {
        1e-5
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
