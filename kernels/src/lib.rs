//! Defines common kernels used to test and benchmark Telamon.
#![deny(bare_trait_objects)]

mod kernel;

pub mod compose;
pub mod linalg;
pub mod statistics;

use std::fmt;

pub use crate::kernel::{analyze_bounds, Kernel, KernelBuilder};

use telamon::device::{self, ArgMap, Context};
use telamon::helper::tensor::DimSize;
use telamon::helper::{self, SignatureBuilder};
use telamon::{explorer, model, search_space};

use ::ndarray::{ArrayBase, Data, Dimension, FoldWhile, Zip};

/// Creates a candidate from the search space and registers the tile sizes in it.
fn build_candidate(
    space: search_space::SearchSpace,
    ctx: &dyn device::Context,
) -> explorer::Candidate {
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
        .and_broadcast(b)
        .fold_while((), |_, x, y| {
            if (*x - *y).abs() < A::atol() + A::rtol() * y.abs() {
                FoldWhile::Continue(())
            } else {
                FoldWhile::Done(())
            }
        })
        .is_done()
}

/// Information about an incorrect output from a kernel.
///
/// This is meant to be used as an error in a `Result`, and will print the corresponding to the
/// maximum and average errors when displayed.
#[derive(Debug)]
struct IncorrectOutputError<S> {
    max_absolute_error: S,
    sum_absolute_error: S,
    max_relative_error: S,
    sum_relative_error: S,
    num_above_threshold: usize,
    total_size: usize,
}

impl<S: Scalar> Default for IncorrectOutputError<S> {
    fn default() -> Self {
        IncorrectOutputError {
            max_absolute_error: S::zero(),
            sum_absolute_error: S::zero(),
            max_relative_error: S::zero(),
            sum_relative_error: S::zero(),
            num_above_threshold: 0,
            total_size: 0,
        }
    }
}

impl<S: Scalar> std::error::Error for IncorrectOutputError<S> {}

impl<S: Scalar> IncorrectOutputError<S> {
    fn max_absolute_error(&self) -> S {
        self.max_absolute_error
    }

    fn avg_absolute_error(&self) -> S {
        self.sum_absolute_error / S::from(self.total_size).unwrap()
    }

    fn max_relative_error(&self) -> S {
        self.max_relative_error
    }

    fn avg_relative_error(&self) -> S {
        self.sum_relative_error / S::from(self.total_size).unwrap()
    }

    fn fraction_above_threshold(&self) -> S {
        S::from(self.num_above_threshold).unwrap() / S::from(self.total_size).unwrap()
    }
}

impl<S> fmt::Display for IncorrectOutputError<S>
where
    S: Scalar,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "error max ± {:.4e} ({:.2}%), avg ± {:.4e} ({:.2}%), with {}/{} ({:.2}%) values above threshold",
            self.max_absolute_error(),
            self.max_relative_error() * S::from(100).unwrap(),
            self.avg_absolute_error(),
            self.avg_relative_error() * S::from(100).unwrap(),
            self.num_above_threshold,
            self.total_size,
            self.fraction_above_threshold() * S::from(100).unwrap(),
        )
    }
}

fn check_output<A, S, D, S2, E>(
    actual: &ArrayBase<S, D>,
    expected: &ArrayBase<S2, E>,
) -> Result<(), IncorrectOutputError<A>>
where
    A: Scalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
    E: Dimension,
{
    if allclose(actual, expected) {
        Ok(())
    } else {
        Err(Zip::from(actual)
            .and_broadcast(expected)
            .fold_while(
                IncorrectOutputError::default(),
                |output_diff, actual, expected| {
                    let absolute_error = (*actual - *expected).abs();
                    let relative_error = absolute_error / expected.abs();
                    FoldWhile::Continue(IncorrectOutputError {
                        max_absolute_error: if absolute_error
                            > output_diff.max_absolute_error
                        {
                            absolute_error
                        } else {
                            output_diff.max_absolute_error
                        },
                        sum_absolute_error: output_diff.sum_absolute_error
                            + absolute_error,
                        max_relative_error: if relative_error
                            > output_diff.max_relative_error
                        {
                            relative_error
                        } else {
                            output_diff.max_relative_error
                        },
                        sum_relative_error: output_diff.sum_relative_error
                            + relative_error,
                        num_above_threshold: output_diff.num_above_threshold
                            + if absolute_error < A::atol() + A::rtol() * expected.abs() {
                                0
                            } else {
                                1
                            },
                        total_size: output_diff.total_size + 1,
                    })
                },
            )
            .into_inner())
    }
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
