//! Defines common kernels used to test and benchmark Telamon.
#[cfg(feature="cuda")]
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

mod kernel;

pub mod linalg;
pub mod statistics;

pub use kernel::{Kernel, analyze_bounds};

use rayon::prelude::*;
use telamon::device::{self, ArgMap, Context};
use telamon::helper::SignatureBuilder;
use telamon::helper::tensor::DimSize;

/// Creates a `DimSize`. If the instantiate flag is true, it uses a constant size,
/// otherwise it creates a parameter with the given name.
fn create_size<'a, AM>(value: i32, name: &'a str,
                       is_generic: bool,
                       builder: &mut SignatureBuilder<AM>) -> DimSize<'a>
    where AM: ArgMap + Context
{
    if is_generic {
        builder.scalar(name, value);
        DimSize::Param(name)
    } else { DimSize::Const(value as u32) }
}

/// Removes tiles of size 1.
fn cleanup_tiling(tiling: &[u32]) -> Vec<u32> {
    tiling.iter().cloned().filter(|&t| t > 1).collect()
}

fn par_iter_product<I1, I2>(i1: I1, i2: I2)
    -> impl ParallelIterator<Item=(I1::Item, I2::Item)>
where
    I1: IntoParallelIterator, I1::Item: Clone + Sync,
    I2: IntoParallelIterator + Clone + Send + Sync
{
    i1.into_par_iter().flat_map(move |x| {
        i2.clone().into_par_iter().map(move |y| (x.clone(), y))
    })
}

/// A scalar that can be used as the data type for tests.
pub trait Scalar: device::ScalarArgument + ndarray::LinalgScalar + ndarray::ScalarOperand
                + PartialOrd + std::ops::Neg<Output=Self> {
    /// Returns the amount of allowed error in tests.
    fn epsilon() -> Self { Self::zero() }

    /// Indicates if the scalar can be considered as zero in the context of error
    /// checking.
    fn is_err_ok(self) -> bool {
        self > Self::epsilon() || -self > Self::epsilon()
    }
}

impl Scalar for f32 {
    fn epsilon() -> Self { 10e-6 }
}

impl Scalar for f64 {
    fn epsilon() -> Self { 10e-6 }
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
