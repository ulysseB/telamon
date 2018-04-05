//! Defines common kernels used to test and benchmark Telamon.
extern crate itertools;
extern crate telamon;
#[macro_use]
extern crate telamon_utils as utils;

mod kernel;

pub mod linalg;

pub use kernel::Kernel;

use telamon::helper::SignatureBuilder;
use telamon::helper::tensor::DimSize;

/// Creates a `DimSize`. If the instantiate flag is true, it uses a constant size,
/// otherwise it creates a parameter with the given name.
fn create_size<'a>(value: i32, name: &'a str,
                   is_generic: bool,
                   builder: &mut SignatureBuilder) -> DimSize<'a> {
    if is_generic {
        builder.param(name, value);
        DimSize::Param(name)
    } else { DimSize::Const(value as u32) }
}
