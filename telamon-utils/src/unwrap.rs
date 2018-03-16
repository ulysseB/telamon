//! A macro to help debug unwraps.
use std;

/// Panics after if the value cannot be unwraped.
#[macro_export]
macro_rules! unwrap {
    ($e:expr) => {
        $crate::unwrap::Unwrap::unwrap($e,
            format_args!("in module {}, file {}, line {}, column {}",
                         module_path!(), file!(), line!(), column!()));
    };
    ($e:expr, $($msg:tt)*) => {
        $crate::unwrap::Unwrap::unwrap($e,
            format_args!(": {} in module {}, file {}, line {}, column {}",
                         format_args!($($msg)*),
                         module_path!(), file!(), line!(), column!()));
    }
}

pub trait Unwrap {
    /// The type retruned by unwraping.
    type Output;

    /// Unwraps the value or panics with the given message.
    fn unwrap(self, msg: std::fmt::Arguments) -> Self::Output;
}

impl<T> Unwrap for Option<T> {
    type Output = T;

    fn unwrap(self, msg: std::fmt::Arguments) -> Self::Output {
        match self {
            Some(t) => t,
            None => panic!("failed unwrapping an option {}", msg),
        }
    }
}

impl<T, E: std::fmt::Debug> Unwrap for Result<T, E> {
    type Output = T;

    fn unwrap(self, msg: std::fmt::Arguments) -> Self::Output {
        match self {
            Ok(t) => t,
            Err(err) => panic!("failed with error {:?} {}", err, msg),
        }
    }
}
