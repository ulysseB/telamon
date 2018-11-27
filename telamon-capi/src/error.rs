use std;
use std::cell::RefCell;

use failure;
use libc;

use telamon;

/// Indicates if a telamon function exited correctly.
#[repr(C)]
pub enum TelamonStatus {
    Ok,
    Fail,
}

#[derive(Debug, Fail)]
#[repr(C)]
pub enum Error {
    #[fail(display = "{}", _0)]
    IRError(#[cause] telamon::ir::Error),
    #[fail(display = "invalid argument: {}", _0)]
    InvalidArgument(String),
    #[fail(display = "unexpected NULL pointer")]
    NullPointer,
    #[fail(display = "unknown error")]
    UnknownError,
    #[fail(display = "{}", _0)]
    StrUtf8Error(#[cause] std::str::Utf8Error),
    #[fail(display = "{}", _0)]
    Custom(failure::Error),
}

impl From<telamon::ir::Error> for Error {
    fn from(error: telamon::ir::Error) -> Error {
        Error::IRError(error)
    }
}

impl From<telamon::ir::TypeError> for Error {
    fn from(error: telamon::ir::TypeError) -> Error {
        Error::IRError(telamon::ir::Error::from(error))
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(error: std::str::Utf8Error) -> Error {
        Error::StrUtf8Error(error)
    }
}

impl From<()> for Error {
    fn from(_error: ()) -> Error {
        Error::UnknownError
    }
}

impl From<failure::Error> for Error {
    fn from(err: failure::Error) -> Error {
        Error::Custom(err)
    }
}

/// Prints the error message in a string. Returns `null` if no error was
/// present. The caller is responsible for freeing the string with `free`.
#[no_mangle]
pub extern "C" fn telamon_strerror() -> *mut libc::c_char {
    ERROR.with(|error| {
        error
            .borrow()
            .as_ref()
            .map(|error| {
                std::ffi::CString::new(error.to_string())
                    .map(std::ffi::CString::into_raw)
                    .unwrap_or(std::ptr::null_mut())
            }).unwrap_or(std::ptr::null_mut())
    })
}

#[no_mangle]
pub unsafe extern "C" fn telamon_str_free(str: *mut libc::c_char) {
    std::mem::drop(std::ffi::CString::from_raw(str))
}

thread_local! {
    pub static ERROR: RefCell<Option<Error>> = RefCell::new(None);
}

/// Helper macro that unwraps a result. Exits with `$error` and sets the global
/// `ERROR` variable when an error is encountered.
///
/// When no value is specified for `$error`, returns with
/// `TELAMON_STATUS_FAIL`. When `null` is specified instead, exits with a null
/// mutable pointer.
#[macro_export]
macro_rules! unwrap_or_exit {
    ($result:expr) => {
        unwrap_or_exit!($result, $crate::error::TelamonStatus::Fail)
    };
    ($result:expr,null) => {
        unwrap_or_exit!($result, ::std::ptr::null_mut())
    };
    ($result:expr, $error:expr) => {
        match $result {
            Ok(data) => data,
            Err(errno) => exit!(errno, $error),
        }
    };
}

#[macro_export]
macro_rules! exit {
    ($errno:expr) => {
        exit!($errno, $crate::error::TelamonStatus::Fail)
    };
    ($errno:expr,null) => {
        exit!($errno, ::std::ptr::null_mut())
    };
    ($errno:expr, $error:expr) => {{
        $crate::error::ERROR.with(|error_var| {
            *error_var.borrow_mut() = Some($errno.into());
        });

        return $error;
    }};
}

#[macro_export]
macro_rules! exit_if_null {
    ($e:expr) => {
        exit_if_null!($e, $crate::error::TelamonStatus::Fail)
    };
    ($e:expr,null) => {
        exit_if_null!($e, ::std::ptr::null_mut())
    };
    ($e:expr, $error:expr) => {{
        if $e.is_null() {
            exit!($crate::error::Error::NullPointer, $error)
        }
    }};
}
