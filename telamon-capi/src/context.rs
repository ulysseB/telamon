use libc;
use std::sync::Arc;

use telamon::{device, ir};

use error::TelamonStatus;

// Pointers to `device::Context` are not C-like pointers.  Instead, they are fat pointers
// containing both a regular pointer to the object and a pointer to the vtable. Thus, we
// define wrappers to encapsulate the pointers in an opaque type and we return pointers to
// the wrappers to C users.

/// Description of the evaluation context. In particular, in contains the mapping between
/// argument names and argument values.
pub struct Context(pub(crate) *mut dyn DContext<'static>);

pub struct ContextRef(pub(crate) *const dyn device::Context);

#[no_mangle]
pub unsafe extern "C" fn telamon_context_free(context: *mut Context) {
    std::mem::drop(Box::from_raw(context))
}

impl Context {
    pub unsafe fn as_inner(&self) -> &dyn DContext<'_> {
        std::mem::transmute::<&dyn DContext<'static>, &dyn DContext<'_>>(&*self.0)
    }

    pub unsafe fn as_inner_mut(&mut self) -> &mut dyn DContext<'_> {
        std::mem::transmute::<&mut dyn DContext<'static>, &mut dyn DContext<'_>>(
            &mut *self.0,
        )
    }
}

pub trait DContext<'a>: device::Context {
    fn bind_scalar(&mut self, param: &telamon::ir::Parameter, value: &dyn DValue);

    fn bind_array(
        &mut self,
        param: &::telamon::ir::Parameter,
        size: usize,
    ) -> Arc<dyn device::ArrayArgument + 'a>;

    fn as_context(&self) -> &(dyn device::Context + 'a);
}

impl<'a> AsRef<dyn device::Context + 'a> for dyn DContext<'a> {
    fn as_ref(&self) -> &(dyn device::Context + 'a) {
        self.as_context()
    }
}

macro_rules! dtype {
    ($($to:ident : $dt:ident => $ty:ident ,)*) => {
        #[repr(C)]
        pub enum DType { $($dt,)* }

        pub trait DValue {
            fn dtype(&self) -> DType;

            $(
                fn $to(&self) -> $ty {
                    panic!(concat!("Can't convert to ", stringify!($ty)));
                }
            )*
        }

        $(
            impl DValue for $ty {
                fn dtype(&self) -> DType {
                    DType::$dt
                }

                fn $to(&self) -> $ty {
                    *self
                }
            }
        )*

        impl<'a, AM> DContext<'a> for AM
        where
            AM: device::ArgMap + device::Context + 'a,
            AM::Array: 'a,
        {
            fn bind_scalar(
                &mut self,
                param: &telamon::ir::Parameter,
                value: &dyn DValue,
            ) {
                match value.dtype() {
                    $(
                        DType::$dt => device::ArgMap::bind_scalar::<$ty>(
                            self, param, value.$to(),
                        )
                    ),*
                }
            }

            fn bind_array(
                &mut self,
                param: &telamon::ir::Parameter,
                size: usize,
            ) -> Arc<dyn device::ArrayArgument + 'a> {
                device::ArgMap::bind_array::<i8>(self, param, size)
            }

            fn as_context(&self) -> &(dyn device::Context + 'a){
                self
            }
        }
    }
}

dtype! {
    to_i8  : I8  => i8,
    to_i16 : I16 => i16,
    to_i32 : I32 => i32,
    to_i64 : I64 => i64,
    to_f32 : F32 => f32,
    to_f64 : F64 => f64,
}

/// Allocates and binds an array to the given parameter. `size` is given in bytes.
///
/// The allocated array is managed by the context and doesn't need to be explicitely
/// destroyed.
#[no_mangle]
pub unsafe extern "C" fn telamon_bind_array(
    context: *mut Context,
    param: *const ir::Parameter,
    size: libc::size_t,
) -> TelamonStatus {
    exit_if_null!(context);
    exit_if_null!(param);

    (*context).as_inner_mut().bind_array(&*param, size);

    TelamonStatus::Ok
}

/// Binds an `int8_t` to a parameter.
#[no_mangle]
pub unsafe extern "C" fn telamon_bind_int8(
    context: *mut Context,
    param: *const ir::Parameter,
    value: libc::int8_t,
) -> TelamonStatus {
    exit_if_null!(context);
    exit_if_null!(param);

    (*context).as_inner_mut().bind_scalar(&*param, &value);

    TelamonStatus::Ok
}

/// Binds an `int16_t` to a parameter.
#[no_mangle]
pub unsafe extern "C" fn telamon_bind_int16(
    context: *mut Context,
    param: *const ir::Parameter,
    value: libc::int16_t,
) -> TelamonStatus {
    exit_if_null!(context);
    exit_if_null!(param);

    (*context).as_inner_mut().bind_scalar(&*param, &value);

    TelamonStatus::Ok
}

/// Binds an `int32_t` to a parameter.
#[no_mangle]
pub unsafe extern "C" fn telamon_bind_int32(
    context: *mut Context,
    param: *const ir::Parameter,
    value: libc::int32_t,
) -> TelamonStatus {
    exit_if_null!(context);
    exit_if_null!(param);

    (*context).as_inner_mut().bind_scalar(&*param, &value);

    TelamonStatus::Ok
}

/// Binds an `int64_t` to a parameter.
#[no_mangle]
pub unsafe extern "C" fn telamon_bind_int64(
    context: *mut Context,
    param: *const ir::Parameter,
    value: libc::int64_t,
) -> TelamonStatus {
    exit_if_null!(context);
    exit_if_null!(param);

    (*context).as_inner_mut().bind_scalar(&*param, &value);

    TelamonStatus::Ok
}

/// Binds a `float` to a parameter.
#[no_mangle]
pub unsafe extern "C" fn telamon_bind_float(
    context: *mut Context,
    param: *const ir::Parameter,
    value: libc::c_float,
) -> TelamonStatus {
    exit_if_null!(context);
    exit_if_null!(param);

    (*context).as_inner_mut().bind_scalar(&*param, &value);

    TelamonStatus::Ok
}

/// Binds a `double` to a parameter.
#[no_mangle]
pub unsafe extern "C" fn telamon_bind_double(
    context: *mut Context,
    param: *const ir::Parameter,
    value: libc::c_double,
) -> TelamonStatus {
    exit_if_null!(context);
    exit_if_null!(param);

    (*context).as_inner_mut().bind_scalar(&*param, &value);

    TelamonStatus::Ok
}
