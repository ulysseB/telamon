use libc;
use std::sync::Arc;

use telamon::{
    codegen,
    device::{self, ArgMapExt},
    ir,
};

use error::TelamonStatus;

pub trait DContext<'a>: device::ArgMap<'a> + device::Context {
    fn as_context(&self) -> &(dyn device::Context + 'a);
}

impl<'a, C> DContext<'a> for C
where
    C: device::ArgMap<'a> + device::Context,
{
    fn as_context(&self) -> &(dyn device::Context + 'a) {
        self
    }
}

impl<'a> device::Context for Box<dyn DContext<'a>> {
    fn device(&self) -> &device::Device {
        (&**self).device()
    }

    fn evaluate(
        &self,
        space: &codegen::Function,
        mode: device::EvalMode,
    ) -> Result<f64, ()> {
        (&**self).evaluate(space, mode)
    }

    fn benchmark(&self, space: &codegen::Function, num_samples: usize) -> Vec<f64> {
        (&**self).benchmark(space, num_samples)
    }

    fn async_eval<'b, 'c>(
        &self,
        num_workers: usize,
        mode: device::EvalMode,
        inner: &(Fn(&mut device::AsyncEvaluator<'b, 'c>) + Sync),
    ) {
        (&**self).async_eval(num_workers, mode, inner)
    }

    fn param_as_size(&self, name: &str) -> Option<u32> {
        (&**self).param_as_size(name)
    }

    fn eval_size(&self, size: &codegen::Size) -> u32 {
        (&**self).eval_size(size)
    }
}

impl<'a> device::ArgMap<'a> for Box<dyn DContext<'a>> {
    fn bind_erased_scalar(
        &mut self,
        param: &ir::Parameter,
        value: Box<dyn device::ScalarArgument>,
    ) {
        (&mut **self).bind_erased_scalar(param, value)
    }

    fn bind_erased_array(
        &mut self,
        param: &ir::Parameter,
        t: ir::Type,
        len: usize,
    ) -> Arc<dyn device::ArrayArgument + 'a> {
        (&mut **self).bind_erased_array(param, t, len)
    }
}

impl<'a> AsRef<dyn device::Context + 'a> for dyn DContext<'a> {
    fn as_ref(&self) -> &(dyn device::Context + 'a) {
        (*self).as_context()
    }
}

// Pointers to `device::Context` are not C-like pointers.  Instead, they are fat pointers
// containing both a regular pointer to the object and a pointer to the vtable. Thus, we
// define wrappers to encapsulate the pointers in an opaque type and we return pointers to
// the wrappers to C users.

/// Description of the evaluation context. In particular, in contains the mapping between
/// argument names and argument values.
pub struct Context(Box<dyn DContext<'static>>);

pub struct ContextRef(pub(crate) *const dyn device::Context);

#[no_mangle]
pub unsafe extern "C" fn telamon_context_ref_free(context_ref: *mut ContextRef) {
    std::mem::drop(Box::from_raw(context_ref))
}

#[no_mangle]
pub unsafe extern "C" fn telamon_context_free(context: *mut Context) {
    std::mem::drop(Box::from_raw(context))
}

impl Context {
    pub fn new<'a, C: device::ArgMap<'a> + device::Context>(context: C) -> Self {
        unsafe {
            Context(std::mem::transmute::<
                Box<dyn DContext<'a>>,
                Box<dyn DContext<'static>>,
            >(Box::new(context)))
        }
    }

    pub unsafe fn as_inner(&self) -> &dyn DContext<'_> {
        std::mem::transmute::<&dyn DContext<'static>, &dyn DContext<'_>>(&self.0)
    }

    pub unsafe fn as_inner_mut<'a>(&mut self) -> &mut Box<dyn DContext<'a>> {
        std::mem::transmute::<&mut Box<dyn DContext<'static>>, &mut Box<dyn DContext<'a>>>(
            &mut self.0,
        )
    }
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

    (*context).as_inner_mut().bind_array::<i8>(&*param, size);

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

    (*context).as_inner_mut().bind_scalar(&*param, value);

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

    (*context).as_inner_mut().bind_scalar(&*param, value);

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

    (*context).as_inner_mut().bind_scalar(&*param, value);

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

    (*context).as_inner_mut().bind_scalar(&*param, value);

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

    (*context).as_inner_mut().bind_scalar(&*param, value);

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

    (*context).as_inner_mut().bind_scalar(&*param, value);

    TelamonStatus::Ok
}
