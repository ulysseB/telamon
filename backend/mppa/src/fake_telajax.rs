use std::{
    ffi::CStr,
    result::Result,
    sync::RwLock,
};

static DEVICE: Device = Device {};

pub struct Buffer<T:Copy> {
    pub len: usize,
    data: RwLock<Vec<T>>
}

impl<T:Copy> Buffer<T> {
    pub fn new(_: &'static Device, len: usize) -> Self {
        Buffer {
            len,
            data: RwLock::new(Vec::with_capacity(len)),
        }
    }

    pub fn raw_ptr(&self) -> *const libc::c_void {
        self.data.read().unwrap().as_ptr() as *const libc::c_void
    }

    pub fn read(&self) -> Result<Vec<T>, ()> {
        Ok(self.data.read().unwrap().clone())
    }

    pub fn write(&self, data: &[T]) -> Result<(), ()> {
        *self.data.write().unwrap() = data.to_vec();
        Ok(())
    }
}

pub struct Device {}

impl Device {
    pub fn get() -> &'static Device {
        &DEVICE
    }

    pub fn build_wrapper(
        &self,
        _: &CStr,
        _: &CStr,
    ) -> Result<Wrapper, ()> {
        Ok(Wrapper {})
    }

    /// Compiles a kernel.
    pub fn build_kernel(
        &self,
        _: &CStr,
        _: &CStr,
        _: &CStr,
        _: &Wrapper,
    ) -> Result<Kernel, ()> {
        Ok(Kernel {})
    }

    pub fn execute_kernel(&self, _: &mut Kernel) -> Result<(), ()> {
        unimplemented!("This fake executor is just here to allow compilation")
    }
}

pub struct Mem {
}

impl Mem {
    pub fn get_mem_size() -> usize {
        8
    }
}

pub struct Kernel {
}

impl Kernel {
    pub fn set_num_clusters(&mut self, _: usize) -> Result<(), ()> {
        Ok(())
    }

    pub fn set_args(
        &mut self,
        _: &[usize],
        _: &[*const libc::c_void],
    ) -> Result<(), ()> {
        Ok(())
    }
}

pub struct Wrapper {}
