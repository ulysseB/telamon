use telamon::device::{self, ScalarArgument};
use libc;
use std::sync::{Mutex, MutexGuard};
use utils::unwrap;

pub enum ArgLock<'a> {
    Scalar(*mut libc::c_void),
    Arr(MutexGuard<'a, Vec<i8>>),
}

pub trait Argument: Sync + Send {
    fn size(&self) -> Option<u32>;
    fn arg_lock(&self) -> ArgLock;
}

pub struct CpuArray(Mutex<Vec<i8>>);

impl CpuArray {
    pub fn new(len: usize) -> Self {
        CpuArray(Mutex::new(vec![0; len]))
    }

    fn size(&self) -> u32 {
        let CpuArray(ref vec_mutex) = self;
        let array = unwrap!(vec_mutex.lock());
        array.len() as u32
    }
}

impl Argument for CpuArray {
    fn size(&self) -> Option<u32> {
        Some(self.size())
    }

    fn arg_lock(&self) -> ArgLock {
        let CpuArray(mutex) = self;
        ArgLock::Arr(unwrap!(mutex.lock()))
    }
}

impl device::ArrayArgument for CpuArray {
    fn read_i8(&self) -> Vec<i8> {
        let CpuArray(ref vec_mutex) = self;
        let array = unwrap!(vec_mutex.lock());
        array.clone()
    }

    fn write_i8(&self, slice: &[i8]) {
        let CpuArray(ref vec_mutex) = self;
        let mut array = unwrap!(vec_mutex.lock());
        *array = slice.to_vec();
    }
}

impl Argument for Box<dyn ScalarArgument> {
    fn size(&self) -> Option<u32> {
        self.as_size()
    }

    fn arg_lock(&self) -> ArgLock {
        ArgLock::Scalar(self.as_ref().raw_ptr() as *mut libc::c_void)
    }
}
