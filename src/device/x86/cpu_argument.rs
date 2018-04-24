<<<<<<< HEAD
use device::{self, ScalarArgument};
use libc;
use std::sync::{ Mutex };


pub trait Argument: Sync + Send {
    fn size(&self) -> Option<u32>;
    fn raw_ptr(&self) -> *mut libc::c_void;
}

pub struct CpuArray(Mutex<Vec<i8>>);

impl CpuArray {
    pub fn new(len: usize) -> Self {
        CpuArray(Mutex::new(vec![0; len]))
    }

    fn size(&self) -> u32 {
        let CpuArray(ref vec_mutex) = self;
        let array = vec_mutex.lock().unwrap();
        array.len() as u32
    }
}

impl Argument for CpuArray {
    fn size(&self) -> Option<u32> {
        Some(self.size())
    }
    fn raw_ptr(&self) -> *mut libc::c_void {
        let CpuArray(mutex) = self;
        let mut vec = mutex.lock().unwrap();
        vec.as_mut_ptr() as *mut libc::c_void
    }
}

impl device::ArrayArgument for CpuArray {
    fn read_i8(&self) -> Vec<i8> {
        let CpuArray(ref vec_mutex) = self;
        let array = vec_mutex.lock().unwrap();
        array.clone()
    }
    
    fn write_i8(&self, slice: &[i8]) {
        let CpuArray(ref vec_mutex) = self;
        let mut array = vec_mutex.lock().unwrap();
        *array = slice.to_vec();
    }
}

pub trait CpuScalarArg: Sync + Send {
    fn as_size(&self) -> Option<u32>;
    fn scal_raw_ptr(&self) -> *mut libc::c_void;
}

impl<T> CpuScalarArg for T where T: ScalarArgument {
    fn as_size(&self) -> Option<u32> {
        //<Self as ScalarArgument>::as_size(self)
        self.as_size()
    }

    fn scal_raw_ptr(&self) -> *mut libc::c_void {
        ScalarArgument::raw_ptr(self) as *mut libc::c_void
    }

}

impl Argument for Box<CpuScalarArg> {
    fn size(&self) -> Option<u32> {
        self.as_size()
    }

    fn raw_ptr(&self) -> *mut libc::c_void {
        self.scal_raw_ptr()
    }
}
