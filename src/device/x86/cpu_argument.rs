use device::{self, Device, ScalarArgument};
use device::ArrayArgument;
use std::sync::{mpsc, Mutex, Arc};

pub struct CpuArray(Mutex<Vec<i8>>);

pub trait Argument: Sync + Send {
    fn size(&self) -> Option<u32>;
}

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

pub trait CpuScalarArg: Sync + Send {
    fn as_size(&self) -> Option<u32>;
}

impl Argument for CpuArray {
    fn size(&self) -> Option<u32> {
        Some(self.size())
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

impl<T> CpuScalarArg for T where T: ScalarArgument {
    fn as_size(&self) -> Option<u32> {
        //<Self as ScalarArgument>::as_size(self)
        self.as_size()
    }
}

impl Argument for Box<CpuScalarArg> {
    fn size(&self) -> Option<u32> {
        self.as_size()
    }
}
