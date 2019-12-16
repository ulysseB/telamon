//! Allows the execution of kernels on the GPU.
use crate::api::wrapper::*;
use crate::api::{Argument, Executor};
use libc;
use num::integer::div_rem;
use telamon::device;

/// An array allocated on a CUDA device.
pub struct Array<T> {
    len: usize,
    array: *mut CudaArray,
    executor: Executor,
    t: std::marker::PhantomData<T>,
}

impl<T> Array<T> {
    fn context(&self) -> &CudaContext {
        self.executor.raw()
    }

    /// Allocates a new array on the device.
    pub fn new(executor: Executor, len: usize) -> Self {
        let n_bytes = (len * std::mem::size_of::<T>()) as u64;
        Array {
            len,
            array: unsafe { allocate_array(executor.raw(), n_bytes) },
            t: std::marker::PhantomData,
            executor,
        }
    }

    /// Copies the array to the host.
    pub fn copy_to_host(&self) -> Vec<T> {
        let mut vec = Vec::with_capacity(self.len);
        unsafe {
            vec.set_len(self.len);
            let host_ptr = vec.as_mut_ptr() as *mut libc::c_void;
            copy_DtoH(self.context(), self.array, host_ptr, self.byte_len() as u64);
        }
        vec
    }

    /// Copies an array from the host.
    pub fn copy_from_host(&self, vec: &[T]) {
        assert_eq!(self.len, vec.len());
        unsafe {
            let host_ptr = vec.as_ptr() as *const libc::c_void;
            copy_HtoD(self.context(), host_ptr, self.array, self.byte_len() as u64);
        }
    }

    /// Returns the number of bytes in the array.
    fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    fn clone_from(&mut self, src: &Self) {
        assert_eq!(self.len, src.len);
        unsafe {
            copy_DtoD(
                self.context(),
                src.array,
                self.array,
                self.byte_len() as u64,
            );
        }
    }
}

impl<T> Clone for Array<T> {
    fn clone(&self) -> Self {
        let mut new_array = Self::new(self.executor.clone(), self.len);
        new_array.clone_from(self);
        new_array
    }

    fn clone_from(&mut self, src: &Self) {
        assert_eq!(self.len, src.len);
        unsafe {
            copy_DtoD(
                self.context(),
                src.array,
                self.array,
                self.byte_len() as u64,
            );
        }
    }
}

impl<T> Drop for Array<T> {
    fn drop(&mut self) {
        unsafe {
            free_array(self.context(), self.array);
        }
    }
}

unsafe impl<T> Sync for Array<T> {}
unsafe impl<T> Send for Array<T> {}

/// Randomize an array of `f32`.
pub fn randomize_f32(array: &mut Array<f32>) {
    unsafe {
        randomize_float_array(array.context(), array.array, array.len as u64, 0.0, 1.0);
    }
}

/// Compares two arrays and returns maximum relative distance between two elements.
pub fn compare_f32(lhs: &Array<f32>, rhs: &Array<f32>) -> f32 {
    assert_eq!(lhs.len, rhs.len);
    let lhs_vec = lhs.copy_to_host();
    let rhs_vec = rhs.copy_to_host();
    lhs_vec
        .iter()
        .zip(&rhs_vec)
        .map(|(x, y)| 2.0 * (x - y) / (x.abs() + y.abs()))
        .fold(0.0, f32::max)
}

impl<T> Argument for Array<T> {
    fn raw_ptr(&self) -> *const libc::c_void {
        self.array as *const libc::c_void
    }
}

impl<T> device::ArrayArgument for Array<T>
where
    T: device::ScalarArgument,
{
    fn read_i8(&self) -> Vec<i8> {
        let mut array = Array::copy_to_host(self);
        let len = array.len() * std::mem::size_of::<T>();
        let capacity = array.capacity() * std::mem::size_of::<T>();
        unsafe {
            let bytes = array.as_mut_ptr() as *mut i8;
            let bytes_vec = Vec::from_raw_parts(bytes, len, capacity);
            std::mem::forget(array);
            bytes_vec
        }
    }

    fn write_i8(&self, bytes: &[i8]) {
        let (len, rem) = div_rem(bytes.len(), std::mem::size_of::<T>());
        assert_eq!(rem, 0);
        let ptr = bytes.as_ptr() as *const T;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        Array::copy_from_host(self, bytes);
    }
}
