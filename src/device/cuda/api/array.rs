//! Allows the execution of kernels on the GPU.
use device::Argument;
use device::cuda::api::wrapper::*;
use ir;
use libc;
use std;

/// An array allocated on a CUDA device.
pub struct Array<'a, T> {
    len: usize,
    array: *mut CudaArray,
    context: &'a CudaContext,
    t: std::marker::PhantomData<T>,
}

impl<'a, T> Array<'a, T> {
    /// Allocates a new array on the device.
    pub fn new(context: &'a CudaContext, len: usize) -> Self {
        let n_bytes = (len * std::mem::size_of::<T>()) as u64;
        Array {
            len, context,
            array:  unsafe { allocate_array(context, n_bytes) },
            t: std::marker::PhantomData
        }
    }

    /// Copies the array to the host.
    pub fn to_host(&self) -> Vec<T> {
        let mut vec = Vec::with_capacity(self.len);
        unsafe {
            vec.set_len(self.len);
            let host_ptr = vec.as_mut_ptr() as *mut libc::c_void;
            copy_DtoH(self.context, self.array, host_ptr, self.byte_len() as u64);
        }
        vec
    }

    /// Copies an array from the host.
    pub fn from_host(&self, vec: &[T]) {
        assert_eq!(self.len, vec.len());
        unsafe {
            let host_ptr = vec.as_ptr() as *const libc::c_void;
            copy_HtoD(self.context, host_ptr, self.array, self.byte_len() as u64);
        }
    }

    /// Returns the number of bytes in the array.
    fn byte_len(&self) -> usize { self.len * std::mem::size_of::<T>() }

    fn clone_from(&mut self, src: &Self) {
        assert_eq!(self.len, src.len);
        unsafe { copy_DtoD(self.context, src.array, self.array, self.byte_len() as u64); }
    }
}

impl<'a, T> Clone for Array<'a, T> {
    fn clone(&self) -> Self {
        let mut new_array = Self::new(self.context, self.len);
        new_array.clone_from(self);
        new_array
    }

    fn clone_from(&mut self, src: &Self) {
        assert_eq!(self.len, src.len);
        unsafe { copy_DtoD(self.context, src.array, self.array, self.byte_len() as u64); }
    }
}

impl<'a, T> Drop for Array<'a, T> {
    fn drop(&mut self) {
        unsafe { free_array(self.context, self.array); }
    }
}

unsafe impl<'a, T> Sync for Array<'a, T> {}
unsafe impl<'a, T> Send for Array<'a, T> {}

/// Randomize an array of `f32`.
pub fn randomize_f32(array: &Array<f32>) {
    unsafe {
        randomize_float_array(array.context, array.array, array.len as u64, 0.0, 1.0);
    }
}

/// Compares two arrays and returns maximum relative distance between two elements.
pub fn compare_f32(lhs: &Array<f32>, rhs: &Array<f32>) -> f32 {
    assert_eq!(lhs.len, rhs.len);
    let lhs_vec = lhs.to_host();
    let rhs_vec = rhs.to_host();
    lhs_vec.iter().zip(&rhs_vec).map(|(x, y)| {
        2.0*(x - y)/(x.abs() + y.abs())
    }).fold(0.0, f32::max)
}

pub struct ArrayArg<'a, T>(pub Array<'a, T>, pub ir::mem::Id);

impl<'a, T> Argument for ArrayArg<'a, T> {
    fn t(&self) -> ir::Type { ir::Type::PtrTo(self.1) }

    fn raw_ptr(&self) -> *const libc::c_void { self.0.array as *const libc::c_void }

    fn size_of(&self) -> usize { std::mem::size_of::<*mut libc::c_void>() }
}

impl<'a, 'b, T> Argument for &'b ArrayArg<'a, T> {
    fn t(&self) -> ir::Type { (**self).t() }

    fn raw_ptr(&self) -> *const libc::c_void { (**self).raw_ptr() }

    fn size_of(&self) -> usize { (**self).size_of() }
}
