
//! Allows the execution of kernels on the GPU.
use device::Argument;
use ir;
use libc;
use std;

pub struct ArrayArg<T> where T:Argument {
    id: ir::mem::Id,
    arr: Vec<T>,
}

impl<T> ArrayArg<T> where T: Argument {
    pub fn new(id: ir::mem::Id, arr: Vec<T>) -> Self {
        ArrayArg { id, arr}
    }
}


impl<T> Argument for ArrayArg<T> where T: Argument {
    fn t(&self) -> ir::Type { ir::Type::PtrTo(self.id) }

    fn raw_ptr(&self) -> *const libc::c_void { self.arr.as_ptr() as *const libc::c_void }

    fn size_of(&self) -> usize { self.arr.len() }
}

impl<'a, T> Argument for &'a ArrayArg<T> where T:Argument {
    fn t(&self) -> ir::Type { (**self).t() }

    fn raw_ptr(&self) -> *const libc::c_void { (**self).raw_ptr() }

    fn size_of(&self) -> usize { (**self).size_of() }
}
