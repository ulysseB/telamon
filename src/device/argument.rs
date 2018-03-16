//! Maps rust types to telamon data types.
use ir::Type;
use libc;
use std;

/// Represents a value that can be used as a `Function` argument.
pub trait Argument: Sync + Send {
    /// Returns the argument interpreted as an iteration dimension size, if applicable.
    fn as_size(&self) -> Option<u32> { None }
    /// Returns the type of the argument.
    fn t(&self) -> Type;
    /// Returns a raw pointer to the object.
    fn raw_ptr(&self) -> *const libc::c_void;
    /// Returns the size of the argument.
    fn size_of(&self) -> usize;
}

impl Argument for f32 {
    fn t(&self) -> Type { Type::F(32) }

    fn raw_ptr(&self) -> *const libc::c_void {
        self as *const f32 as *const libc::c_void
    }

    fn size_of(&self) -> usize { std::mem::size_of::<Self>() }
}

impl Argument for f64 {
    fn t(&self) -> Type { Type::F(64) }

    fn raw_ptr(&self) -> *const libc::c_void {
        self as *const f64 as *const libc::c_void
    }

    fn size_of(&self) -> usize { std::mem::size_of::<Self>() }
}

impl Argument for i8 {
    fn as_size(&self) -> Option<u32> { Some(*self as u32) }

    fn t(&self) -> Type { Type::I(8) }

    fn raw_ptr(&self) -> *const libc::c_void {
        self as *const i8 as *const libc::c_void
    }

    fn size_of(&self) -> usize { std::mem::size_of::<Self>() }
}

impl Argument for i16 {
    fn as_size(&self) -> Option<u32> { Some(*self as u32) }

    fn t(&self) -> Type { Type::I(16) }

    fn raw_ptr(&self) -> *const libc::c_void {
        self as *const i16 as *const libc::c_void
    }

    fn size_of(&self) -> usize { std::mem::size_of::<Self>() }
}

impl Argument for i32 {
    fn as_size(&self) -> Option<u32> { Some(*self as u32) }

    fn t(&self) -> Type { Type::I(32) }

    fn raw_ptr(&self) -> *const libc::c_void {
        self as *const i32 as *const libc::c_void
    }

    fn size_of(&self) -> usize { std::mem::size_of::<Self>() }
}

impl Argument for i64 {
    fn as_size(&self) -> Option<u32> { None }

    fn t(&self) -> Type { Type::I(64) }

    fn raw_ptr(&self) -> *const libc::c_void {
        self as *const i64 as *const libc::c_void
    }

    fn size_of(&self) -> usize { std::mem::size_of::<Self>() }
}
