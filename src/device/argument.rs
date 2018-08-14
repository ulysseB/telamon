//! Maps rust types to telamon data types.
use ir;
use libc;
use num::bigint::BigInt;
use num::integer::div_rem;
use num::rational::Ratio;
use num::traits::FromPrimitive;
use rand::Rng;
use std;
use std::fmt::Display;

/// Represents a value that can be used as a `Function` argument. Must ensures the type
/// is a scalar and does not contains any reference.
pub unsafe trait ScalarArgument: Sync + Send + Copy + PartialEq + Display + 'static {
    /// Returns the argument interpreted as an iteration dimension size, if applicable.
    fn as_size(&self) -> Option<u32> { None }
    /// Returns the type of the argument.
    fn t() -> ir::Type;
    /// Returns a raw pointer to the object.
    fn raw_ptr(&self) -> *const libc::c_void;
    /// Returns an operand holding the argument value as a constant.
    fn as_operand<L>(&self) -> ir::Operand<'static, L>;
    /// Generates a random instance of the argument type.
    fn gen_random<R: Rng>(&mut R) -> Self;
}

unsafe impl ScalarArgument for f32 {
    fn t() -> ir::Type { ir::Type::F(32) }

    fn raw_ptr(&self) -> *const libc::c_void {
        self as *const f32 as *const libc::c_void
    }

    fn as_operand<L>(&self) -> ir::Operand<'static, L> {
        ir::Operand::new_float(unwrap!(Ratio::from_float(*self)), 32)
    }

    fn gen_random<R: Rng>(rng: &mut R) -> Self { rng.gen_range(0., 1.) }
}

unsafe impl ScalarArgument for f64 {
    fn t() -> ir::Type { ir::Type::F(64) }

    fn raw_ptr(&self) -> *const libc::c_void {
        self as *const f64 as *const libc::c_void
    }

    fn as_operand<L>(&self) -> ir::Operand<'static, L> {
        ir::Operand::new_float(unwrap!(Ratio::from_float(*self)), 64)
    }

    fn gen_random<R: Rng>(rng: &mut R) -> Self { rng.gen_range(0., 1.) }
}

unsafe impl ScalarArgument for i8 {
    fn as_size(&self) -> Option<u32> { Some(*self as u32) }

    fn t() -> ir::Type { ir::Type::I(8) }

    fn raw_ptr(&self) -> *const libc::c_void {
        self as *const i8 as *const libc::c_void
    }

    fn as_operand<L>(&self) -> ir::Operand<'static, L> {
        ir::Operand::new_int(unwrap!(BigInt::from_i8(*self)), 8)
    }

    fn gen_random<R: Rng>(rng: &mut R) -> Self { rng.gen_range(0, 10) }
}

unsafe impl ScalarArgument for i16 {
    fn as_size(&self) -> Option<u32> { Some(*self as u32) }

    fn t() -> ir::Type { ir::Type::I(16) }

    fn raw_ptr(&self) -> *const libc::c_void {
        self as *const i16 as *const libc::c_void
    }

    fn as_operand<L>(&self) -> ir::Operand<'static, L> {
        ir::Operand::new_int(unwrap!(BigInt::from_i16(*self)), 16)
    }

    fn gen_random<R: Rng>(rng: &mut R) -> Self { rng.gen_range(0, 100) }
}

unsafe impl ScalarArgument for i32 {
    fn as_size(&self) -> Option<u32> { Some(*self as u32) }

    fn t() -> ir::Type { ir::Type::I(32) }

    fn raw_ptr(&self) -> *const libc::c_void {
        self as *const i32 as *const libc::c_void
    }

    fn as_operand<L>(&self) -> ir::Operand<'static, L> {
        ir::Operand::new_int(unwrap!(BigInt::from_i32(*self)), 32)
    }

    fn gen_random<R: Rng>(rng: &mut R) -> Self { rng.gen_range(0, 100) }
}

unsafe impl ScalarArgument for i64 {
    fn as_size(&self) -> Option<u32> { None }

    fn t() -> ir::Type { ir::Type::I(64) }

    fn raw_ptr(&self) -> *const libc::c_void {
        self as *const i64 as *const libc::c_void
    }

    fn as_operand<L>(&self) -> ir::Operand<'static, L> {
        ir::Operand::new_int(unwrap!(BigInt::from_i64(*self)), 64)
    }

    fn gen_random<R: Rng>(rng: &mut R) -> Self { rng.gen_range(0, 100) }
}

/// Represents an array on the device.
pub trait ArrayArgument: Send + Sync {
    // TODO(cc_perf): return a `Cow` instead of a `Vec` to avoid copying when testing
    // on a local CPU.
    // TODO(cleanup): use a type parameter instead of casting objects into bytes. For
    // this we first need rust/issues/#44265 to be solved.

    /// Copies the array to the host as a vector of bytes.
    fn read_i8(&self) -> Vec<i8>;

    /// Copies an array to the device from a slice of bytes.
    fn write_i8(&self, bytes: &[i8]);
}

/// Copies the array to the host, interpreting it as an array of `T`.
pub fn read_array<T: ScalarArgument>(array: &ArrayArgument) -> Vec<T> {
    let mut bytes_vec = array.read_i8();
    bytes_vec.shrink_to_fit();
    let (len, rem) = div_rem(bytes_vec.len(), std::mem::size_of::<T>());
    assert_eq!(rem, 0);
    unsafe {
        let bytes = bytes_vec.as_mut_ptr() as *mut T;
        std::mem::forget(bytes_vec);
        Vec::from_raw_parts(bytes, len, len)
    }
}

/// Copies an values to the device array from the host array given as argument.
pub fn write_array<T: ScalarArgument>(array: &ArrayArgument, from: &[T]) {
    let bytes_len = from.len()*std::mem::size_of::<T>();
    let bytes_ptr = from.as_ptr() as *const i8;
    let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr, bytes_len) };
    array.write_i8(bytes)
}
