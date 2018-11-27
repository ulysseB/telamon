//! Maps rust types to telamon data types.
use ir;
use libc;
use num::integer::div_rem;
use num::rational::Ratio;
use rand::Rng;
use std;

/// Represents a value that can be used as a `Function` argument. Must ensures the type is a scalar
/// and does not contains any reference.  Also must ensure that no two implementers should have the
/// same `::t()` value, because that is used for downcasting.
pub unsafe trait ScalarArgument:
    std::fmt::Display + Send + Sync + 'static
{
    /// Returns the argument interpreted as an iteration dimension size, if applicable.
    fn as_size(&self) -> Option<u32> {
        None
    }

    /// Returns the type of the argument.
    fn t() -> ir::Type
    where
        Self: Sized;

    /// Returns the type of the argument.  This is a version of `::t()` that can be called on a
    /// trait object.
    fn get_type(&self) -> ir::Type;

    /// Returns a raw pointer to the object.
    fn raw_ptr(&self) -> *const libc::c_void;

    /// Returns an operand holding the argument value as a constant.
    fn as_operand<L>(&self) -> ir::Operand<'static, L>
    where
        Self: Sized;

    /// Generates a random instance of the argument type.
    fn gen_random<R: Rng>(&mut R) -> Self
    where
        Self: Sized;
}

// Returns the size of a type in bits.  Used for the `ScalarArgument` implementations below.
macro_rules! size_bits {
    ($ty:ty) => {
        8 * std::mem::size_of::<$ty>() as u16
    };
}

// `ScalarArgument` implementation for floating point types.  Requires that the type is a
// floating-point scalar type (see safety requirements for `ScalarArgument`).
macro_rules! float_scalar_argument {
    (unsafe impl ScalarArgument for $ty:ident [ ($start:expr) .. ($stop:expr) ]) => {
        unsafe impl ScalarArgument for $ty {
            fn t() -> ir::Type {
                ir::Type::F(size_bits!($ty))
            }

            fn get_type(&self) -> ir::Type {
                Self::t()
            }

            fn raw_ptr(&self) -> *const libc::c_void {
                self as *const $ty as *const libc::c_void
            }

            fn as_operand<L>(&self) -> ir::Operand<'static, L> {
                ir::Operand::new_float(unwrap!(Ratio::from_float(*self)), size_bits!($ty))
            }

            fn gen_random<R: Rng>(rng: &mut R) -> Self {
                rng.gen_range($start, $stop)
            }
        }
    };
}

float_scalar_argument!(unsafe impl ScalarArgument for f32 [(0.) .. (1.)]);
float_scalar_argument!(unsafe impl ScalarArgument for f64 [(0.) .. (1.)]);

// `ScalarArgument` implementation for integer types.  Requires that the type is an integer scalar
// type (see safety requirements for `ScalarArgument`).
macro_rules! int_scalar_argument {

    (unsafe impl ScalarArgument for $ty:ident [ ($start:expr) .. ($stop:expr) ]) => {
        unsafe impl ScalarArgument for $ty {
            fn as_size(&self) -> Option<u32> {
                if std::mem::size_of::<$ty>() <= std::mem::size_of::<u32>() {
                    Some(*self as u32)
                } else {
                    None
                }
            }

            fn t() -> ir::Type {
                ir::Type::I(size_bits!($ty))
            }

            fn get_type(&self) -> ir::Type {
                Self::t()
            }

            fn raw_ptr(&self) -> *const libc::c_void {
                self as *const $ty as *const libc::c_void
            }

            fn as_operand<L>(&self) -> ir::Operand<'static, L> {
                ir::Operand::new_int((*self).into(), size_bits!($ty))
            }

            fn gen_random<R: Rng>(rng: &mut R) -> Self {
                rng.gen_range($start, $stop)
            }
        }
    };
}

int_scalar_argument!(unsafe impl ScalarArgument for i8  [(0) .. ( 10)]);
int_scalar_argument!(unsafe impl ScalarArgument for i16 [(0) .. (100)]);
int_scalar_argument!(unsafe impl ScalarArgument for i32 [(0) .. (100)]);
int_scalar_argument!(unsafe impl ScalarArgument for i64 [(0) .. (100)]);

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

pub trait ArrayArgumentExt: ArrayArgument {
    /// Copies the array to the host, interpreting it as an array of `T`.
    fn read<T: ScalarArgument>(&self) -> Vec<T> {
        let mut bytes_vec = self.read_i8();
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
    fn write<T: ScalarArgument>(&self, from: &[T]) {
        let bytes_len = from.len() * std::mem::size_of::<T>();
        let bytes_ptr = from.as_ptr() as *const i8;
        let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr, bytes_len) };
        self.write_i8(bytes)
    }
}

impl<A: ?Sized> ArrayArgumentExt for A where A: ArrayArgument {}
