//! This module provides the `WriteArrayExt` extension trait which allows to serialize arrays from
//!  the `ndarray` crate using the NumPy `.npy` file format.
//!
//!  The format 1.0 as described in the NumPy documentation[1] is supported.
//!
//!  1: https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html

use std::borrow::Cow;
use std::io::{self, Write};

use byteorder::{LittleEndian, WriteBytesExt};
use ndarray::{ArrayBase, Data, Dimension};

pub trait Value {
    /// The serialized Python value to pass to `np.dtype`.  For portability, this SHOULD specify an
    /// alignment when applicable.
    fn descr() -> Cow<'static, str>;

    /// Write the value into a specific writer.  This must write values in the appropriate
    /// serialization format as described in the `descr()` method.
    fn write_to<W: io::Write + ?Sized>(&self, writer: &mut W) -> io::Result<()>;
}

macro_rules! impl_value {
    () => {};
    ($ty:ty , $write:expr , $descr:expr ; $($rest:tt)*) => {
        impl Value for $ty {
            fn descr() -> Cow<'static, str> {
                $descr.into()
            }

            fn write_to<W: io::Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
                $write(writer, *self)
            }
        }

        impl_value!($($rest)*);
    };
}

impl_value! {
    i8 , WriteBytesExt::write_i8                 , "'|i1'";
    i16, WriteBytesExt::write_i16::<LittleEndian>, "'<i2'";
    i32, WriteBytesExt::write_i32::<LittleEndian>, "'<i4'";
    i64, WriteBytesExt::write_i64::<LittleEndian>, "'<i8'";
    u8 , WriteBytesExt::write_u8                 , "'|u1'";
    u16, WriteBytesExt::write_u16::<LittleEndian>, "'<u2'";
    u32, WriteBytesExt::write_u32::<LittleEndian>, "'<u4'";
    u64, WriteBytesExt::write_u64::<LittleEndian>, "'<u8'";
    f32, WriteBytesExt::write_f32::<LittleEndian>, "'<f4'";
    f64, WriteBytesExt::write_f64::<LittleEndian>, "'<f8'";
}

pub trait WriteArrayExt: io::Write {
    /// A helper function to write an array in NumPy format.
    fn write_array<A, D, S>(&mut self, array: &ArrayBase<S, D>) -> io::Result<()>
    where
        A: Value,
        S: Data<Elem = A>,
        D: Dimension,
    {
        let mut header = Vec::new();

        // Write the header data describing the data format.
        {
            let f = &mut header;

            f.push(b'{');

            // Ensure keys are written in alpha order: descr, fortran_order, shape
            write!(f, "'descr': {}, ", A::descr())?;

            // We always write in C order
            // TODO: figure out a way to cleanly and reliably determine if the array is fortran
            f.write_all(b"'fortran_order': False, ")?;

            f.write_all(b"'shape': (")?;
            if let Some((first, shape)) = array.shape().split_first() {
                write!(f, "{}", first)?;
                for ix in shape {
                    write!(f, ", {}", ix)?;
                }
                // Single-element tuples need their trailing commas
                if shape.is_empty() {
                    f.push(b',');
                }
            }
            f.write_all(b"), ")?;

            f.push(b'}');

            let magic_bytes = 6;
            let format_bytes = 2;
            // +1 for the mandatory final \n
            let header_bytes = f.len() + 1;
            let length_bytes = 2;
            let offset = (magic_bytes + format_bytes + length_bytes + header_bytes) % 64;
            if offset > 0 {
                for _ in 0..(64 - offset) {
                    f.push(b' ');
                }
            }
            f.push(b'\n');
        }

        // We don't support writing in v2 format.  According to the NumPy docs, this should only
        // happen with structured arrays with a large number of fields... but we don't support
        // structured arrays anyways.  If support is added in the future, don't forget to adapt
        // `length_bytes` in the offset computation as well!
        if header.len() > u16::max_value().into() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "NumPy format 2.0 is required but not supported",
            ));
        }

        let header_len_u16 = header.len() as u16;

        self.write_all(b"\x93NUMPY")?;
        self.write_all(&[1, 0])?;
        self.write_u16::<LittleEndian>(header_len_u16)?;
        self.write_all(&header)?;

        for elem in array {
            elem.write_to(self)?;
        }

        Ok(())
    }
}

/// All types that implement `Write` get methods defined in `WriteArrayExt` for free.
impl<W: io::Write + ?Sized> WriteArrayExt for W {}

#[cfg(test)]
pub mod tests {
    use ndarray::array;
    use ndarray::prelude::*;

    use super::WriteArrayExt;

    macro_rules! npy_equiv {
        () => {};
        ([$($name:ident : $t:ty),*] $python:literal => $rust:expr , $($rest:tt)*) => {
            $(
                #[test]
                fn $name() {
                    let array: Array<$t, _> = $rust;

                    let mut buf = Vec::new();
                    (&mut buf).write_array(&array).unwrap();

                    assert_eq!(&buf[..], &include_bytes!(concat!("tests/data/", stringify!($name), ".npy"))[..]);
                }
            )*

            npy_equiv!($($rest)*);
        };
    }

    npy_equiv! {
        // Lines between the `NPY_START` and `NPY_END` lines are parsed by the Python test
        // generation script and *MUST* fit on one line each.  Some of these are taken from the
        // examples in the "ndarray for numpy users" doc.
        // @NPY_START
        [literal_2x3_f32:f32, literal_2x3_f64:f64] "np.array([[1., 2., 3.], [4., 5., 6.]], dtype=?)"
            => array![[1., 2., 3.], [4., 5., 6.]],
        [range_f32:f32, range_f64:f64] "np.arange(0., 10., 0.5, dtype=?)"
            => Array::range(0., 10., 0.5),
        [linspace_f32:f32, linspace_f64:f64] "np.linspace(0., 10., 11, dtype=?)"
            => Array::linspace(0., 10., 11),
        [ones_i8:i8, ones_i16:i16, ones_i32:i32, ones_i64:i64, ones_u8:u8, ones_u16:u16, ones_u32:u32, ones_u64:u64, ones_f32:f32, ones_f64:f64] "np.ones((3, 4, 5), dtype=?)"
            => Array::ones((3, 4, 5)),
        [ones_f_i8:i8, ones_f_i16:i16, ones_f_i32:i32, ones_f_i64:i64, ones_f_u8:u8, ones_f_u16:u16, ones_f_u32:u32, ones_f_u64:u64, ones_f_f32:f32, ones_f_f64:f64] "np.ones((3, 4, 5), dtype=?)"
            => Array::ones((3, 4, 5).f()),
        [zeros_i8:i8, zeros_i16:i16, zeros_i32:i32, zeros_i64:i64, zeros_u8:u8, zeros_u16:u16, zeros_u32:u32, zeros_u64:u64, zeros_f32:f32, zeros_f64:f64] "np.zeros((3, 4, 5), dtype=?)"
            => Array::zeros((3, 4, 5)),
        [zeros_f_i8:i8, zeros_f_i16:i16, zeros_f_i32:i32, zeros_f_i64:i64, zeros_f_u8:u8, zeros_f_u16:u16, zeros_f_u32:u32, zeros_f_u64:u64, zeros_f_f32:f32, zeros_f_f64:f64] "np.zeros((3, 4, 5), dtype=?)"
            => Array::zeros((3, 4, 5).f()),
        [eye_i8:i8, eye_i16:i16, eye_i32:i32, eye_i64:i64, eye_u8:u8, eye_u16:u16, eye_u32:u32, eye_u64:u64, eye_f32:f32, eye_f64:f64] "np.eye(3, dtype=?)"
            => Array::eye(3),
        // @NPY_END
    }
}
