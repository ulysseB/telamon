//! This module provides Read and Write extensions for reading and
//! writing TFRecord files. TFRecord is a simple container file format
//! for embedding a sequence of data used notably by TensorFlow (see
//! https://www.tensorflow.org/api_guides/python/python_io).
//!
//! A TFRecords file contains a sequence of strings with CRC32C
//! checksums. Each record is constituded of a 12 bytes header
//! containing the length of the data with checksum, followed by a
//! `len` + 4 bytes payload containing the raw binary data and its
//! checksum. All integers are encoded in little-endian format.
//!
//! u64       len
//! u32       len_masked_crc32c
//! [u8; len] data
//! u32       data_masked_crc32c
//!
//! All records are concatenated together to create the final
//! file. The checksums are 32-bit CRC using the Castagnoli polynomial
//! masked as follow:
//!
//! masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8u32
//!
extern crate byteorder;
extern crate crc;

use self::byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use self::crc::crc32::checksum_castagnoli;
use std::io::{Read, Write};

/// The error type for errors occuring while reading a tfrecord file.
#[derive(Debug, Fail)]
pub enum ReadError {
    /// An I/O error occured.
    #[fail(display = "{}", _0)]
    IOError(#[cause] ::std::io::Error),
    /// The underlying data was shorter than advertised in the
    /// header's length field. If this happens because the end-of-file
    /// was reached, an I/O error will be raised instead.
    #[fail(display = "truncated record")]
    TruncatedRecord,
    /// Either the header or the data was corrupted and failed the CRC
    /// check.
    #[fail(display = "corrupted record")]
    CorruptedRecord,
}

/// For usage with ? when creating `ReadError`s.
impl From<::std::io::Error> for ReadError {
    #[inline]
    fn from(error: ::std::io::Error) -> ReadError {
        ReadError::IOError(error)
    }
}

/// The error type for errors occuring while writing a tfrecord file.
#[derive(Debug, Fail)]
pub enum WriteError {
    /// An I/O error occured.
    #[fail(display = "{}", _0)]
    IOError(#[cause] ::std::io::Error),
}

/// For usage with ? when creating `WriteError`s.
impl From<::std::io::Error> for WriteError {
    fn from(error: ::std::io::Error) -> WriteError {
        WriteError::IOError(error)
    }
}

/// Compute a masked CRC32. See module documentation for details.
fn masked_crc32(bytes: &[u8]) -> u32 {
    // https://www.tensorflow.org/api_guides/python/python_io
    let crc = checksum_castagnoli(bytes);
    ((crc >> 15) | (crc << 17)).wrapping_add(0xa282ead8u32)
}

/// A trait extension for reading records.
///
/// Inspired from the C++ implementation at: *
///  https://github.com/tensorflow/tensorflow/blob/f318765ad5a50b2fbd7cc08dd4ebc249b3924270/tensorflow/core/lib/io/record_reader.h
///  *
///  https://github.com/tensorflow/tensorflow/blob/f318765ad5a50b2fbd7cc08dd4ebc249b3924270/tensorflow/core/lib/io/record_reader.cc
pub trait RecordReader: Read {
    /// Read a single record.
    fn read_record(&mut self) -> Result<Vec<u8>, ReadError> {
        let len = {
            let mut len_bytes = [0u8; 8];
            self.read_exact(&mut len_bytes)?;
            if self.read_u32::<LittleEndian>()? != masked_crc32(&len_bytes) {
                return Err(ReadError::CorruptedRecord);
            }
            // We `unwrap` here because reading from the on-stack
            // buffer cannnot fail.
            len_bytes.as_ref().read_u64::<LittleEndian>().unwrap()
        };

        let mut record_bytes = Vec::with_capacity(len as usize);
        let nread = self.take(len).read_to_end(&mut record_bytes)? as u64;
        if nread != len {
            return Err(ReadError::TruncatedRecord);
        }
        if self.read_u32::<LittleEndian>()? != masked_crc32(&record_bytes) {
            return Err(ReadError::CorruptedRecord);
        }
        Ok(record_bytes)
    }
}

impl<R: Read + ?Sized> RecordReader for R {}

/// A trait extension for writing records.
///
/// Inspired from the C++ implementation at: *
///  https://github.com/tensorflow/tensorflow/blob/f318765ad5a50b2fbd7cc08dd4ebc249b3924270/tensorflow/core/lib/io/record_writer.h
///  *
///  https://github.com/tensorflow/tensorflow/blob/f318765ad5a50b2fbd7cc08dd4ebc249b3924270/tensorflow/core/lib/io/record_writer.cc
pub trait RecordWriter: Write {
    fn write_record(&mut self, bytes: &[u8]) -> Result<(), WriteError> {
        // We use a temporary buffer on the stack for the header
        // because we need to compute its crc32. We `unwrap` here
        // because writing to the on-stack buffer cannot fail.
        let mut len_bytes = [0u8; 8];
        len_bytes
            .as_mut()
            .write_u64::<LittleEndian>(bytes.len() as u64)
            .unwrap();

        self.write_all(&len_bytes)?;
        self.write_u32::<LittleEndian>(masked_crc32(&len_bytes))?;
        self.write_all(bytes)?;
        self.write_u32::<LittleEndian>(masked_crc32(bytes))?;
        Ok(())
    }
}

impl<W: Write + ?Sized> RecordWriter for W {}
