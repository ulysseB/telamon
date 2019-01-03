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

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc::crc32::checksum_castagnoli;
use failure::Fail;
use flate2::write::{GzEncoder, ZlibEncoder};

/// The error type for errors occuring while reading a tfrecord file.
#[derive(Debug, Fail)]
pub enum ReadError {
    /// An I/O error occured.
    #[fail(display = "{}", _0)]
    IOError(#[cause] io::Error),
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
impl From<io::Error> for ReadError {
    #[inline]
    fn from(error: io::Error) -> ReadError {
        ReadError::IOError(error)
    }
}

/// The error type for errors occuring while writing a tfrecord file.
#[derive(Debug, Fail)]
pub enum WriteError {
    /// An I/O error occured.
    #[fail(display = "{}", _0)]
    IOError(#[cause] io::Error),
}

/// For usage with ? when creating `WriteError`s.
impl From<io::Error> for WriteError {
    fn from(error: io::Error) -> WriteError {
        WriteError::IOError(error)
    }
}

/// For usage with ? when creating `WriteError`s
impl<W> From<io::IntoInnerError<W>> for WriteError {
    fn from(error: io::IntoInnerError<W>) -> WriteError {
        WriteError::IOError(error.into())
    }
}

/// Compute a masked CRC32. See module documentation for details.
fn masked_crc32(bytes: &[u8]) -> u32 {
    // https://www.tensorflow.org/api_guides/python/python_io
    let crc = checksum_castagnoli(bytes);
    ((crc >> 15) | (crc << 17)).wrapping_add(0xa282_ead8u32)
}

// Wrapper around Read::read which retries when receiving an Interrupted error
fn retry_read<R: Read + ?Sized>(read: &mut R, mut buf: &mut [u8]) -> io::Result<usize> {
    let mut nread = 0;
    while !buf.is_empty() {
        match read.read(buf) {
            Ok(0) => break,
            Ok(n) => {
                let tmp = buf;
                buf = &mut tmp[n..];
                nread += n;
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }

    Ok(nread)
}

/// A trait extension for reading records.
///
/// Inspired from the C++ implementation at: *
///  https://github.com/tensorflow/tensorflow/blob/f318765ad5a50b2fbd7cc08dd4ebc249b3924270/tensorflow/core/lib/io/record_reader.h
///  *
///  https://github.com/tensorflow/tensorflow/blob/f318765ad5a50b2fbd7cc08dd4ebc249b3924270/tensorflow/core/lib/io/record_reader.cc
pub trait RecordReader: Read {
    /// Read a single record, placing the bytes into `buf`.
    ///
    /// All bytes read from this source will be appended to the specified buffer `buf`.
    ///
    /// If successful, this function returns the total number of bytes read, including the tfrecord
    /// header and footer sizes.  If the total number of bytes read is `0`, then the reader has
    /// reached end of file.
    ///
    /// # Errors
    ///
    /// If this function encounters an error of the kind `ErrorKind::Interrupted` then the error is
    /// ignored and the operation will continue.
    ///
    /// If any other read error is encountered then this function immediately returns.  Any bytes
    /// which have already been read will be appended to `buf`.  If an error occurs while reading
    /// the initial tfrecord header, buf is unchanged.
    fn read_record(&mut self, buf: &mut Vec<u8>) -> Result<usize, ReadError> {
        let len = {
            let mut len_bytes = [0u8; 8];
            if retry_read(self, &mut len_bytes)? == 0 {
                return Ok(0);
            }

            if self.read_u32::<LittleEndian>()? != masked_crc32(&len_bytes) {
                return Err(ReadError::CorruptedRecord);
            }

            // We `unwrap` here because reading from the on-stack
            // buffer cannnot fail.
            (&len_bytes[..]).read_u64::<LittleEndian>().unwrap()
        };

        // TODO(bclement): Consider adding a safety check that we are not allocating too large a
        // buffer here.
        let buf_start = buf.len();
        buf.reserve_exact(len as usize);
        let nread = self.take(len).read_to_end(buf)?;
        if nread as u64 != len {
            return Err(ReadError::TruncatedRecord);
        }
        if self.read_u32::<LittleEndian>()? != masked_crc32(&buf[buf_start..]) {
            return Err(ReadError::CorruptedRecord);
        }

        Ok(8 + 4 + 4 + nread)
    }

    /// Transforms this `Read` instance to an `Iterator` over the contained records.
    ///
    /// The returned type implements `Iterator` where the `Item` is `Result<u8, ReadError>`.  The
    /// yielded item is `Ok` if a record was successfuly read and `Err` otherwise.  EOF is mapped
    /// to returning `None` from this iterator.
    fn records(self) -> Records<Self>
    where
        Self: Sized,
    {
        Records { read: self }
    }
}

impl<R: Read + ?Sized> RecordReader for R {}

/// A simple iterator over the records stored in a file.
#[derive(Debug)]
pub struct Records<R> {
    read: R,
}

impl<R: Read> Iterator for Records<R> {
    type Item = Result<Vec<u8>, ReadError>;

    fn next(&mut self) -> Option<Result<Vec<u8>, ReadError>> {
        let mut buf = Vec::new();
        match self.read.read_record(&mut buf) {
            Ok(0) => None,
            Ok(..) => Some(Ok(buf)),
            Err(e) => Some(Err(e)),
        }
    }
}

/// A trait extension for writing records.
///
/// Inspired from the C++ implementation at: *
///  https://github.com/tensorflow/tensorflow/blob/f318765ad5a50b2fbd7cc08dd4ebc249b3924270/tensorflow/core/lib/io/record_writer.h
///  *
///  https://github.com/tensorflow/tensorflow/blob/f318765ad5a50b2fbd7cc08dd4ebc249b3924270/tensorflow/core/lib/io/record_writer.cc
pub trait RecordWriter: Write {
    type Writer;

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

    /// Writes all output to the file and conclude the stream. Returns
    /// the underlying writer. *Does not* flush the underlying writer.
    ///
    /// This is the conceptual equivalent of RecordWriter::Close in
    /// TensorFlow's C++ implementation.
    fn finish(self) -> Result<Self::Writer, WriteError>;

    /// Equivalent to `finish` for boxed trait objects, because we
    /// otherwise can't move out of the unsized trait object.
    fn finish_box(self: Box<Self>) -> Result<Self::Writer, WriteError>;
}

impl RecordWriter for File {
    type Writer = File;

    fn finish(self) -> Result<File, WriteError> {
        Ok(self)
    }

    fn finish_box(self: Box<Self>) -> Result<File, WriteError> {
        RecordWriter::finish(*self)
    }
}

impl<W: Write> RecordWriter for BufWriter<W> {
    type Writer = W;

    fn finish(self) -> Result<W, WriteError> {
        Ok(BufWriter::into_inner(self)?)
    }

    fn finish_box(self: Box<Self>) -> Result<W, WriteError> {
        RecordWriter::finish(*self)
    }
}

impl<W: Write> RecordWriter for GzEncoder<W> {
    type Writer = W;

    fn finish(self) -> Result<W, WriteError> {
        Ok(GzEncoder::finish(self)?)
    }

    fn finish_box(self: Box<Self>) -> Result<W, WriteError> {
        RecordWriter::finish(*self)
    }
}

impl<W: Write> RecordWriter for ZlibEncoder<W> {
    type Writer = W;

    fn finish(self) -> Result<W, WriteError> {
        Ok(ZlibEncoder::finish(self)?)
    }

    fn finish_box(self: Box<Self>) -> Result<W, WriteError> {
        RecordWriter::finish(*self)
    }
}
