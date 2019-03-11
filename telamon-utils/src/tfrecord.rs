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

use std::io::{self, Read, Write};
use std::{error, fmt};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc::crc32::checksum_castagnoli;

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

/// A tfrecord reader.
///
/// Inspired from the C++ implementation at: *
///  https://github.com/tensorflow/tensorflow/blob/f318765ad5a50b2fbd7cc08dd4ebc249b3924270/tensorflow/core/lib/io/record_reader.h
///  *
///  https://github.com/tensorflow/tensorflow/blob/f318765ad5a50b2fbd7cc08dd4ebc249b3924270/tensorflow/core/lib/io/record_reader.cc
#[derive(Debug)]
pub struct Reader<R> {
    // The underlying reader
    reader: R,
}

impl<R: Read> Reader<R> {
    /// Create a new TFRecord reader from the given reader.
    pub fn from_reader(reader: R) -> Reader<R> {
        Reader { reader }
    }

    /// Returns a reference to the underlying reader.
    pub fn get_ref(&self) -> &R {
        &self.reader
    }

    /// Returns a mutable reference to the underlying reader.
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    /// Unwraps the tfrecord reader, returning the underlying reader.
    ///
    /// Note that any leftover data in the reader's internal buffer is lost.
    pub fn into_inner(self) -> R {
        self.reader
    }

    /// Read a single record, appending the bytes to `buf`.  Returns `false` when no record could
    /// be read.
    ///
    /// This functions performs at most one reallocation of `buf` if it is not large enough to
    /// contain the read data.
    ///
    /// # Errors
    ///
    /// If this function encounters an error of the kind `ErrorKind::Interrupted` then the error is
    /// ignored and the operation will continue.
    ///
    /// If any other read error is encountered then this function immediately returns.  Any bytes
    /// which have already been read will be appended to `buf`.  If an error occurs while reading
    /// the initial tfrecord header, buf is unchanged.
    pub fn read_record(&mut self, buf: &mut Vec<u8>) -> io::Result<bool> {
        let len = {
            let mut len_bytes = [0u8; 8];
            if retry_read(&mut self.reader, &mut len_bytes)? == 0 {
                return Ok(false);
            }

            if self.reader.read_u32::<LittleEndian>()? != masked_crc32(&len_bytes) {
                return Err(io::Error::new(io::ErrorKind::Other, "corrupted record"));
            }

            // We `unwrap` here because reading from the on-stack
            // buffer cannnot fail.
            (&len_bytes[..]).read_u64::<LittleEndian>().unwrap()
        };

        // TODO(bclement): Consider adding a safety check that we are not allocating too large a
        // buffer here.
        let buf_start = buf.len();
        buf.reserve_exact(len as usize);
        let nread = (&mut self.reader).take(len).read_to_end(buf)?;
        if nread as u64 != len {
            return Err(io::Error::new(io::ErrorKind::Other, "truncated record"));
        }
        if self.reader.read_u32::<LittleEndian>()? != masked_crc32(&buf[buf_start..]) {
            return Err(io::Error::new(io::ErrorKind::Other, "corrupted record"));
        }

        Ok(true)
    }

    /// Transforms this `Read` instance to an `Iterator` over the contained records.
    ///
    /// The returned type implements `Iterator` where the `Item` is `io::Result<u8>`.  The yielded
    /// item is `Ok` if a record was successfuly read and `Err` otherwise.  EOF is mapped to
    /// returning `None` from this iterator.
    pub fn records(self) -> Records<R> {
        Records::new(self)
    }
}

/// A simple iterator over the records stored in a file.
#[derive(Debug)]
pub struct Records<R> {
    reader: Reader<R>,
}

impl<R: Read> Records<R> {
    fn new(reader: Reader<R>) -> Records<R> {
        Records { reader }
    }
}

impl<R: Read> Iterator for Records<R> {
    type Item = io::Result<Vec<u8>>;

    fn next(&mut self) -> Option<io::Result<Vec<u8>>> {
        let mut buffer = Vec::new();

        match self.reader.read_record(&mut buffer) {
            Err(err) => Some(Err(err)),
            Ok(true) => Some(Ok(buffer)),
            Ok(false) => None,
        }
    }
}

/// A tfrecord writer.
///
/// Inspired from the C++ implementation at: *
///  https://github.com/tensorflow/tensorflow/blob/f318765ad5a50b2fbd7cc08dd4ebc249b3924270/tensorflow/core/lib/io/record_writer.h
///  *
///  https://github.com/tensorflow/tensorflow/blob/f318765ad5a50b2fbd7cc08dd4ebc249b3924270/tensorflow/core/lib/io/record_writer.cc
#[derive(Debug)]
pub struct Writer<W> {
    writer: W,
}

/// An error returned by `into_inner` which combines an error that happened while writing out the
/// buffer, and the records writer object which may be used to recover from the condition.
#[derive(Debug)]
pub struct IntoInnerError<W>(W, io::Error);

impl<W: Write> Writer<W> {
    /// Creates a new tfrecord writer.
    pub fn from_writer(w: W) -> Writer<W> {
        Writer { writer: w }
    }

    /// Acquires a reference to the underlying writer.
    pub fn get_ref(&self) -> &W {
        &self.writer
    }

    /// Acquires a mutable reference to the underlying writer.
    ///
    /// Note that mutation of the writer may result in surprising results if the `Writer` is
    /// continued to be used.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.writer
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    /// Flush the contents of the internal buffer and return the underlying writer.
    pub fn into_inner(mut self) -> Result<W, IntoInnerError<Writer<W>>> {
        match self.flush() {
            Ok(()) => Ok(self.writer),
            Err(err) => Err(IntoInnerError(self, err)),
        }
    }

    /// Write data into a single record.
    pub fn write_record(&mut self, bytes: &[u8]) -> io::Result<()> {
        // We use a temporary buffer on the stack for the header
        // because we need to compute its crc32. We `unwrap` here
        // because writing to the on-stack buffer cannot fail.
        let mut len_bytes = [0u8; 8];
        len_bytes
            .as_mut()
            .write_u64::<LittleEndian>(bytes.len() as u64)
            .unwrap();

        self.writer.write_all(&len_bytes)?;
        self.writer
            .write_u32::<LittleEndian>(masked_crc32(&len_bytes))?;
        self.writer.write_all(bytes)?;
        self.writer.write_u32::<LittleEndian>(masked_crc32(bytes))?;
        Ok(())
    }
}

impl<W> IntoInnerError<W> {
    /// Returns the error which caused the call to `into_inner()` to fail.
    pub fn error(&self) -> &io::Error {
        &self.1
    }

    /// Returns the TFRecord writer instance which generated the error.
    pub fn into_inner(self) -> W {
        self.0
    }
}

impl<W> From<IntoInnerError<W>> for io::Error {
    fn from(iie: IntoInnerError<W>) -> io::Error {
        iie.1
    }
}

impl<W: Send + fmt::Debug> error::Error for IntoInnerError<W> {}

impl<W> fmt::Display for IntoInnerError<W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.error().fmt(f)
    }
}
