use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use flate2::{read, write, Compression};
use utils::tfrecord;

#[allow(clippy::large_enum_variant)]
enum EventLogInner {
    Raw(File),
    Gz(read::GzDecoder<write::GzEncoder<File>>),
    Zlib(read::ZlibDecoder<write::ZlibEncoder<File>>),
}

pub struct EventLog {
    inner: EventLogInner,
}

impl EventLog {
    fn wrap(file: File, extension: Option<&str>) -> Self {
        let inner = match extension {
            Some("gz") => EventLogInner::Gz(read::GzDecoder::new(write::GzEncoder::new(
                file,
                Compression::default(),
            ))),
            Some("zz") => EventLogInner::Zlib(read::ZlibDecoder::new(
                write::ZlibEncoder::new(file, Compression::default()),
            )),
            _ => EventLogInner::Raw(file),
        };

        EventLog { inner }
    }

    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<tfrecord::Reader<Self>> {
        let extension = path
            .as_ref()
            .extension()
            .and_then(OsStr::to_str)
            .map(str::to_string);
        let file = File::open(path)?;
        Ok(tfrecord::Reader::from_reader(Self::wrap(
            file,
            extension.as_ref().map(String::as_ref),
        )))
    }

    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<tfrecord::Writer<Self>> {
        let extension = path
            .as_ref()
            .extension()
            .and_then(OsStr::to_str)
            .map(str::to_string);
        let file = File::create(path)?;
        Ok(tfrecord::Writer::from_writer(Self::wrap(
            file,
            extension.as_ref().map(String::as_ref),
        )))
    }

    pub fn finish(self) -> io::Result<File> {
        match self.inner {
            EventLogInner::Raw(file) => Ok(file),
            EventLogInner::Gz(file) => file.into_inner().finish(),
            EventLogInner::Zlib(file) => file.into_inner().finish(),
        }
    }
}

impl Read for EventLog {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match &mut self.inner {
            EventLogInner::Raw(file) => file.read(buf),
            EventLogInner::Gz(file) => file.read(buf),
            EventLogInner::Zlib(file) => file.read(buf),
        }
    }
}

impl Write for EventLog {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match &mut self.inner {
            EventLogInner::Raw(file) => file.write(buf),
            EventLogInner::Gz(file) => file.write(buf),
            EventLogInner::Zlib(file) => file.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match &mut self.inner {
            EventLogInner::Raw(file) => file.flush(),
            EventLogInner::Gz(file) => file.flush(),
            EventLogInner::Zlib(file) => file.flush(),
        }
    }
}
