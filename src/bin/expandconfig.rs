extern crate serde_yaml;
extern crate toml;

extern crate structopt;

extern crate telamon;

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::io;
use std::io::prelude::*;

use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "expandconfig",
    about = "
Loads an explorer configuration file and outputs it with the default values added.  This can also
be used as a simple validator of an explorer configuration file by discarding the output.
",
    raw(setting = "structopt::clap::AppSettings::ColoredHelp")
)]
struct Opt {
    /// Path to the input file.  Use default config if not specified.
    #[structopt(parse(from_os_str))]
    input: Option<OsString>,

    /// Path to the output file.  Use stdout if not specified.
    #[structopt(
        short = "o",
        long = "output",
        default_value = "-",
        parse(from_os_str)
    )]
    output: OsString,

    /// Output format.
    #[structopt(
        short = "f",
        long = "format",
        default_value = "toml",
        possible_value = "toml",
        possible_value = "yaml"
    )]
    format: Format,
}

impl Opt {
    /// Open the input file and returns a `io::Read` handle to it
    fn open_input(&self) -> io::Result<Option<Box<dyn Read>>> {
        match self.input.as_ref().map(|input| try_parse_read(&*input)) {
            Some(Err(err)) => Err(err),
            Some(Ok(input)) => Ok(Some(input)),
            None => Ok(None),
        }
    }

    /// Create the output file and returns a `io::Write` handle to it
    fn create_output(&self) -> io::Result<Box<dyn Write>> {
        try_parse_write(&*self.output)
    }

    /// Parse the existing configuration
    fn parse_config(&self) -> io::Result<telamon::explorer::config::Config> {
        let input_str = match self.open_input()? {
            Some(mut input) => {
                let mut input_str = String::new();
                input.read_to_string(&mut input_str)?;
                input_str
            }
            None => "".to_string(),
        };

        Ok(toml::from_str(&input_str).map_err(|err| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("TOML deserialization error: {}", err),
            )
        })?)
    }

    /// Dump a configuration in the requested format
    fn dump_config(&self, config: &telamon::explorer::config::Config) -> io::Result<()> {
        let output_str = match self.format {
            Format::Toml => toml::to_string(config).map_err(|err| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("TOML serialization error: {}", err),
                )
            })?,
            Format::Yaml => serde_yaml::to_string(config).map_err(|err| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("YAML serialization error: {}", err),
                )
            })?,
        };

        write!(self.create_output()?, "{}", output_str);

        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Format {
    Toml,
    Yaml,
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Format::*;

        match self {
            Toml => write!(f, "toml"),
            Yaml => write!(f, "yaml"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseFormatError {
    _priv: (),
}

impl fmt::Display for ParseFormatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "valid formats are `toml` and `yaml`".fmt(f)
    }
}

impl std::str::FromStr for Format {
    type Err = ParseFormatError;

    fn from_str(s: &str) -> Result<Format, Self::Err> {
        use Format::*;

        match &*s.to_lowercase() {
            "toml" => Ok(Toml),
            "yaml" => Ok(Yaml),
            _ => Err(ParseFormatError { _priv: () }),
        }
    }
}

/// Parse a `io::Read` instance from a path.  This
fn try_parse_read(source: &OsStr) -> io::Result<Box<dyn Read>> {
    match source.to_str() {
        Some("-") => Ok(Box::new(io::stdin())),
        _ => Ok(Box::new(fs::File::open(source)?)),
    }
}

/// Parse a `io::Write` instance from a path.
fn try_parse_write(source: &OsStr) -> io::Result<Box<dyn Write>> {
    match source.to_str() {
        Some("-") => Ok(Box::new(io::stdout())),
        _ => Ok(Box::new(fs::File::create(source)?)),
    }
}

fn main() {
    let opt = Opt::from_args();

    let config = opt
        .parse_config()
        .expect("Unable to parse configuration file");

    opt.dump_config(&config)
        .expect("Unable to dump configuration file");
}
