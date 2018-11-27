extern crate telamon;

extern crate env_logger;
#[macro_use]
extern crate log;
extern crate num_cpus;
extern crate serde;
extern crate structopt;
#[macro_use]
extern crate serde_derive;
extern crate serde_yaml;
extern crate toml;

use telamon::explorer::config::*;

use std::{
    error, fmt, fs, io,
    path::{Path, PathBuf},
};

use structopt::StructOpt;

#[derive(StructOpt, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    #[structopt(name = "bench")]
    Bench(bench::Opt),
}

mod bench {
    use std::io::prelude::*;
    use std::path::Path;

    use structopt::StructOpt;

    #[derive(StructOpt, Serialize, Deserialize)]
    pub struct Opt {
        // TODO(bclement): Migrate to f64 and Duration::from_float_secs once that is stabilizes.
        // See https://github.com/rust-lang/rust/issues/54361
        #[structopt(
            name = "timeout",
            help = "Timeout for the search (in seconds)"
        )]
        pub timeout: Option<u64>,

        // TODO(bclement): This should be u32 (or u64), but it should not depend on the machine.
        #[structopt(
            name = "num_evaluations",
            help = "Limit the number of evaluations to perform"
        )]
        pub num_evaluations: Option<usize>,
    }

    pub fn exec(out_dir: &Path, config: &super::Opt) -> Result<(), super::Error> {
        let settings = config.to_explorer_config()?;

        write!(
            std::fs::File::create(out_dir.join("Settings.toml"))?,
            "{}",
            &toml::to_string(&settings)?
        )?;

        Ok(())
    }

}

#[derive(StructOpt, Serialize, Deserialize)]
#[structopt(name = "telamon")]
pub struct Opt {
    #[structopt(long = "output", short = "o", help = "Output directory")]
    pub output: String,

    #[structopt(long = "dummy", help = "Mark the experiment as dummy")]
    pub dummy: bool,

    #[structopt(subcommand)]
    pub cmd: Command,
}

#[derive(Debug)]
pub enum Error {
    IoError(io::Error),
    YamlError(serde_yaml::Error),
    TomlSerError(toml::ser::Error),
    UnsafeOverwrite(PathBuf),
    Custom(String),
}

impl Error {
    fn unwrap<T>(result: Result<T, Error>) -> T {
        match result {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{}", err);
                std::process::exit(1);
            }
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::IoError(err) => write!(f, "I/O error: {}", err),
            Error::YamlError(err) => write!(f, "YAML error: {}", err),
            Error::TomlSerError(err) => write!(f, "TOML serialization error: {}", err),
            Error::UnsafeOverwrite(path) => write!(
                f,
                "The output directory `{}` already exists and may contain valuable data.",
                path.display()
            ),
            Error::Custom(err) => write!(f, "{}", err),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Error::IoError(err) => Some(err),
            Error::YamlError(err) => Some(err),
            Error::TomlSerError(err) => Some(err),
            Error::UnsafeOverwrite(_) => None,
            Error::Custom(_) => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::IoError(err)
    }
}

impl From<serde_yaml::Error> for Error {
    fn from(err: serde_yaml::Error) -> Self {
        Error::YamlError(err)
    }
}

impl From<toml::ser::Error> for Error {
    fn from(err: toml::ser::Error) -> Self {
        Error::TomlSerError(err)
    }
}

impl Opt {
    fn output_directory(&self) -> &Path {
        Path::new(&self.output)
    }

    fn create_output_directory(&self) -> Result<&Path, Error> {
        let out_dir = self.output_directory();

        // Dummy safeguard
        let dummy_path = out_dir.join("DUMMY");

        // Ensure parent directory exists
        if let Some(parent) = out_dir.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::create_dir(&out_dir).or_else(|err| {
            if err.kind() == io::ErrorKind::AlreadyExists {
                if self.dummy && dummy_path.exists() {
                    debug!("Overwriting existing dummy directory.");

                    fs::remove_dir_all(&out_dir)?;
                    fs::create_dir(&out_dir)?;

                    Ok(())
                } else {
                    Err(Error::UnsafeOverwrite(out_dir.into()))
                }
            } else {
                Err(err.into())
            }
        })?;

        if self.dummy {
            fs::write(&dummy_path, b"")?;
        }

        // Write the configuration to the directory
        #[derive(Serialize)]
        struct YamlConfig<'a> {
            params: &'a Opt,
        }

        serde_yaml::to_writer(
            &mut std::fs::File::create(out_dir.join("config.yaml"))?,
            &YamlConfig { params: &self },
        )?;

        Ok(out_dir)
    }

    fn to_explorer_config(&self) -> Result<Config, Error> {
        let out_dir = self.output_directory();

        Ok(Config {
            log_file: out_dir
                .join("watch.log")
                .to_str()
                .ok_or_else(|| {
                    Error::Custom("Invalid UTF-8 in output path.".to_string())
                })?.to_string(),

            event_log: out_dir
                .join("eventlog.tfrecord.gz")
                .to_str()
                .ok_or_else(|| {
                    Error::Custom("Invalid UTF-8 in output path.".to_string())
                })?.to_string(),

            num_workers: num_cpus::get(),

            stop_bound: None,

            timeout: match self.cmd {
                Command::Bench(ref bench) => bench.timeout,
            },

            max_evaluations: match self.cmd {
                Command::Bench(ref bench) => bench.num_evaluations,
            },

            distance_to_best: None,

            algorithm: SearchAlgorithm::MultiArmedBandit(BanditConfig {
                new_nodes_order: NewNodeOrder::WeightedRandom,
                old_nodes_order: OldNodeOrder::Bandit,
                threshold: 10,
                delta: 0.1,
                monte_carlo: true,
                choice_ordering: ChoiceOrdering::default(),
            }),
        })
    }
}

fn main() {
    env_logger::init();

    let opt = Opt::from_args();
    let out_dir = Error::unwrap(opt.create_output_directory());

    match opt.cmd {
        Command::Bench(_) => Error::unwrap(bench::exec(out_dir, &opt)),
    }
}
