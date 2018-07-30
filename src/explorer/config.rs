//! Defines a structure to store the configuration of the exploration. The configuration
//! is read from the `Setting.toml` file if it exists. Some parameters can be overridden
//! from the command line.

extern crate toml;

use config;
use std;
use std::fmt;
use getopts;
use itertools::Itertools;
use num_cpus;

/// Stores the configuration of the exploration.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Name of the file in wich to store the logs.
    pub log_file: String,
    /// Name of the file in which to store the binary event log.
    pub event_log: String,
    /// Number of exploration threads.
    pub num_workers: usize,
    /// Indicates the search must be stopped if a candidate with an execution time better
    /// than the bound (in ns) is found.
    pub stop_bound: Option<f64>,
    /// Indicates the search must be stopped after the given number of minutes.
    pub timeout: Option<u64>,
    /// Indicates the search must be stopped after the given number of
    /// candidates have been evaluated.
    pub max_evaluations: Option<usize>,
    /// A percentage cut indicate that we only care to find a candidate that is in a
    /// certain range above the best Therefore, if cut_under is 20%, we can discard any
    /// candidate whose bound is above 80% of the current best.
    pub distance_to_best: Option<f64>,
    /// Exploration algorithm to use. Needs to be last for TOML serialization, because it is a table.
    pub algorithm: SearchAlgorithm,
}

impl Config {
    fn create_parser() -> config::Config {
        let mut config_parser = config::Config::new();
        // If there is nothing in the config, the parser fails by
        // saying that it found a unit value where it expected a
        // Config (see
        // https://github.com/mehcode/config-rs/issues/60). As a
        // workaround, we set an explicit default for the "timeout"
        // option, which makes the parsing succeed even if there is
        // nothing to parse.
        unwrap!(config_parser.set_default::<Option<f64>>("timeout", None));
        let config_path = std::path::Path::new("Settings.toml");
        if config_path.exists() {
            unwrap!(config_parser.merge(config::File::from(config_path)));
        }
        config_parser
    }

    /// Reads the configuration from the "Settings.toml" file and from the command line.
    pub fn read() -> Self {
        let arg_parser = Self::setup_args_parser();
        let args = std::env::args().collect_vec();
        let arg_matches = arg_parser.parse(&args[1..]).unwrap_or_else(|err| {
            println!("{} Use '--help' to display a list of valid options.", err);
            std::process::exit(-1);
        });
        if arg_matches.opt_present("h") {
            let brief = arg_parser.short_usage(&args[0]);
            println!("{}", arg_parser.usage(&brief));
            std::process::exit(0);
        }
        let mut config_parser = Self::create_parser();
        Self::parse_arguments(&arg_matches, &mut config_parser);
        unwrap!(config_parser.try_into::<Self>())
    }

    /// Extract the configuration from the configuration file, if any.
    pub fn read_from_file() -> Self {
        unwrap!(Self::create_parser().try_into::<Self>())
    }

    /// Parse the configuration from a JSON string. Primary user is
    /// the Python API (through the C API).
    pub fn from_json(json: &str) -> Self {
        let mut parser = Self::create_parser();
        unwrap!(parser.merge(
            config::File::from_str(json, config::FileFormat::Json)));
        unwrap!(parser.try_into::<Self>())
    }

    /// Sets up the parser of command line arguments.
    fn setup_args_parser() -> getopts::Options {
        let mut opts = getopts::Options::new();
        opts.optflag("h", "help", "Print the help menu.");
        opts.optopt("j", "jobs", "number of explorer working in parallel", "N_THREAD");
        opts.optopt("f", "log_file", "name of watcher file", "string");
        SearchAlgorithm::setup_args_parser(&mut opts);
        opts
    }

    /// Overwrite the configuration with the parameters from the command line.
    fn parse_arguments(arguments: &getopts::Matches, config: &mut config::Config) {
        if let Some(num_workers) = arguments.opt_str("j") {
            let num_workers: i64 = num_workers.parse().unwrap_or_else(|_| {
                println!("Could not parse the number of workers.");
                std::process::exit(-1)
            });
            unwrap!(config.set("num_workers", num_workers));
        }
        if let Some(log_file) = arguments.opt_str("f") {
            unwrap!(config.set("log_file", log_file));
        }
        SearchAlgorithm::parse_arguments(arguments, config);
    }
}


impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", unwrap!(toml::to_string(self)))
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            log_file: String::from("watch.log"),
            event_log: String::from("eventlog.tfrecord"),
            num_workers: num_cpus::get(),
            algorithm: SearchAlgorithm::default(),
            stop_bound: None,
            timeout: None,
            max_evaluations: None,
            distance_to_best: None,
        }
    }
}

/// Exploration algorithm to use.
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag="type")]
#[serde(rename_all="snake_case")]
pub enum SearchAlgorithm {
    /// Evaluate all the candidates that cannot be pruned.
    BoundOrder,
    /// Use a multi-armed bandit algorithm.
    #[serde(rename="bandit")]
    MultiArmedBandit(BanditConfig),
}

impl SearchAlgorithm {
    /// Sets up the options that can be passed on the command line.
    fn setup_args_parser(opts: &mut getopts::Options) {
        opts.optopt("a", "algorithm", "exploration algorithm: bound_order or bandit",
                    "bound_order:bandit");
        BanditConfig::setup_args_parser(opts);
    }

    /// Overwrite the configuration with the parameters from the command line.
    fn parse_arguments(arguments: &getopts::Matches, config: &mut config::Config) {
        if let Some(algo) = arguments.opt_str("a") {
            unwrap!(config.set("algorithm", algo));
        }
        BanditConfig::parse_arguments(arguments, config);
    }
}

impl Default for SearchAlgorithm {
    fn default() -> Self { SearchAlgorithm::MultiArmedBandit(BanditConfig::default()) }
}

/// Configuration parameters specific to the multi-armed bandit algorithm.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct BanditConfig {
    /// Indicates how to select between nodes of the search tree when none of their
    /// children have been evaluated.
    pub new_nodes_order: NewNodeOrder,
    /// Indicates how to choose between nodes with at least one children evaluated.
    pub old_nodes_order: OldNodeOrder,
    /// The number of best execution times to remember.
    pub threshold: usize,
    /// The biggest delta is, the more focused on the previous best candidates the
    /// exploration is.
    pub delta: f64,
    /// If true, does not expand tree until end - instead, starts a montecarlo descend after each
    /// expansion of a node
    pub monte_carlo: bool,
}

impl BanditConfig {
    /// Sets up the options that can be passed on the command line.
    fn setup_args_parser(opts: &mut getopts::Options) {
        opts.optopt("s", "default_node_selection",
                    "selection algorithm for nodes without evaluations: \
                    api, random, bound, weighted_random",
                    "api|random|bound|weighted_random");
    }

    /// Overwrite the configuration with the parameters from the command line.
    fn parse_arguments(arguments: &getopts::Matches, config: &mut config::Config) {
        if let Some(algo) = arguments.opt_str("s") {
            unwrap!(config.set("new_nodes_order", algo));
        }
    }
}

impl Default for BanditConfig {
    fn default() -> Self {
        BanditConfig {
            new_nodes_order: NewNodeOrder::default(),
            old_nodes_order: OldNodeOrder::default(),
            threshold: 10,
            delta: 1.,
            monte_carlo: true,
        }
    }
}

/// Indicates how to choose between nodes of the search tree when no children have been
/// evaluated.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all="snake_case")]
pub enum NewNodeOrder {
    /// Consider the nodes in the order given by the search space API.
    Api,
    /// Consider the nodes in a random order.
    Random,
    /// Consider the nodes with the lowest bound first.
    Bound,
    /// Consider the nodes with a probability proportional to the distance between the
    /// cut and the bound.
    WeightedRandom,
}

impl Default for NewNodeOrder {
    fn default() -> Self { NewNodeOrder::WeightedRandom }
}

/// Indicates how to choose between nodes of the search tree with at least one descendent
/// evaluated.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all="snake_case")]
pub enum OldNodeOrder {
    /// Use the weights from the bandit algorithm.
    Bandit,
    /// Take the candidate with the best bound.
    Bound,
    /// Consider the nodes with a probability proportional to the distance between the
    /// cut and the bound.
    WeightedRandom,
}

impl Default for OldNodeOrder {
    fn default() -> Self { OldNodeOrder::Bandit }
}
