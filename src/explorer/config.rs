//! Defines a structure to store the configuration of the exploration. The configuration
//! is read from the `Setting.toml` file if it exists. Some parameters can be overridden
//! from the command line.
use config;
use std;
use getopts;
use itertools::Itertools;
use num_cpus;

/// Stores the configuration of the exploration.
#[derive(Clone)]
pub struct Config {
    /// Name of the file in wich to store the logs.
    pub log_file: String,
    /// Number of exploration threads.
    pub num_workers: usize,
    /// Exploration algorithm to use.
    pub algorithm: SearchAlgorithm,
    /// Indicates the search must be stopped if a candidate with an execution time better
    /// than the bound (in ns) is found.
    pub stop_bound: Option<f64>,
    /// Indicates the search must be stopped after the given number of minutes.
    pub timeout: Option<u64>,
    /// A percentage cut indicate that we only care to find a candidate that is in a
    /// certain range above the best Therefore, if cut_under is 20%, we can discard any
    /// candidate whose bound is above 80% of the current best.
    pub distance_to_best: Option<f64>,
}

impl Config {
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
        let mut config_parser = Self::setup_config_parser();
        let config_path = std::path::Path::new("Settings.toml");
        if config_path.exists() {
            unwrap!(config_parser.merge(config::File::from(config_path)));
        }
        Self::parse_arguments(&arg_matches, &mut config_parser);
        Self::parse_config(&config_parser)
    }

    /// Sets up the parser of the configuration file.
    fn setup_config_parser() -> config::Config {
        let mut parser = config::Config::new();
        unwrap!(parser.set_default("log_file", String::from("watch.log")));
        unwrap!(parser.set_default("num_workers", num_cpus::get() as i64));
        SearchAlgorithm::setup_config_parser(&mut parser);
        parser
    }

    /// Extracts the parameters from the configuration file.
    fn parse_config(parser: &config::Config) -> Self {
        Config {
            log_file: unwrap!(parser.get_str("log_file")),
            num_workers: unwrap!(parser.get_int("num_workers")) as usize,
            algorithm: SearchAlgorithm::parse_config(parser),
            stop_bound: optional_param(parser.get_float("stop_bound")),
            timeout: optional_param(parser.get_int("timeout")).map(|x| x as u64),
            distance_to_best: optional_param(parser.get_float("distance_to_best")),
        }
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

fn optional_param<T>(res: Result<T, config::ConfigError>) -> Option<T> {
    match res {
        Ok(t) => Some(t),
        Err(config::ConfigError::NotFound(_)) => None,
        Err(err) => panic!(err),
    }
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "log_file = \"{}\"", self.log_file)?;
        writeln!(f, "num_workers = {}", self.num_workers)?;
        if let Some(b) = self.stop_bound { writeln!(f, "stop_bound = {}", b)?; }
        if let Some(b) = self.timeout { writeln!(f, "timeout = {}", b)?; }
        self.algorithm.fmt(f)?;
        Ok(())
    }
}

/// Exploration algorithm to use.
#[derive(Clone)]
pub enum SearchAlgorithm {
    /// Evaluate all the candidates that cannot be pruned.
    BoundOrder,
    /// Use a multi-armed bandit algorithm.
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

    /// Sets up the parser of the configuration file.
    fn setup_config_parser(parser: &mut config::Config) {
        unwrap!(parser.set_default("algorithm", String::from("bandit")));
        BanditConfig::setup_config_parser(parser);
    }

    /// Extracts the parameters from the configuration file.
    fn parse_config(parser: &config::Config) -> Self {
        match &unwrap!(parser.get_str("algorithm")) as &str {
            "bound_order" => SearchAlgorithm::BoundOrder,
            "bandit" => {
                let bandit_config = BanditConfig::parse_config(parser);
                SearchAlgorithm::MultiArmedBandit(bandit_config)
            },
            algo => panic!("invalid search algorithm: {}. \
                           Must be algorithm=bound_order|bandit", algo),
        }
    }
}

impl std::fmt::Display for SearchAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            SearchAlgorithm::BoundOrder => writeln!(f, "algorithm = \"bound_order\""),
            SearchAlgorithm::MultiArmedBandit(ref config) => {
                writeln!(f, "algorithm = \"bandit\"")?;
                config.fmt(f)
            },
        }
    }
}

/// Configuration parameters specific to the multi-armed bandit algorithm.
#[derive(Clone)]
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

    /// Sets up the parser of the configuration file.
    fn setup_config_parser(parser: &mut config::Config) {
        NewNodeOrder::setup_config_parser(parser);
        OldNodeOrder::setup_config_parser(parser);
        unwrap!(parser.set_default("threshold", 10));
        unwrap!(parser.set_default("delta", 0.001));
        unwrap!(parser.set_default("monte_carlo", false));
    }

    /// Extracts the parameters from the configuration file.
    fn parse_config(parser: &config::Config) -> Self {
        BanditConfig {
            new_nodes_order: NewNodeOrder::parse_config(parser),
            old_nodes_order: OldNodeOrder::parse_config(parser),
            threshold: unwrap!(parser.get_int("threshold")) as usize,
            delta: unwrap!(parser.get_float("delta")),
            monte_carlo: unwrap!(parser.get_bool("monte_carlo")),
        }
    }
}

impl std::fmt::Display for BanditConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "new_nodes_order = \"{}\"", self.new_nodes_order)?;
        writeln!(f, "old_nodes_order = \"{}\"", self.old_nodes_order)?;
        writeln!(f, "threshold = {}", self.threshold)?;
        writeln!(f, "delta = {}", self.delta)?;
        writeln!(f, "monte_carlo = {}", self.monte_carlo)?;
        Ok(())
    }
}

/// Indicates how to choose between nodes of the search tree when no children have been
/// evaluated.
#[derive(Clone, Copy)]
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

impl NewNodeOrder {
    /// Sets up the parser of the configuration file.
    fn setup_config_parser(parser: &mut config::Config) {
        unwrap!(parser.set_default("new_nodes_order", "weighted_random"));
    }

    /// Extracts the parameters from the configuration file.
    fn parse_config(parser: &config::Config) -> Self {
        match &unwrap!(parser.get_str("new_nodes_order")) as &str {
            "api" => NewNodeOrder::Api,
            "random" => NewNodeOrder::Random,
            "bound" => NewNodeOrder::Bound,
            "weighted_random" => NewNodeOrder::WeightedRandom,
            _ => panic!("Wrong new_nodes_order option, \
                       must be new_nodes_order = api|random|bound|weighted_random")
        }
    }
}

impl std::fmt::Display for NewNodeOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            NewNodeOrder::Api => "api",
            NewNodeOrder::Random => "random",
            NewNodeOrder::Bound => "bound",
            NewNodeOrder::WeightedRandom => "weighted_random",
        }.fmt(f)
    }
}

/// Indicates how to choose between nodes of the search tree with at least one descendent
/// evaluated.
#[derive(Clone)]
pub enum OldNodeOrder {
    /// Use the weights from the bandit algorithm.
    Bandit,
    /// Take the candidate with the best bound.
    Bound,
    /// Consider the nodes with a probability proportional to the distance between the
    /// cut and the bound.
    WeightedRandom,
}

impl OldNodeOrder {
    /// Sets up the parser of the configuration file.
    fn setup_config_parser(parser: &mut config::Config) {
        unwrap!(parser.set_default("old_nodes_order", "bandit"));
    }

    /// Extracts the parameters from the configuration file.
    fn parse_config(parser: &config::Config) -> Self {
        match &unwrap!(parser.get_str("old_nodes_order")) as &str {
            "bandit" => OldNodeOrder::Bandit,
            "bound" => OldNodeOrder::Bound,
            "weighted_random" => OldNodeOrder::WeightedRandom,
            _ =>  panic!("Wrong old_nodes_order option, \
                         must be old_nodes_order = bound|bandit|weighted_random")
        }
    }
}

impl std::fmt::Display for OldNodeOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            OldNodeOrder::Bandit => "bandit",
            OldNodeOrder::Bound => "bound",
            OldNodeOrder::WeightedRandom => "weighted_random",
        }.fmt(f)
    }
}
