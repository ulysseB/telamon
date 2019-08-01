//! Defines a structure to store the configuration of the exploration. The configuration
//! is read from the `Setting.toml` file if it exists. Some parameters can be overridden
//! from the command line.

extern crate toml;

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::{self, error, fmt, str::FromStr};

use config;
use log::warn;
use num_cpus;
use serde::{Deserialize, Serialize};
use utils::{tfrecord, unwrap};

use crate::explorer::eventlog::EventLog;

/// Stores the configuration of the exploration.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Path to the output directory to use.  All other paths (e.g. `log_file`) are relative to
    /// this directory.  Defaults to the current working directory.
    pub output_dir: String,
    /// Name of the file in wich to store the logs.
    pub log_file: String,
    /// Name of the file in which to store the binary event log.
    pub event_log: String,
    /// Number of exploration threads.
    pub num_workers: usize,
    /// Indicates the search must be stopped if a candidate with an execution time better
    /// than the bound (in ns) is found.
    pub stop_bound: Option<f64>,
    /// If true, check all implementations found for correctness.  Otherwise (the default) only the
    /// best candidates are checked.
    pub check_all: bool,
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
        config_parser
    }

    /// Create a new configuration from the hardcoded `Settings.toml` configuration file if it
    /// exists.
    pub fn from_settings_toml() -> Self {
        let settings_path = std::path::Path::new("Settings.toml");
        if settings_path.exists() {
            warn!("*** Loading config from Settings.toml ***");
            warn!("*** Pay careful attention to the parameters used as they may differ from the defaults. ***");
            unwrap!(Self::from_path(settings_path))
        } else {
            Self::default()
        }
    }

    /// Extract the configuration from the given configuration file.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, config::ConfigError> {
        let mut parser = Self::create_parser();
        parser.merge(config::File::from(path.as_ref()))?;
        parser.try_into::<Self>()
    }

    /// Parse the configuration from a JSON string. Primary user is
    /// the Python API (through the C API).
    pub fn from_json(json: &str) -> Result<Self, config::ConfigError> {
        let mut parser = Self::create_parser();
        parser.merge(config::File::from_str(json, config::FileFormat::Json))?;
        parser.try_into::<Self>()
    }

    pub fn output_path<P: AsRef<Path>>(&self, path: P) -> io::Result<PathBuf> {
        let output_dir = Path::new(&self.output_dir);
        // Ensure the output directory exists
        std::fs::create_dir_all(output_dir)?;
        Ok(output_dir.join(path))
    }

    pub fn create_log(&self) -> io::Result<BufWriter<File>> {
        let mut f = File::create(self.output_path(&self.log_file)?)?;
        writeln!(f, "LOGGER\n{}", self)?;
        Ok(BufWriter::new(f))
    }

    pub fn create_eventlog(&self) -> io::Result<tfrecord::Writer<EventLog>> {
        EventLog::create(self.output_path(&self.event_log)?)
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
            output_dir: ".".to_string(),
            log_file: "watch.log".to_string(),
            event_log: "eventlog.tfrecord.gz".to_string(),
            check_all: false,
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
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum SearchAlgorithm {
    /// Evaluate all the candidates that cannot be pruned.
    BoundOrder,
    /// Use a MCTS algorithm
    Mcts(BanditConfig),
}

impl Default for SearchAlgorithm {
    fn default() -> Self {
        SearchAlgorithm::Mcts(BanditConfig::default())
    }
}

/// Configuration parameters specific to the multi-armed bandit algorithm.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct BanditConfig {
    /// Indicates the initial cut to use (in nanoseconds).  This can be used when an existing
    /// program (e.g. from a precedent run) is known to take that much amount of time.
    pub initial_cut: Option<f64>,
    /// Indicates whether we should backtrack locally when a dead-end is encountered.  If false,
    /// dead-ends will cause a restart from the root.
    pub backtrack_deadends: bool,
    /// Indicates how to select between nodes of the search tree when none of their
    /// children have been evaluated.
    pub new_nodes_order: NewNodeOrder,
    /// Order in which the different choices are going to be determined
    pub choice_ordering: ChoiceOrdering,
    /// Indicates how to choose between nodes with at least one children evaluated.
    pub tree_policy: TreePolicy,
}

/// Tree policy configuration
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum TreePolicy {
    /// Take the candidate with the best bound.
    Bound,

    /// Consider the nodes with a probability proportional to the distance between the
    /// cut and the bound.
    WeightedRandom,

    /// Policies based on TAG, as described in
    ///
    ///   Bandit-Based Optimization on Graphs with Application to Library Performance Tuning
    ///   De Mesmay, Rimmel, Voronenko, Püschel
    ///   ICML 2009
    ///
    /// Those policies make decisions based on the relative proportions of the top N samples in
    /// each branch, without relying directly on the values.  This obviates the issue of selecting
    /// a good threshold value for good vs bad samples (we don't know what a good value is until
    /// after we have found it!), but introduces an issue with staleness:
    #[serde(rename = "tag")]
    TAG(TAGConfig),

    /// Policies based on UCT, including variants such as p-UCT.  Those policies optimize a reduced
    /// value (such as average score or best score) among the samples seen in a branch, along with
    /// an uncertainty term to boost rarely explored branches.
    #[serde(rename = "uct")]
    UCT(UCTConfig),

    /// Always select the least visited child.
    RoundRobin,
}

impl Default for TreePolicy {
    fn default() -> Self {
        TreePolicy::TAG(TAGConfig::default())
    }
}

/// Configuration for the TAG algorithm
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct TAGConfig {
    /// The number of best execution times to remember.
    pub topk: usize,
    /// The biggest delta is, the more focused on the previous best candidates the
    /// exploration is.
    pub delta: f64,
}

impl Default for TAGConfig {
    fn default() -> Self {
        TAGConfig {
            topk: 10,
            delta: 1.,
        }
    }
}

/// Configuration for the UCT algorithm
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct UCTConfig {
    /// Constant multiplier for the exploration term.  This is controls the
    /// exploration-exploitation tradeoff, and is normalized using the `normalization` term.
    pub exploration_constant: f64,
    /// Normalization to use for the exploration term.
    pub normalization: Option<Normalization>,
    /// Reduction function to use when computing the state value.
    pub value_reduction: ValueReduction,
    /// Target to use as a reward.
    pub reward: Reward,
    /// Formula to use for the exploration term.
    pub formula: Formula,
}

impl Default for UCTConfig {
    fn default() -> Self {
        UCTConfig {
            exploration_constant: 2f64.sqrt(),
            normalization: None,
            // We use best value reduction by default because the mean value does not make much
            // sense considering that we can have infinite/very large values due to both timeouts
            // and cutting later in the evaluation process.
            value_reduction: ValueReduction::Best,
            reward: Reward::Speed,
            formula: Formula::Uct,
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Formula {
    /// Regular UCT formula: sqrt(log(\sum visits) / visits)
    Uct,
    /// AlphaGo PUCT variant: p * sqrt(\sum visits) / visits
    /// Currently only uniform prior is supported (p = 1 / k where k is the number of children).
    AlphaPuct,
}

#[derive(Copy, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reward {
    NegTime,
    Speed,
    LogSpeed,
}

#[derive(Copy, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueReduction {
    /// Use the mean evaluation time.  This yields the regular UCT value function.
    Mean,
    /// Use the best evaluation time.  This yields an algorithm similar to maxUCT from
    ///
    ///   Trial-based Heuristic Tree Search for Finite Horizon MDPs,
    ///   Thomas Keller and Malte Helmert
    Best,
}

#[derive(Copy, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Normalization {
    /// Normalize the exploration term according to the current global best.
    GlobalBest,
}

impl Default for BanditConfig {
    fn default() -> Self {
        BanditConfig {
            initial_cut: None,
            new_nodes_order: NewNodeOrder::default(),
            tree_policy: TreePolicy::default(),
            choice_ordering: ChoiceOrdering::default(),
            backtrack_deadends: false,
        }
    }
}

/// Indicates how to choose between nodes of the search tree when no children have been
/// evaluated.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
    fn default() -> Self {
        NewNodeOrder::WeightedRandom
    }
}

/// An enum listing the Group of choices we can make
/// For example, we can make first all DimKind decisions, then all Order decisions, etc.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ChoiceGroup {
    LowerLayout,
    Size,
    DimKind,
    DimMap,
    /// Order between all statements
    Order,
    /// Order between instructions and dimensions
    InstOrder,
    /// Order between non-vector dimensions
    DimOrder,
    MemSpace,
    InstFlag,
    /// Select thread dimensions (and prevent others from being threads)
    Threads,
    /// Pick size for dimensions which may be threads
    ThreadSize,
    /// Select vector dimensions (and prevent others from being vectors)
    Vectors,

    /// Exposes choices from Order defining whether two dimension are
    /// fused (explicitly sets Order::MERGED or explicitly eliminates
    /// Order:MERGED)
    DimFusion,

    /// Exposes choices from Order defining whether two dimension are
    /// nested (explicitly sets Order::INNER, Order::OUTER or
    /// eliminates these two orders)
    DimNesting,
}

impl fmt::Display for ChoiceGroup {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ChoiceGroup::*;

        f.write_str(match self {
            LowerLayout => "lower_layout",
            Size => "size",
            DimKind => "dim_kind",
            DimMap => "dim_map",
            Order => "order",
            InstOrder => "inst_order",
            DimOrder => "dim_order",
            MemSpace => "mem_space",
            InstFlag => "inst_flag",
            Threads => "threads",
            Vectors => "vectors",
            ThreadSize => "thread_size",
            DimFusion => "dim_fusion",
            DimNesting => "dim_nesting",
        })
    }
}

/// An error which can be returned when parsing a group of choices.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseChoiceGroupError(String);

impl error::Error for ParseChoiceGroupError {}

impl fmt::Display for ParseChoiceGroupError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid choice group value `{}`", self.0)
    }
}

impl FromStr for ChoiceGroup {
    type Err = ParseChoiceGroupError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use self::ChoiceGroup::*;

        Ok(match s {
            "lower_layout" => LowerLayout,
            "size" => Size,
            "dim_kind" => DimKind,
            "dim_map" => DimMap,
            "order" => Order,
            "inst_order" => InstOrder,
            "dim_order" => DimOrder,
            "mem_space" => MemSpace,
            "inst_flag" => InstFlag,
            "threads" => Threads,
            "vectors" => Vectors,
            "thread_size" => ThreadSize,
            "dim_fusion" => DimFusion,
            "dim_nesting" => DimNesting,
            _ => return Err(ParseChoiceGroupError(s.to_string())),
        })
    }
}

/// A list of ChoiceGroup representing the order in which we want to determine choices
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChoiceOrdering(Vec<ChoiceGroup>);

impl<'a> IntoIterator for &'a ChoiceOrdering {
    type Item = &'a ChoiceGroup;
    type IntoIter = std::slice::Iter<'a, ChoiceGroup>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

pub(super) const DEFAULT_ORDERING: [ChoiceGroup; 7] = [
    ChoiceGroup::LowerLayout,
    ChoiceGroup::Size,
    ChoiceGroup::DimKind,
    ChoiceGroup::DimMap,
    ChoiceGroup::MemSpace,
    ChoiceGroup::Order,
    ChoiceGroup::InstFlag,
];

impl Default for ChoiceOrdering {
    fn default() -> Self {
        ChoiceOrdering(DEFAULT_ORDERING.to_vec())
    }
}

impl fmt::Display for ChoiceOrdering {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some((first, rest)) = self.0.split_first() {
            write!(f, "{:?}", first)?;

            for elem in rest {
                write!(f, ",{:?}", elem)?;
            }
        }

        Ok(())
    }
}

impl FromStr for ChoiceOrdering {
    type Err = ParseChoiceGroupError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ChoiceOrdering(
            s.split(',')
                .map(str::parse)
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}
