//! Helpers to generate code from an IR instance and fully specified decisions.
mod dimension;

pub mod cfg;
pub mod function;
pub mod namer;

pub use self::cfg::Cfg;
pub use self::dimension::{Dimension, InductionLevel, InductionVar};
pub use self::function::{Function, ParamVal, InternalMemBlock, AllocationScheme};
pub use self::namer::{Namer, NameMap, Value};

// TODO(cleanup): refactor function
// - extend instructions with additional information: vector factor, flag, instantiated dims
// TODO(cleanup): refactor namer
