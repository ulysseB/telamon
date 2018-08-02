//! Helpers to generate code from an IR instance and fully specified decisions.
mod cfg;
mod dimension;
mod function;
mod namer;
mod printer;

pub use self::cfg::Cfg;
pub use self::dimension::{Dimension, InductionLevel, InductionVar};
pub use self::function::*;
pub use self::namer::{Namer, NameMap, Value, VarType};
pub use self::printer::{MulMode, Printer};

// TODO(cleanup): refactor function
// - extend instructions with additional information: vector factor, flag, instantiated dims
// TODO(cleanup): refactor namer
