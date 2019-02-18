//! Helpers to generate code from an IR instance and fully specified decisions.
mod cfg;
mod dimension;
mod function;
mod name_map;
mod printer;
mod size;
mod variable;

pub use self::cfg::Cfg;
pub use self::dimension::{Dimension, InductionLevel, InductionVar};
pub use self::function::*;
pub use self::name_map::{DeclType, NameMap, Namer, Operand};
pub use self::printer::{MulMode, Printer};
pub use self::size::Size;
pub use self::variable::Variable;

// TODO(cleanup): refactor function
// - extend instructions with additional information: vector factor, flag, instantiated dims
// TODO(cleanup): refactor namer
