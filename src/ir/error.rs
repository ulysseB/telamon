//! Error management for IR creation.
use ir;
use std;

/// Errors that can be raised when creating an IR instance.
#[derive(Debug, Fail)]
pub enum TypeError {
    #[fail(display="type `{}` is not valid on the targeted device", t)]
    InvalidType { t: ir::Type },
    #[fail(display="{} must have a return type", inst)]
    ExpectedReturnType { inst: ir::InstId },
    #[fail(display="{} rounding is incompatible with type `{}`", rounding, t)]
    InvalidRounding { rounding: ir::op::Rounding, t: ir::Type },
    #[fail(display="expected {}, got `{}`", expected, given)]
    WrongType { given: ir::Type, expected: ExpectedType },
    #[fail(display="unexpected type `{}`", t)]
    UnexpectedType { t: ir::Type }

}

impl TypeError {
    /// Ensures a type is equal to the expected one.
    pub fn check_equals(given: ir::Type, expected: ir::Type) -> Result<(), Self> {
        if given == expected { Ok(()) } else {
            Err(TypeError::WrongType { given, expected: expected.into() })
        }
    }

    /// Ensures the given type is an integer type.
    pub fn check_integer(given: ir::Type) -> Result<(), Self> {
        if given.is_integer() { Ok(()) } else {
            Err(TypeError::WrongType { given, expected: ExpectedType::Integer })
        }
    }
}

/// Indicates what kind of type was expected.
#[derive(Debug)]
pub enum ExpectedType {
    /// An integer type was expected.
    Integer,
    /// A specific type was expected.
    Specific(ir::Type),
}

impl std::fmt::Display for ExpectedType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ExpectedType::Integer => write!(f, "an integer type"),
            ExpectedType::Specific(t) => write!(f, "type `{}`", t),
        }
    }
}

impl From<ir::Type> for ExpectedType {
    fn from(t: ir::Type) -> Self { ExpectedType::Specific(t) }
}

/// An error occuring while manipulating an ir instance.
#[derive(Debug, Fail)]
pub enum Error {
    #[fail(display="{}", _0)]
    Type(#[cause] TypeError),
    #[fail(display="dimensions must have a size of at least 2")]
    InvalidDimSize,
    #[fail(display="dimension {} appears twice in the increment list", dim)]
    DuplicateIncrement { dim: ir::dim::Id },
    #[fail(display="the access pattern references dimension {}, but is not nested inside", dim)]
    InvalidDimInPattern { dim: ir::dim::Id },
}

impl From<TypeError> for Error {
    fn from(e: TypeError) -> Self { Error::Type(e) }
}
