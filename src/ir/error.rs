//! Error management for IR creation.
use ir;

/// Errors that can be raised when creating an IR instance.
#[derive(Debug, Fail)]
pub enum TypeError {
    #[fail(display="type `{}` is not valid on the targeted device", t)]
    InvalidType { t: ir::Type },
    #[fail(display="{} must have a return type", inst)]
    ExpectedReturnType { inst: ir::InstId },
    #[fail(display="{} rounding is incompatible with type `{}`", rounding, t)]
    InvalidRounding { rounding: ir::op::Rounding, t: ir::Type },
    #[fail(display="expected type `{}`, got `{}`", expected, given)]
    WrongType { given: ir::Type, expected: ir::Type },
    #[fail(display="unexpected type `{}`", t)]
    UnexpectedType { t: ir::Type }

}

impl TypeError {
    /// Ensures a type is equal to the expected one.
    pub fn check_equals(given: ir::Type, expected: ir::Type) -> Result<(), Self> {
        if given == expected { Ok(()) } else {
            Err(TypeError::WrongType { given, expected })
        }
    }
}

/// An error occuring while manipulating an ir instance.
#[derive(Debug, Fail)]
pub enum Error {
    #[fail(display="{}", _0)]
    Type(#[cause] TypeError),
    #[fail(display="dimensions must have a size of at least 2")]
    InvalidDimSize,
}

impl From<TypeError> for Error {
    fn from(e: TypeError) -> Self { Error::Type(e) }
}
