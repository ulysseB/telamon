/// TypeError describes the Ast error top level.
use ast::{ChoiceDef, Statement};
use lexer::Spanned;

/// Hint is a token representation.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Hint {
    /// Set interface.
    Set,
    /// Set attribute.
    SetAttribute,
    /// Enum interface.
    Enum,
    /// Enum attribute.
    EnumAttribute,
    /// Integer interface.
    Integer,
    /// Integer attribute.
    IntegerAttribute,
    /// Counter.
    Counter,
}

impl Hint {
    pub fn from(statement: &Statement) -> Self {
        match statement {
            Statement::SetDef(..) => Hint::Set,
            Statement::ChoiceDef(choice) => Hint::from_choice(choice),
            _ => unreachable!(),
        }
    }

    pub fn from_choice(choice: &ChoiceDef) -> Self {
        match choice {
            ChoiceDef::EnumDef(..) => Hint::Enum,
            ChoiceDef::IntegerDef(..) => Hint::Integer,
            ChoiceDef::CounterDef(..) => Hint::Counter,
        }
    }
}

/// TypeEror is the error representation of telamon's.
#[derive(Debug, Fail, PartialEq)]
pub enum TypeError {
    /// Redefinition of a name and hint.
    #[fail(display = "redefinition {:?} and {:?}", object_kind, object_name)]
    Redefinition {
        object_kind: Spanned<Hint>,
        object_name: Spanned<String>,
    },
    /// Undefinition of set, enum or field.
    #[fail(display = "undefined {:?}", object_name)]
    Undefined { object_name: Spanned<String> },
    /// Unvalid arguments of a symmetric enum.
    #[fail(
        display = "unvalid symmetric arguments {:?} and {:?}",
        object_name, object_variables
    )]
    BadSymmetricArg {
        object_name: Spanned<String>,
        object_variables: Vec<(Spanned<String>, String)>,
    },
    /// Missing
    /// Happens when the Set's object has a missing field.
    #[fail(display = "missing entry {:?} of set {:?}", object_name, object_field)]
    MissingEntry {
        object_name: String,
        object_field: Spanned<String>,
    },
    /// Conflict between incompatible keywords.
    /// Happens when the object has symmetric and antisimmetric fields.
    #[fail(display = "conflict between {:?}", object_fields)]
    Conflict {
        object_fields: (Spanned<String>, Spanned<String>),
    },
}
