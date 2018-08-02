/// TypeError describes the Ast error top level.

use super::{Spanned, ChoiceDef, VarDef, Statement};

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
#[derive(Debug, PartialEq)]
pub enum TypeError {
    /// Redefinition of a name and hint.
    Redefinition { object_kind: Spanned<Hint>, object_name: Spanned<String> },
    /// Undefinition of set, enum or field.
    Undefined { object_name: Spanned<String> },
    /// Unvalid arguments of a symmetric enum.
    BadSymmetricArg { object_name: Spanned<String>, object_variables: Vec<VarDef> },
    /// Missing
    /// Happens when the Set's object has a missing field.
    MissingEntry { object_name: String, object_field: Spanned<String> },
    /// Conflict between incompatible keywords.
    /// Happens when the object has symmetric and antisimmetric fields. 
    Conflict { object_fields: (Spanned<String>, Spanned<String>) },
}
