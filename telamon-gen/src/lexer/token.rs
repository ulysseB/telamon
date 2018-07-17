/// Tokens from the textual representation of constraints.
use ir;

use std::fmt;
use ::errno::Errno;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// The token incrimined.
    InvalidToken(String),
    /// The errno message of invalid include header file.
    InvalidInclude(String, Errno),
    ValueIdent(String), ChoiceIdent(String), Var(String), Doc(String), CmpOp(ir::CmpOp),
    Code(String), CounterKind(ir::CounterKind), Bool(bool),
    CounterVisibility(ir::CounterVisibility),
    And, Trigger, When, Alias, Counter, Define, Enum, Equal, Forall, In, Is, Not, Require,
    Requires, Value, End, Symmetric, AntiSymmetric, Arrow, Colon, Comma, LParen, RParen,
    BitOr, Or, SetDefKey(ir::SetDefKey), Set, SubsetOf, SetIdent(String), Base, Disjoint,
    Quotient, Of, Divide, Integer, 
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
