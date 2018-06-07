use super::RcStr;
use super::{HashMap, HashSet};
use super::{Symmetry, Constraint};


/// A statement in an enum definition.
#[derive(Clone, Debug)]
pub enum EnumStatement {
    /// Defines a possible decision for th enum.
    Value(String, Option<String>, Vec<Constraint>),
    /// Defines a set of possible decisions for the enum.
    Alias(String, Option<String>, Vec<String>, Vec<Constraint>),
    /// Specifies that the enum is symmetric.
    Symmetric,
    /// Specifies that the enum is antisymmetric and given the inverse function.
    AntiSymmetric(Vec<(String, String)>),
}

impl PartialEq for EnumStatement {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (EnumStatement::Value(name, .. ),
             EnumStatement::Value(rhs_name, .. )) |
            (EnumStatement::Alias(name, .. ),
             EnumStatement::Alias(rhs_name, .. )) => {
                name.eq(rhs_name)
            },
            (EnumStatement::Symmetric,
             EnumStatement::Symmetric) => true,
            _ => false,
        }
    }
}

/// Gathers the different statements of an enum.
#[derive(Debug, Default)]
pub struct EnumStatements {
    /// The values the enum can take, with the atached documentation.
    pub values: HashMap<RcStr, Option<String>>,
    /// Aliases mapped to the corresponding documentation and value set.
    pub aliases: HashMap<RcStr, (Option<String>, HashSet<RcStr>)>,
    /// Symmetry information.
    pub symmetry: Option<Symmetry>,
    /// Constraints on a value.
    pub constraints: Vec<(RcStr, Constraint)>,
}

impl EnumStatements {
    /// Registers an `EnumStatement`.
    pub fn add_statement(&mut self, statement: EnumStatement) {
        match statement {
            EnumStatement::Value(name, doc, constraints) => {
                let name = RcStr::new(name);
                assert!(self.values.insert(name.clone(), doc).is_none());
                for c in constraints { self.constraints.push((name.clone(), c)); }
            },
            EnumStatement::Alias(name, doc, values, constraints) => {
                let name = RcStr::new(name);
                let values = values.into_iter().map(RcStr::new).collect();
                assert!(self.aliases.insert(name.clone(), (doc, values)).is_none());
                for c in constraints { self.constraints.push((name.clone(), c)); }
            },
            EnumStatement::Symmetric => {
                assert!(self.symmetry.is_none());
                self.symmetry = Some(Symmetry::Symmetric);
            },
            EnumStatement::AntiSymmetric(mapping) => {
                assert!(self.symmetry.is_none());
                let mapping = mapping.into_iter()
                    .map(|(x, y)| (RcStr::new(x), RcStr::new(y))).collect();
                self.symmetry = Some(Symmetry::AntiSymmetric(mapping));
            },
        }
    }
}
