#![allow(unused_variables)]

//! Syntaxic tree for the constraint description.

mod choice;
mod constrain;
mod context;
mod error;
mod set;
mod trigger;

use indexmap::IndexMap;
use ir;
use itertools::Itertools;
use print;
use regex::Regex;
use std::collections::{hash_map, BTreeSet};
use std::fmt;
use utils::{HashMap, HashSet, RcStr};

pub use constraint::dedup_inputs;
pub use constraint::Constraint as TypedConstraint;

pub use super::lexer::{Position, Spanned};

pub use self::choice::{ChoiceDef, CounterDef, EnumDef, IntegerDef};
pub use self::constrain::Constraint;
pub use self::context::CheckerContext;
pub use self::error::{Hint, TypeError};
pub use self::set::SetDef;
pub use self::trigger::TriggerDef;

#[derive(Default, Clone, Debug)]
pub struct Ast {
    pub statements: Vec<Statement>,
    pub ir_desc: ir::IrDesc,
    pub set_defs: Vec<SetDef>,
    pub choice_defs: Vec<ChoiceDef>,
    pub triggers: Vec<TriggerDef>,
    pub constraints: Vec<Constraint>,
    pub checks: Vec<Check>,
}

impl Ast {
    /// Generate the defintion of choices and the list of constraints.
    pub fn type_check(mut self) -> Result<(ir::IrDesc, Vec<TypedConstraint>), TypeError> {
        let mut context = CheckerContext::default();

        // declare
        for statement in self.statements.iter() {
            statement.declare(&mut context)?;
        }
        let statements: Vec<Statement> = self.statements.clone();
        for statement in statements {
            statement.define(
                &mut context,
                &mut self.set_defs,
                &mut self.ir_desc,
                &mut self.checks,
                &mut self.choice_defs,
                &mut self.constraints,
                &mut self.triggers,
            )?;
        }
        Ok(self.finalize())
    }

    /// Type-checks the statements in the correct order.
    pub fn finalize(mut self) -> (ir::IrDesc, Vec<TypedConstraint>) {
        for choice_def in self.choice_defs.iter() {
            match choice_def {
                ChoiceDef::CounterDef(counter_def) => {
                    counter_def
                        .register_counter(&mut self.ir_desc, &mut self.constraints);
                }
                _ => {}
            }
        }
        for trigger in self.triggers.iter() {
            trigger.register_trigger(&mut self.ir_desc);
        }
        let constraints = {
            let ir_desc = &self.ir_desc;
            self.constraints
                .into_iter()
                .flat_map(move |constraint| constraint.type_check(ir_desc))
                .collect_vec()
        };
        for check in self.checks {
            check.check(&self.ir_desc);
        }
        (self.ir_desc, constraints)
    }
}

/// A toplevel definition or constraint.
#[derive(Debug, Clone)]
pub enum Statement {
    ChoiceDef(ChoiceDef),
    TriggerDef(TriggerDef),
    SetDef(SetDef),
    Require(Constraint),
}

impl Statement {
    pub fn declare(&self, checker: &mut CheckerContext) -> Result<(), TypeError> {
        match self {
            Statement::SetDef(def) => def.declare(checker),
            Statement::ChoiceDef(def) => def.declare(checker),
            _ => Ok(()),
        }
    }

    pub fn define(
        self,
        context: &mut CheckerContext,
        set_defs: &mut Vec<SetDef>,
        ir_desc: &mut ir::IrDesc,
        checks: &mut Vec<Check>,
        choice_defs: &mut Vec<ChoiceDef>,
        constraints: &mut Vec<Constraint>,
        triggers: &mut Vec<TriggerDef>,
    ) -> Result<(), TypeError> {
        match self {
            Statement::SetDef(def) => def.define(
                context,
                set_defs,
                ir_desc,
                checks,
                choice_defs,
                constraints,
                triggers,
            ),
            Statement::ChoiceDef(def) => {
                def.define(context, ir_desc, constraints, choice_defs)
            }
            Statement::TriggerDef(def) => def.define(context, triggers),
            Statement::Require(def) => def.define(context, constraints),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Quotient {
    pub item: VarDef,
    pub representant: RcStr,
    pub conditions: Vec<Condition>,
    pub equiv_relation: (String, Vec<RcStr>),
}

/// Checks to perform once the statements have been declared.
#[derive(Clone, Debug)]
pub enum Check {
    /// Ensures the inverse of the value set is itself.
    IsSymmetric { choice: RcStr, values: Vec<RcStr> },
}

impl Check {
    /// Performs the check.
    fn check(&self, ir_desc: &ir::IrDesc) {
        match *self {
            Check::IsSymmetric {
                ref choice,
                ref values,
            } => Check::is_symmetric(ir_desc, choice, values),
        }
    }

    /// Ensures the value set of the given choice is symmetric.
    fn is_symmetric(ir_desc: &ir::IrDesc, choice: &str, values: &[RcStr]) {
        let choice = ir_desc.get_choice(choice);
        assert!(choice.arguments().is_symmetric());
        let enum_ = choice.choice_def().as_enum().expect("enum choice expected");
        let enum_ = ir_desc.get_enum(enum_);
        let value_set = enum_.expand(values.iter().cloned());
        let inverse_set = value_set.iter().map(|v| enum_.inverse(v).clone()).collect();
        assert_eq!(value_set, inverse_set);
    }
}

#[derive(Clone, Debug)]
pub struct CounterBody {
    pub base: String,
    pub kind: ir::CounterKind,
    pub iter_vars: Vec<VarDef>,
    pub value: CounterVal,
    pub conditions: Vec<Condition>,
}

/// Indicates if an enum exhibits symmetry.
#[derive(Debug)]
pub enum Symmetry {
    Symmetric,
    AntiSymmetric(Vec<(RcStr, RcStr)>),
}

/// References a set.
#[derive(Clone, Debug)]
pub struct SetRef {
    pub name: RcStr,
    pub var: Option<RcStr>,
}

impl PartialEq for SetRef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}

impl SetRef {
    fn type_check(&self, ir_desc: &ir::IrDesc, var_map: &VarMap) -> ir::Set {
        let set_def = ir_desc.get_set_def(&self.name);
        let var = if let Some(ref var) = self.var {
            Some(var_map.type_check(var, set_def.arg().unwrap()))
        } else {
            assert!(set_def.arg().is_none());
            None
        };
        ir::Set::new(set_def, var)
    }
}

/// Defines a variable.
#[derive(Debug, Clone)]
pub struct VarDef {
    pub name: Spanned<RcStr>,
    pub set: SetRef,
}

impl PartialEq for VarDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.set == rhs.set
    }
}

/// Maps variables to their set and position.
#[derive(Default)]
pub struct VarMap {
    map: HashMap<RcStr, (ir::Variable, ir::Set)>,
    next_arg_id: usize,
    next_forall_id: usize,
}

impl VarMap {
    /// Declares a new argument variable.
    fn decl_argument(&mut self, ir_desc: &ir::IrDesc, var_def: VarDef) -> ir::Set {
        let var = ir::Variable::Arg(self.next_arg_id);
        self.next_arg_id += 1;
        self.decl_var(ir_desc, var_def, var)
    }

    /// Declares a new forall variable.
    fn decl_forall(&mut self, ir_desc: &ir::IrDesc, var_def: VarDef) -> ir::Set {
        let var = ir::Variable::Forall(self.next_forall_id);
        self.next_forall_id += 1;
        self.decl_var(ir_desc, var_def, var)
    }

    /// Declares a variable.
    fn decl_var(
        &mut self,
        ir_desc: &ir::IrDesc,
        var_def: VarDef,
        var: ir::Variable,
    ) -> ir::Set {
        let set = var_def.set.type_check(ir_desc, self);
        match self.map.entry(var_def.name.data.clone()) {
            hash_map::Entry::Occupied(..) => {
                panic!("variable {} defined twice", var_def.name.data)
            }
            hash_map::Entry::Vacant(entry) => {
                entry.insert((var, set.clone()));
            }
        };
        set
    }

    /// Returns the variable associated with a name,
    fn get_var(&self, name: &str) -> ir::Variable {
        self.map[name].0
    }

    /// Returns the variable associated with a name and checks its set.
    fn type_check(&self, name: &str, expected: &ir::Set) -> ir::Variable {
        let &(var, ref given_t) = self.get(name);
        assert!(
            given_t.is_subset_of_def(expected),
            "{:?} !< {:?}",
            given_t,
            expected
        );
        var
    }

    /// Returns the entry for a variable given its name.
    fn get(&self, name: &str) -> &(ir::Variable, ir::Set) {
        self.map
            .get(name)
            .unwrap_or_else(|| panic!("undefined variable {}", name))
    }

    /// Returns the maping of variables to sets.
    fn env(&self) -> HashMap<ir::Variable, ir::Set> {
        self.map.values().cloned().collect()
    }
}

/// One of the condition that has to be respected by a constraint.
#[derive(Debug, Clone)]
pub enum Condition {
    Is {
        lhs: ChoiceInstance,
        rhs: Vec<RcStr>,
        is: bool,
    },
    Code(RcStr, bool),
    Bool(bool),
    CmpCode {
        lhs: ChoiceInstance,
        rhs: RcStr,
        op: ir::CmpOp,
    },
    CmpInput {
        lhs: ChoiceInstance,
        rhs: ChoiceInstance,
        op: ir::CmpOp,
    },
}

impl Condition {
    fn new_is_bool(choice: ChoiceInstance, is: bool) -> Self {
        Condition::Is {
            lhs: choice,
            rhs: vec!["TRUE".into()],
            is,
        }
    }

    /// Type checks the condition.
    fn type_check(
        self,
        ir_desc: &ir::IrDesc,
        var_map: &VarMap,
        inputs: &mut Vec<ir::ChoiceInstance>,
    ) -> ir::Condition {
        match self {
            Condition::Code(code, negate) => ir::Condition::Code {
                code: type_check_code(code, var_map),
                negate,
            },
            Condition::Is { lhs, rhs, is } => {
                let choice = ir_desc.get_choice(&lhs.name);
                let enum_ = ir_desc.get_enum(choice.choice_def().as_enum().unwrap());
                let input_id = add_input(lhs, ir_desc, var_map, inputs);
                ir::Condition::Enum {
                    input: input_id,
                    values: type_check_enum_values(enum_, rhs),
                    negate: !is,
                    inverse: false,
                }
            }
            Condition::CmpCode { lhs, rhs, op } => {
                let choice = ir_desc.get_choice(&lhs.name);
                choice.choice_def().is_valid_operator(op);
                if let ir::ChoiceDef::Enum(..) = *choice.choice_def() {
                    panic!("enums cannot be compared to host code");
                }
                ir::Condition::CmpCode {
                    lhs: add_input(lhs, ir_desc, var_map, inputs),
                    rhs: type_check_code(rhs, var_map),
                    op: op,
                }
            }
            Condition::CmpInput { lhs, rhs, op } => {
                assert_eq!(lhs.name, rhs.name);
                let choice = ir_desc.get_choice(&lhs.name);
                let lhs_input = add_input(lhs, ir_desc, var_map, inputs);
                let rhs_input = add_input(rhs, ir_desc, var_map, inputs);
                assert!(choice.choice_def().is_valid_operator(op));
                assert!(choice.choice_def().is_valid_operator(op.inverse()));
                ir::Condition::CmpInput {
                    lhs: lhs_input,
                    rhs: rhs_input,
                    op: op,
                    inverse: false,
                }
            }
            Condition::Bool(b) => ir::Condition::Bool(b),
        }
    }

    /// Negates the condition.
    fn negate(&mut self) {
        match *self {
            Condition::Is {
                is: ref mut negate, ..
            }
            | Condition::Code(_, ref mut negate) => *negate = !*negate,
            Condition::CmpCode { ref mut op, .. }
            | Condition::CmpInput { ref mut op, .. } => op.negate(),
            Condition::Bool(ref mut b) => *b = !*b,
        }
    }
}

/// Typecheck and adds an input to the inputs vector.
fn add_input(
    choice: ChoiceInstance,
    ir_desc: &ir::IrDesc,
    var_map: &VarMap,
    inputs: &mut Vec<ir::ChoiceInstance>,
) -> usize {
    let choice_def = ir_desc.get_choice(&choice.name);
    let vars = choice
        .vars
        .iter()
        .zip_eq(choice_def.arguments().sets())
        .map(|(v, expected_t)| var_map.type_check(v, expected_t))
        .collect();
    let input_id = inputs.len();
    inputs.push(ir::ChoiceInstance {
        choice: choice.name,
        vars: vars,
    });
    input_id
}

/// A reference to a choice instantiated with the given variables.
#[derive(Debug, Clone)]
pub struct ChoiceInstance {
    pub name: RcStr,
    pub vars: Vec<RcStr>,
}

impl ChoiceInstance {
    /// Type check the choice instance.
    fn type_check(&self, ir_desc: &ir::IrDesc, var_map: &VarMap) -> ir::ChoiceInstance {
        let choice = ir_desc.get_choice(&self.name);
        let vars = self
            .vars
            .iter()
            .zip_eq(choice.arguments().sets())
            .map(|(v, s)| var_map.type_check(v, s))
            .collect();
        ir::ChoiceInstance {
            choice: self.name.clone(),
            vars,
        }
    }
}

/// Returns the variable present in a piece of code.
fn get_code_vars(code: &str) -> HashSet<String> {
    lazy_static! {
        static ref VAR_PATTERN: Regex = Regex::new(r"\$[a-zA-Z_][a-zA-Z_0-9]*").unwrap();
    }
    let vars = VAR_PATTERN.find_iter(code);
    vars.map(|m| m.as_str()[1..].to_string()).collect()
}

fn type_check_code(code: RcStr, var_map: &VarMap) -> ir::Code {
    let vars = get_code_vars(&code)
        .into_iter()
        .flat_map(|var| {
            if var == "fun" {
                None
            } else {
                Some((var_map.get_var(&var), RcStr::new(var)))
            }
        }).collect();
    ir::Code { code, vars }
}

fn type_check_enum_values(enum_: &ir::Enum, values: Vec<RcStr>) -> BTreeSet<RcStr> {
    let values = enum_.expand(values).into_iter().collect();
    for value in &values {
        assert!(enum_.values().contains_key(value));
    }
    values
}

/// The value of a counter increment.
#[derive(Clone, Debug)]
pub enum CounterVal {
    Code(String),
    Choice(ChoiceInstance),
}

/// A statement in an enum definition.
#[derive(Clone, Debug)]
pub enum EnumStatement {
    /// Defines a possible decision for th enum.
    Value(Spanned<String>, Option<String>, Vec<Constraint>),
    /// Defines a set of possible decisions for the enum.
    Alias(
        Spanned<String>,
        Option<String>,
        Vec<String>,
        Vec<Constraint>,
    ),
    /// Specifies that the enum is symmetric.
    Symmetric(Spanned<()>),
    /// Specifies that the enum is antisymmetric and given the inverse function.
    AntiSymmetric(Spanned<Vec<(String, String)>>),
}

impl EnumStatement {
    pub fn is_symmetric(&self) -> bool {
        if let EnumStatement::Symmetric(..) = self {
            true
        } else {
            false
        }
    }

    pub fn is_antisymmetric(&self) -> bool {
        if let EnumStatement::AntiSymmetric(..) = self {
            true
        } else {
            false
        }
    }
}

impl fmt::Display for EnumStatement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EnumStatement::Alias(name, ..) => write!(f, "{}", name.data),
            EnumStatement::Value(name, ..) => write!(f, "{}", name.data),
            EnumStatement::Symmetric(..) => write!(f, "Symmetric"),
            EnumStatement::AntiSymmetric(..) => write!(f, "AntiSymmetric"),
        }
    }
}

impl PartialEq for EnumStatement {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (EnumStatement::Value(name, ..), EnumStatement::Value(rhs_name, ..))
            | (EnumStatement::Alias(name, ..), EnumStatement::Alias(rhs_name, ..)) => {
                name.data.eq(&rhs_name.data)
            }
            (EnumStatement::Symmetric(..), EnumStatement::Symmetric(..)) => true,
            _ => false,
        }
    }
}

/// Gathers the different statements of an enum.
#[derive(Debug, Default)]
struct EnumStatements {
    /// The values the enum can take, with the atached documentation.
    values: IndexMap<RcStr, Option<String>>,
    /// Aliases mapped to the corresponding documentation and value set.
    aliases: IndexMap<RcStr, (Option<String>, HashSet<RcStr>)>,
    /// Symmetry information.
    symmetry: Option<Symmetry>,
    /// Constraints on a value.
    constraints: Vec<(RcStr, Constraint)>,
}

impl EnumStatements {
    /// Registers an `EnumStatement`.
    fn add_statement(&mut self, statement: EnumStatement) {
        match statement {
            EnumStatement::Value(name, doc, constraints) => {
                let name = RcStr::new(name.data);
                assert!(self.values.insert(name.clone(), doc).is_none());
                for c in constraints {
                    self.constraints.push((name.clone(), c));
                }
            }
            EnumStatement::Alias(name, doc, values, constraints) => {
                let name = RcStr::new(name.data);
                let values = values.into_iter().map(RcStr::new).collect();
                assert!(self.aliases.insert(name.clone(), (doc, values)).is_none());
                for c in constraints {
                    self.constraints.push((name.clone(), c));
                }
            }
            EnumStatement::Symmetric(..) => {
                assert!(self.symmetry.is_none());
                self.symmetry = Some(Symmetry::Symmetric);
            }
            EnumStatement::AntiSymmetric(mapping) => {
                assert!(self.symmetry.is_none());
                let mapping = mapping
                    .data
                    .into_iter()
                    .map(|(x, y)| (RcStr::new(x), RcStr::new(y)))
                    .collect();
                self.symmetry = Some(Symmetry::AntiSymmetric(mapping));
            }
        }
    }
}
