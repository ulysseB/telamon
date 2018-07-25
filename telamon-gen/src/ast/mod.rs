#![allow(unused_variables)]

//! Syntaxic tree for the constraint description.

mod set;
mod trigger;
mod choice;
mod context;
mod constrain;

use constraint::Constraint as TypedConstraint;
use constraint::dedup_inputs;
use ir;
use itertools::Itertools;
use print;
use regex::Regex;
use std;
use std::fmt;
use std::collections::{BTreeSet, hash_map};
use std::ops::Deref;
use utils::*;
use indexmap::IndexMap;

pub use self::set::SetDef;
pub use self::choice::integer::IntegerDef;
pub use self::choice::enumeration::EnumDef;
pub use self::choice::ChoiceDef;
use self::trigger::TriggerDef;
use self::context::TypingContext;
pub use self::constrain::Constraint;

pub use super::lexer::{Position, Spanned};

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
}

impl Hint {
    fn from(statement: &Statement) -> Self {
        match statement {
            Statement::SetDef(..) => Hint::Set,
            Statement::ChoiceDef(ChoiceDef::EnumDef(..)) => Hint::Enum,
            Statement::ChoiceDef(ChoiceDef::IntegerDef(..)) => Hint::Integer,
            _ => unreachable!(),
        }
    }
}

/// TypeEror is the error representation of telamon's.
#[derive(Debug, PartialEq)]
pub enum TypeError {
    /// Redefinition of a name and hint..
    Redefinition(Spanned<Hint>, Spanned<String>),
    /// Undefinition of set, enum or field.
    Undefined(Spanned<String>),
    /// Unvalid arguments of a symmetric enum.
    BadSymmetricArg(Vec<VarDef>),
    /// Missing
    MissingEntry(String, Spanned<String>),
}

/// CheckContext is a type system.
#[derive(Debug, Default)]
struct CheckerContext {
    /// Map Name of unique identifiant.
    hash: HashMap<String, Spanned<Hint>>,
}

impl CheckerContext {
    /// This checks the undefined of EnumDef or IntegerDef.
    fn check_undefined_choicedef(
        &self, statement: ChoiceDef
    ) -> Result<(), TypeError> {
        match statement {
            ChoiceDef::EnumDef(EnumDef {
                name: Spanned { beg, end, data: _ }, doc: _, variables, .. }) |
            ChoiceDef::IntegerDef(IntegerDef {
                name: Spanned { beg, end, data: _ }, doc: _, variables, .. }) => {
                for VarDef { name: _, set: SetRef { name, .. } } in variables {
                    let name: &String = name.deref();
                    if !self.hash.contains_key(name) {
                        Err(TypeError::Undefined(Spanned {
                            beg, end,
                            data: name.to_owned(),
                        }))?;
                    }
                }
            },
            _ => {},
        }
        Ok(())
    }

    /// This checks the undefined of SetDef superset and arg.
    fn check_undefined_setdef(
        &self, statement: &SetDef
    ) -> Result<(), TypeError> {
        match statement {
            SetDef { name: Spanned { beg, end, data: ref name},
            doc: _, arg, superset, disjoint: _, keys, ..  } => {
                if let Some(VarDef { name: _, set: SetRef { name, .. } }) = arg {
                    let name: &String = name.deref();
                    if !self.hash.contains_key(name) {
                        Err(TypeError::Undefined(Spanned {
                            beg: *beg, end: *end, data: name.to_owned()
                        }))?;
                    }
                }
                if let Some(SetRef { name: supername, .. }) = superset {
                    let name: &String = supername.deref();
                    if !self.hash.contains_key(name) {
                        Err(TypeError::Undefined(Spanned {
                            beg: *beg, end: *end, data: name.to_owned()
                        }))?;
                    }
                }
            },
        }
        Ok(())
    }
    
    /// This checks the redefinition of SetDef, EnumDef and IntegerDef.
    fn check_redefinition(
        &mut self, statement: &Statement
    ) -> Result<(), TypeError> {
        match statement {
            Statement::SetDef(SetDef { name: Spanned { beg, end, data: name }, .. }) |
            Statement::ChoiceDef(ChoiceDef::EnumDef(
                EnumDef { name: Spanned { beg, end, data: name, }, ..  })) |
            Statement::ChoiceDef(ChoiceDef::IntegerDef(
                IntegerDef { name: Spanned { beg, end, data: name }, .. })) => {
                let data: Hint = Hint::from(&statement);
                let value: Spanned<Hint> = Spanned { beg: *beg, end: *end, data };
                if let Some(pre) = self.hash.insert(name.to_owned(), value) {
                    Err(TypeError::Redefinition(pre, Spanned {
                        beg: *beg, end: *end, data: name.to_owned(),
                    }))
                } else {
                    Ok(())
                }
            },
            _ => Ok(()),
        }
    }

    /// Type checks the condition.
    pub fn type_check(
        &mut self, statement: &Statement
    ) -> Result<(), TypeError> {
        self.check_redefinition(&statement)?;
        match statement {
            Statement::ChoiceDef(ChoiceDef::EnumDef(ref enumeration)) => {
                self.check_undefined_choicedef(
                    ChoiceDef::EnumDef(enumeration.clone()))?;
            },
            Statement::ChoiceDef(
                ChoiceDef::IntegerDef(ref integer)
            ) => {
                self.check_undefined_choicedef(
                    ChoiceDef::IntegerDef(integer.clone()))?;
            },
            Statement::SetDef(ref set) => {
                self.check_undefined_setdef(set)?;
            },
            _ => {},
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Ast { 
    pub statements: Vec<Statement>,
}

impl Ast {
    /// Generate the defintion of choices and the list of constraints.
    pub fn type_check(self)
        -> Result<(ir::IrDesc, Vec<TypedConstraint>), TypeError> {
        let mut checker = CheckerContext::default();
        let mut context = TypingContext::default();

        for statement in self.statements {
            checker.type_check(&statement)?;
            statement.type_check()?;
            context.add_statement(statement);
        }
        Ok(context.finalize())
    }
}

/// A toplevel definition or constraint.
#[derive(Debug)]
pub enum Statement {
    ChoiceDef(ChoiceDef),
    TriggerDef {
        foralls: Vec<VarDef>,
        conditions: Vec<Condition>,
        code: String,
    },
    SetDef(SetDef),
    Require(Constraint),
}

impl Statement {
    pub fn type_check(&self) -> Result<(), TypeError> {
        match self {
            Statement::SetDef(def) => def.type_check(),
            Statement::ChoiceDef(def) => def.type_check(),
            _ => Ok(()),
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
enum Check {
    /// Ensures the inverse of the value set is itself.
    IsSymmetric { choice: RcStr, values: Vec<RcStr> },
}

impl Check {
    /// Performs the check.
    fn check(&self, ir_desc: &ir::IrDesc) {
        match *self {
            Check::IsSymmetric { ref choice, ref values } =>
                Check::is_symmetric(ir_desc, choice, values),
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
pub enum Symmetry { Symmetric, AntiSymmetric(Vec<(RcStr, RcStr)>) }

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
    pub name: RcStr,
    pub set: SetRef,
}

impl PartialEq for VarDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.set == rhs.set
    }
}

/// Maps variables to their set and position.
#[derive(Default)]
struct VarMap {
    map: HashMap<RcStr, (ir::Variable, ir::Set)>,
    next_arg_id: usize,
    next_forall_id: usize,
}

impl VarMap {
    /// Declares a new argument variable.
    fn decl_argument(&mut self, ir_desc: &ir::IrDesc,  var_def: VarDef) -> ir::Set {
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
    fn decl_var(&mut self, ir_desc: &ir::IrDesc, var_def: VarDef,
                var: ir::Variable) -> ir::Set {
        let set = var_def.set.type_check(ir_desc, self);
        match self.map.entry(var_def.name.clone()) {
            hash_map::Entry::Occupied(..) =>
                panic!("variable {} defined twice", var_def.name),
            hash_map::Entry::Vacant(entry) => { entry.insert((var, set.clone())); },
        };
        set
    }

    /// Returns the variable associated with a name,
    fn get_var(&self, name: &str) -> ir::Variable { self.map[name].0 }

    /// Returns the variable associated with a name and checks its set.
    fn type_check(&self, name: &str, expected: &ir::Set) -> ir::Variable {
        let &(var, ref given_t) = self.get(name);
        assert!(given_t.is_subset_of_def(expected), "{:?} !< {:?}", given_t, expected);
        var
    }

    /// Returns the entry for a variable given its name.
    fn get(&self, name: &str) -> &(ir::Variable, ir::Set) {
        self.map.get(name).unwrap_or_else(|| {
            panic!("undefined variable {}", name)
        })
    }

    /// Returns the maping of variables to sets.
    fn env(&self) -> HashMap<ir::Variable, ir::Set> {
        self.map.values().cloned().collect()
    }
}

/// One of the condition that has to be respected by a constraint.
#[derive(Debug, Clone)]
pub enum Condition {
    Is { lhs: ChoiceInstance, rhs: Vec<RcStr>, is: bool },
    Code(RcStr, bool),
    Bool(bool),
    CmpCode { lhs: ChoiceInstance, rhs: RcStr, op: ir::CmpOp },
    CmpInput { lhs: ChoiceInstance, rhs: ChoiceInstance, op: ir::CmpOp },
}

impl Condition {
    fn new_is_bool(choice: ChoiceInstance, is: bool) -> Self {
        Condition::Is { lhs: choice, rhs: vec!["TRUE".into()], is }
    }

    /// Type checks the condition.
    fn type_check(self, ir_desc: &ir::IrDesc, var_map: &VarMap,
                  inputs: &mut Vec<ir::ChoiceInstance>)
            -> ir::Condition {
        match self {
            Condition::Code(code, negate) =>
                ir::Condition::Code { code: type_check_code(code, var_map), negate },
            Condition::Is { lhs, rhs, is } => {
                let choice = ir_desc.get_choice(&lhs.name);
                let enum_ = ir_desc.get_enum(choice.choice_def().as_enum().unwrap());
                let input_id = add_input(lhs, ir_desc, var_map, inputs);
                ir::Condition::Enum {
                    input: input_id,
                    values: type_check_enum_values(enum_, rhs),
                    negate: !is,
                    inverse: false
                }
            },
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
            },
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
                    inverse: false
                }
            },
            Condition::Bool(b) => ir::Condition::Bool(b),
        }
    }

    /// Negates the condition.
    fn negate(&mut self) {
        match *self {
            Condition::Is { is: ref mut negate, .. } |
            Condition::Code(_, ref mut negate) => *negate = !*negate,
            Condition::CmpCode { ref mut op, .. } |
            Condition::CmpInput { ref mut op, .. } => op.negate(),
            Condition::Bool(ref mut b) => *b = !*b,
        }
    }
}

/// Typecheck and adds an input to the inputs vector.
fn add_input(choice: ChoiceInstance, ir_desc: &ir::IrDesc, var_map: &VarMap,
             inputs: &mut Vec<ir::ChoiceInstance>) -> usize {
    let choice_def = ir_desc.get_choice(&choice.name);
    let vars = choice.vars.iter().zip_eq(choice_def.arguments().sets())
        .map(|(v, expected_t)| var_map.type_check(v, expected_t))
        .collect();
    let input_id = inputs.len();
    inputs.push(ir::ChoiceInstance { choice: choice.name, vars: vars });
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
        let vars = self.vars.iter().zip_eq(choice.arguments().sets())
            .map(|(v, s)| var_map.type_check(v, s)).collect();
        ir::ChoiceInstance { choice: self.name.clone(), vars }
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
    let vars = get_code_vars(&code).into_iter().flat_map(|var| {
        if var == "fun" { None } else {
            Some((var_map.get_var(&var), RcStr::new(var)))
        }
    }).collect();
    ir::Code { code, vars }
}

fn type_check_enum_values(enum_: &ir::Enum, values: Vec<RcStr>) -> BTreeSet<RcStr> {
    let values = enum_.expand(values).into_iter().collect();
    for value in &values { assert!(enum_.values().contains_key(value)); }
    values
}

/// The value of a counter increment.
#[derive(Clone, Debug)]
pub enum CounterVal { Code(String), Choice(ChoiceInstance) }

/// A statement in an enum definition.
#[derive(Clone, Debug)]
pub enum EnumStatement {
    /// Defines a possible decision for th enum.
    Value(Spanned<String>, Option<String>, Vec<Constraint>),
    /// Defines a set of possible decisions for the enum.
    Alias(Spanned<String>, Option<String>, Vec<String>, Vec<Constraint>),
    /// Specifies that the enum is symmetric.
    Symmetric(Spanned<()>),
    /// Specifies that the enum is antisymmetric and given the inverse function.
    AntiSymmetric(Spanned<Vec<(String, String)>>),
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
            (EnumStatement::Value(name, .. ),
             EnumStatement::Value(rhs_name, .. )) |
            (EnumStatement::Alias(name, .. ),
             EnumStatement::Alias(rhs_name, .. )) => {
                name.data.eq(&rhs_name.data)
            },
            (EnumStatement::Symmetric(..),
             EnumStatement::Symmetric(..)) => true,
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
                for c in constraints { self.constraints.push((name.clone(), c)); }
            },
            EnumStatement::Alias(name, doc, values, constraints) => {
                let name = RcStr::new(name.data);
                let values = values.into_iter().map(RcStr::new).collect();
                assert!(self.aliases.insert(name.clone(), (doc, values)).is_none());
                for c in constraints { self.constraints.push((name.clone(), c)); }
            },
            EnumStatement::Symmetric(..) => {
                assert!(self.symmetry.is_none());
                self.symmetry = Some(Symmetry::Symmetric);
            },
            EnumStatement::AntiSymmetric(mapping) => {
                assert!(self.symmetry.is_none());
                let mapping = mapping.data.into_iter()
                    .map(|(x, y)| (RcStr::new(x), RcStr::new(y))).collect();
                self.symmetry = Some(Symmetry::AntiSymmetric(mapping));
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct CounterDef {
    pub name: RcStr,
    pub doc: Option<String>,
    pub visibility: ir::CounterVisibility,
    pub vars: Vec<VarDef>,
    pub body: CounterBody,
}

impl PartialEq for CounterDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
