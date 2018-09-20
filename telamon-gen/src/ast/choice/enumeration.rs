use std::ops::Deref;

use super::ChoiceDef;

use ast::error::{Hint, TypeError};
use ast::context::CheckerContext;
use ast::typing_context::TypingContext;
use ast::{
    SetRef,
    VarDef,
    VarMap,
    Condition,
    EnumStatements,
    EnumStatement,
    ChoiceInstance,
    Symmetry,
    HashSet
};
use ast::constrain::Constraint;
use lexer::Spanned;
use ir;

use utils::{RcStr, HashMap};
use itertools::Itertools;

/// A toplevel definition or constraint.
#[derive(Clone, Debug)]
pub struct EnumDef {
    pub name: Spanned<String>,
    pub doc: Option<String>,
    pub variables: Vec<VarDef>,
    pub statements: Vec<EnumStatement>,
}

impl EnumDef {
    /// This checks that there isn't any doublon in the field list.
    fn check_redefinition_field(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<String, _> = HashMap::default();
        let mut symmetric: Option<Spanned<()>> = None;
        let mut antisymmetric: Option<Spanned<()>> = None;

        for stmt in self.statements.iter() {
            match stmt {
                EnumStatement::AntiSymmetric(spanned) => {
                    if let Some(ref before) = antisymmetric {
                        Err(TypeError::Redefinition {
                            object_kind: before.with_data(Hint::EnumAttribute),
                            object_name: spanned.with_data(String::from("Antisymmetric")),
                        })?;
                    } else {
                        antisymmetric = Some(spanned.with_data(()));
                    }
                }
                EnumStatement::Symmetric(spanned) => {
                    if let Some(ref before) = symmetric {
                        Err(TypeError::Redefinition {
                            object_kind: before.with_data(Hint::EnumAttribute),
                            object_name: spanned.with_data(String::from("Symmetric")),
                        })?;
                    } else {
                        symmetric = Some(spanned.with_data(()));
                    }
                }
                EnumStatement::Value(spanned, ..) | EnumStatement::Alias(spanned, ..) => {
                    if let Some(before) =
                        hash.insert(spanned.data.to_owned(), spanned.with_data(()))
                    {
                        Err(TypeError::Redefinition {
                            object_kind: before.with_data(Hint::EnumAttribute),
                            object_name: spanned.with_data(spanned.data.to_owned()),
                        })?;
                    }
                }
            }
        }
        Ok(())
    }

    /// This checks that there isn't any doublon in parameter list.
    fn check_redefinition_parameter(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<String, _> = HashMap::default();
        for VarDef { name, .. } in self.variables.as_slice() {
            if let Some(before) = hash.insert(name.data.to_string(), name.with_data(())) {
                Err(TypeError::Redefinition {
                    object_kind: before.with_data(Hint::EnumAttribute),
                    object_name: name.with_data(name.data.to_string()),
                })?;
            }
        }
        Ok(())
    }

    /// This checks that both fields symmetric and antisymmetric aren't defined
    /// in the same enumeration.
    fn check_conflict(&self) -> Result<(), TypeError> {
        let mut symmetric: Option<Spanned<()>> = None;
        let mut antisymmetric: Option<Spanned<()>> = None;

        for stmt in self.statements.iter() {
            match stmt {
                EnumStatement::AntiSymmetric(spanned) => {
                    if let Some(ref symmetric) = symmetric {
                        Err(TypeError::Conflict {
                            object_fields: (
                                symmetric.with_data(String::from("Symmetric")),
                                spanned.with_data(String::from("Antisymmetric")),
                            ),
                        })?;
                    } else {
                        antisymmetric = Some(spanned.with_data(()));
                    }
                }
                EnumStatement::Symmetric(spanned) => {
                    if let Some(ref antisymmetric) = antisymmetric {
                        Err(TypeError::Conflict {
                            object_fields: (
                                antisymmetric.with_data(String::from("Antisymmetric")),
                                spanned.with_data(String::from("Symmetric")),
                            ),
                        })?;
                    } else {
                        symmetric = Some(spanned.with_data(()));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Checks if the values referenced in EnumStatements are defined.
    fn check_field(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<String, _> = HashMap::default();

        for stmt in self.statements.iter() {
            match stmt {
                EnumStatement::Value(spanned, ..) | EnumStatement::Alias(spanned, ..) => {
                    hash.insert(spanned.data.to_owned(), ());
                }
                _ => {}
            }
        }
        for stmt in self.statements.iter() {
            match stmt {
                EnumStatement::AntiSymmetric(spanned) => {
                    for (first, second) in spanned.data.iter() {
                        if !hash.contains_key(&first.to_owned()) {
                            Err(TypeError::Undefined {
                                object_name: spanned.with_data(first.to_owned()),
                            })?;
                        }
                        if !hash.contains_key(&second.to_owned()) {
                            Err(TypeError::Undefined {
                                object_name: spanned.with_data(second.to_owned()),
                            })?;
                        }
                    }
                }
                EnumStatement::Alias(spanned, _, sets, ..) => {
                    for set in sets {
                        if !hash.contains_key(&set.to_owned()) {
                            Err(TypeError::Undefined {
                                object_name: spanned.with_data(set.to_owned()),
                            })?;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// This checks that there is two parameters if the field symmetric is defined.
    fn check_two_parameter(&self) -> Result<(), TypeError> {
        if self
            .statements
            .iter()
            .find(|item| item.is_symmetric() || item.is_antisymmetric())
            .is_some()
        {
            if self.variables.len() != 2 {
                Err(TypeError::BadSymmetricArg {
                    object_name: self.name.to_owned(),
                    object_variables: self
                        .variables
                        .iter()
                        .map(|variable: &VarDef| {
                            let set_name: &String = variable.set.name.deref();

                            (variable.name.to_owned().into(), set_name.to_owned())
                        })
                        .collect::<Vec<(Spanned<String>, String)>>(),
                })?;
            }
        }
        Ok(())
    }

    /// This checkls that the parameters share the same type.
    fn check_same_parameter(&self) -> Result<(), TypeError> {
        if self
            .statements
            .iter()
            .find(|item| item.is_symmetric() || item.is_antisymmetric())
            .is_some()
        {
            match self.variables.as_slice() {
                [VarDef {
                    name: _,
                    set: SetRef { name, .. },
                }, VarDef {
                    name: _,
                    set: SetRef { name: rhs_name, .. },
                }] => {
                    if name != rhs_name {
                        Err(TypeError::BadSymmetricArg {
                            object_name: self.name.to_owned(),
                            object_variables: self
                                .variables
                                .iter()
                                .map(|variable| {
                                    let set_name: &String = variable.set.name.deref();

                                    (variable.name.to_owned().into(), set_name.to_owned())
                                })
                                .collect::<Vec<(Spanned<String>, String)>>(),
                        })?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// This checks if the variables are defined in the context.
    fn check_undefined_variables(
        &self,
        context: &CheckerContext,
    ) -> Result<(), TypeError> {
        for VarDef { name: _, ref set } in self.variables.iter() {
            if !context.check_set_define(set) {
                let name: &String = set.name.deref();

                Err(TypeError::Undefined {
                    object_name: self.name.with_data(name.to_owned()),
                })?;
            }
        }
        Ok(())
    }

    /// Type checks the declare's condition.
    pub fn declare(&self, context: &mut CheckerContext) -> Result<(), TypeError> {
        Ok(())
    }

    /// Register a constraint on an enum value.
    fn register_value_constraint(
        &self,
        choice: RcStr,
        args: Vec<VarDef>,
        value: RcStr,
        mut constraint: Constraint,
        tc: &mut TypingContext,
    ) {
        let choice_args = args.iter().map(|def| def.name.clone()).collect::<Vec<_>>();
        let self_instance = ChoiceInstance {
            name: choice,
            vars: choice_args.into_iter().map(|n| n.data).collect::<Vec<_>>(),
        };
        let condition = Condition::Is {
            lhs: self_instance,
            rhs: vec![value],
            is: false,
        };
        constraint.forall_vars.extend(args);
        for disjunction in &mut constraint.disjunctions {
            disjunction.push(condition.clone());
        }
        tc.constraints.push(constraint);
    }

    /// Registers an enum definition.
    fn register_enum(&self, tc: &mut TypingContext) {
        trace!("defining enum {}", self.name.data);
        let doc = self.doc.clone().map(RcStr::new);
        let enum_name = RcStr::new(::to_type_name(&self.name.data));
        let choice_name = RcStr::new(self.name.data.to_owned());
        let mut stmts = EnumStatements::default();
        for s in self.statements.iter().cloned() {
            stmts.add_statement(s);
        }
        // Register constraints
        for (value, constraint) in stmts.constraints {
            let choice = choice_name.clone();
            self.register_value_constraint(
                choice,
                self.variables.clone(),
                value,
                constraint,
                tc,
            );
        }
        // Typechek the anti-symmetry mapping.
        let (symmetric, inverse) = match stmts.symmetry {
            None => (false, false),
            Some(Symmetry::Symmetric) => (true, false),
            Some(Symmetry::AntiSymmetric(..)) => (true, true),
        };
        let mut var_map = VarMap::default();
        let vars = self
            .variables
            .to_owned()
            .into_iter()
            .map(|v| {
                let name = v.name.clone();
                (name, var_map.decl_argument(&tc.ir_desc, v))
            })
            .collect::<Vec<_>>();
        let arguments = ir::ChoiceArguments::new(
            vars.into_iter()
                .map(|(n, s)| (n.data, s))
                .collect::<Vec<_>>(),
            symmetric,
            inverse,
        );
        let inverse = if let Some(Symmetry::AntiSymmetric(mapping)) = stmts.symmetry {
            {
                let mut mapped = HashSet::default();
                for &(ref lhs, ref rhs) in &mapping {
                    assert!(stmts.values.contains_key(lhs), "unknown value {}", lhs);
                    assert!(stmts.values.contains_key(rhs), "unknown value {}", rhs);
                    assert!(mapped.insert(lhs), "{} is mapped twice", lhs);
                    assert!(mapped.insert(rhs), "{} is mapped twice", rhs);
                }
            }
            Some(mapping)
        } else {
            None
        };
        let mut enum_ = ir::Enum::new(enum_name.clone(), doc.clone(), inverse);
        // Register values and aliases
        for (name, doc) in stmts.values {
            enum_.add_value(name, doc);
        }
        for name in stmts.aliases.keys().cloned().collect_vec() {
            assert!(!enum_.values().contains_key(&name));
            let mut expanded_values = HashSet::default();
            let mut values = stmts
                .aliases
                .get_mut(&name)
                .unwrap()
                .1
                .drain()
                .collect_vec();
            while let Some(val) = values.pop() {
                if enum_.values().contains_key(&val) {
                    expanded_values.insert(val);
                } else if name == val {
                    panic!("loop in alias definition");
                } else if let Some(&(_, ref sub_vals)) = stmts.aliases.get(&val) {
                    values.extend(sub_vals.iter().cloned());
                } else {
                    panic!("undefined value in alias definition");
                }
            }
            stmts.aliases.get_mut(&name).unwrap().1 = expanded_values;
        }
        // Register aliases
        for (name, (doc, values)) in stmts.aliases {
            enum_.add_alias(name, values, doc);
        }
        // Register the enum and the choice.
        tc.ir_desc.add_enum(enum_);
        let choice_def = ir::ChoiceDef::Enum(enum_name);
        tc.ir_desc
            .add_choice(ir::Choice::new(choice_name, doc, arguments, choice_def));
    }

    /// Type checks the define's condition.
    pub fn define(
        self,
        context: &mut CheckerContext,
        tc: &mut TypingContext,
    ) -> Result<(), TypeError> {
        self.check_undefined_variables(context)?;
        self.check_redefinition_parameter()?;
        self.check_redefinition_field()?;
        self.check_field()?;
        self.check_two_parameter()?;
        self.check_same_parameter()?;
        self.check_conflict()?;

        self.register_enum(tc);

        tc.choice_defs.push(ChoiceDef::EnumDef(self));
        Ok(())
    }
}

impl PartialEq for EnumDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
