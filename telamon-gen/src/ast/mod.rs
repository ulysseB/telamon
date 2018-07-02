#![allow(unused_variables)]

//! Syntaxic tree for the constraint description.
use constraint::Constraint as TypedConstraint;
use constraint::dedup_inputs;
use ir::{self, Adaptable};
use itertools::Itertools;
use print;
use regex::Regex;
use std;
use std::fmt;
use std::collections::{BTreeSet, hash_map};
use std::ops::Deref;
use utils::*;

pub use super::lexer::Spanned;

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
    fn from(statement: &Spanned<Statement>) -> Self {
        match statement {
            Spanned {
                beg: _, end: _, data: Statement::SetDef { ..  }
            } => Hint::Set,
            Spanned {
                beg: _, end: _, data: Statement::EnumDef { .. }
            } => Hint::Enum,
            Spanned {
                beg: _, end: _, data: Statement::IntegerDef { .. }
            } => Hint::Integer,
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
}

/// CheckContext is a type system.
#[derive(Debug, Default)]
struct CheckerContext {
    /// Map Name of unique identifiant.
    hash: HashMap<String, Spanned<Hint>>,
}

impl CheckerContext {

    fn undefined_set(&self, name: &RcStr) -> Result<(), String> {
        let name: &String = name.deref();
        if !self.hash.contains_key(name) {
            Err(name.to_owned())
        } else {
            Ok(())
        }
    }

    pub fn undefined(
        &self, statement: &Spanned<Statement>
    ) -> Result<(), TypeError> {
        match statement {
            Spanned { beg, end, data: Statement::EnumDef {
                name: _, doc: _, variables, ..  } } |
            Spanned { beg, end, data: Statement::IntegerDef {
                name: _, doc: _, variables, ..  } } => {
                for VarDef { name: _, set: SetRef { name, .. } } in variables {
                    let name: &String = name.deref();
                    if !self.hash.contains_key(name) {
                        Err(TypeError::Undefined(Spanned {
                            beg: *beg, end: *end,
                            data: name.to_owned(),
                        }))?;
                    }
                }
                Ok(())
            },
            Spanned { beg, end, data: Statement::SetDef {
                name: _, doc: _, arg, superset, disjoint: _, keys, ..  } } => {
                let keys = keys.iter().map(|(k, _, _)| k)
                                      .collect::<Vec<&ir::SetDefKey>>();

                if !keys.contains(&&ir::SetDefKey::ItemType) {
                    Err(TypeError::Undefined(Spanned {
                        beg: *beg, end: *end,
                        data: ir::SetDefKey::ItemType.to_string()
                    }))?;
                }
                if !keys.contains(&&ir::SetDefKey::IdType) {
                    Err(TypeError::Undefined(Spanned {
                        beg: *beg, end: *end,
                        data: ir::SetDefKey::IdType.to_string()
                    }))?;
                }
                if !keys.contains(&&ir::SetDefKey::ItemGetter) {
                    Err(TypeError::Undefined(Spanned {
                        beg: *beg, end: *end,
                        data: ir::SetDefKey::ItemGetter.to_string()
                    }))?;
                }
                if !keys.contains(&&ir::SetDefKey::IdGetter) {
                    Err(TypeError::Undefined(Spanned {
                        beg: *beg, end: *end,
                        data: ir::SetDefKey::IdGetter.to_string()
                    }))?;
                }
                if !keys.contains(&&ir::SetDefKey::Iter) {
                    Err(TypeError::Undefined(Spanned {
                        beg: *beg, end: *end,
                        data: ir::SetDefKey::Iter.to_string()
                    }))?;
                }
                if let Some(VarDef { name: _, set: SetRef { name, .. } }) = arg {
                    self.undefined_set(name).map_err(|s| TypeError::Undefined(Spanned {
                        beg: *beg, end: *end, data: s
                    }))?;
                }
                if let Some(SetRef { name, .. }) = superset {
                    self.undefined_set(name).map_err(|s| TypeError::Undefined(Spanned {
                        beg: *beg, end:*end, data: s
                    }))?;
                }
                Ok(())
            },
            _ => Ok(()),
        }
    }

    pub fn subredefinition(
        &mut self, statement: &Spanned<Statement>
    ) -> Result<(), TypeError> {
        match statement {
            Spanned { beg, end, data: Statement::EnumDef {
                name: _, doc: _, variables: _, statements, .. } } => {
                let mut hash: HashMap<String, _> = HashMap::default();
                for stmt in statements {
                    match stmt {
                        EnumStatement::Value(name, _, _) |
                        EnumStatement::Alias(name, ..) => {
                            if let Some(_) = hash.insert(
                                name.to_owned(),
                                Spanned {
                                    data: (),
                                    beg: Default::default(),
                                    end: Default::default(),
                                }
                            ) {
                                Err(TypeError::Redefinition(Spanned {
                                    data: Hint::EnumAttribute,
                                    beg: Default::default(),
                                    end: Default::default(),
                                }, Spanned {
                                    data: name.to_owned(),
                                    beg: Default::default(),
                                    end: Default::default(),
                                }))?;
                            }
                        },
                        _ => {},
                    }
                }
                Ok(())
            },
            _ => Ok(()),
        }
    }

    pub fn redefinition(
        &mut self, statement: &Spanned<Statement>
    ) -> Result<(), TypeError> {
        match statement {
            Spanned { beg: _, end: _, data: Statement::SetDef {
                name: Spanned { beg, end, data: name, }, .. } } |
            Spanned { beg: _, end: _, data: Statement::EnumDef {
                name: Spanned { beg, end, data: name, }, .. } } |
            Spanned { beg: _, end: _, data: Statement::IntegerDef {
                name: Spanned { beg, end, data: name, }, .. } } => {
                let data: Hint = Hint::from(statement);
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
}

/// Syntaxic tree for the constraint description.
#[derive(Debug)]
pub struct Ast { 
    pub statements: Vec<Spanned<Statement>>,
}

impl Ast {
    /// Generate the defintion of choices and the list of constraints.
    pub fn type_check(self) -> Result<(ir::IrDesc, Vec<TypedConstraint>), TypeError> {
        let mut checker = CheckerContext::default();
        let mut context = TypingContext::default();

        for statement in self.statements {
            checker.redefinition(&statement)?;
            checker.undefined(&statement)?;
            checker.subredefinition(&statement)?;
            context.add_statement(statement);
        }
        Ok(context.finalize())
    }
}

#[derive(Default)]
struct TypingContext {
    ir_desc: ir::IrDesc,
    set_defs: Vec<SetDef>,
    choice_defs: Vec<ChoiceDef>,
    triggers: Vec<TriggerDef>,
    constraints: Vec<Constraint>,
    checks: Vec<Check>,
}

impl TypingContext {
    /// Adds a statement to the typing context.
    fn add_statement(&mut self, statement: Spanned<Statement>) {
        match statement {
            Spanned { beg, end, data: Statement::SetDef {
                name: Spanned {
                    beg: _,
                    end: _,
                    data: name,
                }, doc, arg, superset, disjoint, keys, quotient
            } } => {
                self.set_defs.push(SetDef {
                    name, doc, arg, superset, disjoint, keys, quotient
                })
            },
            Spanned { beg, end, data: stmt @ Statement::EnumDef { .. } } |
            Spanned { beg, end, data: stmt @ Statement::IntegerDef { .. } } |
            Spanned { beg, end, data: stmt @ Statement::CounterDef { .. } } => {
                self.choice_defs.push(ChoiceDef::from(stmt))
            },
            Spanned { beg, end,
                data: Statement::TriggerDef { foralls, conditions, code }
            } => {
                self.triggers.push(TriggerDef {
                    foralls: foralls,
                    conditions: conditions,
                    code: code,
                })
            },
            Spanned { beg, end,
                data: Statement::Require(constraint)
            } => self.constraints.push(constraint),
        }
    }

    /// Type-checks the statements in the correct order.
    fn finalize(mut self) -> (ir::IrDesc, Vec<TypedConstraint>) {
        for def in std::mem::replace(&mut self.set_defs, vec![]) {
            self.type_set_def(
                def.name, def.arg, def.superset, def.keys, def.disjoint, def.quotient
            );
        }
        for choice_def in std::mem::replace(&mut self.choice_defs, vec![]) {
            match choice_def {
                ChoiceDef::EnumDef(EnumDef { name, doc, variables, statements }) =>
                    self.register_enum(name, doc, variables, statements),
                ChoiceDef::CounterDef(CounterDef { name, doc, visibility,
                    vars, body, }) =>
                    self.register_counter(name, doc, visibility, vars, body),
                ChoiceDef::IntegerDef(def) => self.define_integer(def),
            }
        }
        for trigger in std::mem::replace(&mut self.triggers, vec![]) {
            self.register_trigger(trigger.foralls, trigger.conditions, trigger.code);
        }
        let constraints = {
            let ir_desc = &self.ir_desc;
            self.constraints.into_iter()
                .flat_map(move |constraint| constraint.type_check(ir_desc))
                .collect_vec()
        };
        for check in self.checks { check.check(&self.ir_desc); }
        (self.ir_desc, constraints)
    }

    fn type_set_def(&mut self, name: String,
                    arg_def: Option<VarDef>,
                    superset: Option<SetRef>,
                    keys: Vec<(ir::SetDefKey, Option<VarDef>, String)>,
                    disjoints: Vec<String>,
                    quotient: Option<Quotient>) {
        trace!("defining set {}", name);
        let mut var_map = VarMap::default();
        let arg_name = arg_def.as_ref().map(|var| "$".to_string() + &var.name);
        let arg = arg_def.clone().map(|arg| var_map.decl_argument(&self.ir_desc, arg));
        let superset = superset.map(|set| set.type_check(&self.ir_desc, &var_map));
        for disjoint in &disjoints { self.ir_desc.get_set_def(disjoint); }
        let mut keymap = HashMap::default();
        let mut reverse = None;
        for (key, var, mut value) in keys {
            let mut env = key.env();
            // Add the set argument to the environement.
            if let Some(ref arg_name) = arg_name {
                // TODO(cleanup): use ir::Code to avoid using a dummy name.
                // Currently, we may have a collision on the $var name.
                if key.is_arg_in_env() {
                    value = value.replace(arg_name, "$var");
                    env.push("var");
                }
            }
            // Handle the optional forall.
            if key == ir::SetDefKey::Reverse {
                let var_def = var.as_ref().unwrap();
                let var_name = "$".to_string() + &var_def.name;
                value = value.replace(&var_name, "$var");
                env.push("var");
            } else { assert!(var.is_none()); }
            // Type-check the key.
            for var in get_code_vars(&value) {
                assert!(env.contains(&(&var as &str)),
                        "unexpected variable ${} for key {:?}", var, key);
            }
            // Register the key.
            if key == ir::SetDefKey::Reverse {
                let set = var.unwrap().set.type_check(&self.ir_desc, &VarMap::default());
                assert!(superset.as_ref().unwrap().is_subset_of_def(&set));
                assert!(std::mem::replace(&mut reverse, Some((set, value))).is_none());
            } else {
                assert!(keymap.insert(key, value).is_none());
            }
        }
        // Ensure required keys are present
        assert_eq!(arg.is_some() && superset.is_some(), reverse.is_some(),
                   "reverse key is missing");
        for key in &ir::SetDefKey::REQUIRED { assert!(keymap.contains_key(key)); }
        if superset.is_some() {
            assert!(keymap.contains_key(&ir::SetDefKey::FromSuperset));
        }
        let def = ir::SetDef::new(name, arg, superset, reverse, keymap, disjoints);
        if let Some(quotient) = quotient {
            self.create_quotient(&def, quotient, arg_def);
        }
        self.ir_desc.add_set_def(def);
    }

    /// Creates the choices that implement the quotient set.
    fn create_quotient(&mut self, set: &ir::SetDef,
                       quotient: Quotient,
                       arg: Option<VarDef>) {
        trace!("defining quotient {}", set.name());
        assert!(set.attributes().contains_key(&ir::SetDefKey::AddToSet));
        let repr_name = quotient.representant;
        // Create decisions to back the quotient set
        self.create_repr_choice(repr_name.clone(), set, arg.clone(),
                                quotient.item.name.clone());
        let item_name = quotient.item.name.clone();
        let arg_name = arg.as_ref().map(|x| x.name.clone());
        let forall_vars = arg.clone().into_iter()
            .chain(std::iter::once(quotient.item)).collect_vec();
        let counter_name = self.create_repr_counter(
            set.name().clone(), &repr_name, arg.clone(), item_name.clone(),
            forall_vars.clone(),
            RcStr::new(quotient.equiv_relation.0),
            quotient.equiv_relation.1);
        // Generate the code that set an item as representant.
        let trigger_code = print::add_to_quotient(
            set, &repr_name, &counter_name, &item_name, &arg_name);
        // Constraint the representative value.
        let forall_names = forall_vars.iter().map(|x| x.name.clone()).collect_vec();
        let repr_instance = ChoiceInstance { name: repr_name, vars: forall_names.clone() };
        let counter_instance = ChoiceInstance { name: counter_name, vars: forall_names };
        let not_repr = Condition::new_is_bool(repr_instance.clone(), false);
        let counter_leq_zero = Condition::CmpCode {
            lhs: counter_instance, rhs: "0".into(), op: ir::CmpOp::Leq
        };
        // Add the constraints `repr is FALSE || dividend is true` and
        // `repr is FALSE || counter <= 0`.
        let mut disjunctions = quotient.conditions.iter()
            .map(|c| vec![not_repr.clone(), c.clone()]).collect_vec();
        disjunctions.push(vec![not_repr, counter_leq_zero.clone()]);
        let repr_constraints = Constraint::new(forall_vars.clone(), disjunctions);
        self.constraints.push(repr_constraints);
        // Add the constraint `repr is TRUE || counter > 0 || dividend is false`.
        let repr_true = Condition::new_is_bool(repr_instance, true);
        let mut counter_gt_zero = counter_leq_zero.clone();
        counter_gt_zero.negate();
        let mut repr_true_conditions = vec![repr_true.clone(), counter_gt_zero];
        for mut cond in quotient.conditions.iter().cloned() {
            cond.negate();
            repr_true_conditions.push(cond);
        }
        self.constraints.push(Constraint {
            forall_vars: forall_vars.clone(),
            disjunctions: vec![repr_true_conditions],
            restrict_fragile: false,
        });
        // Add the constraint `item in set => repr is TRUE`.
        let quotient_item_def = VarDef {
            name: item_name,
            set: SetRef { name: set.name().clone(), var: arg_name }
        };
        let item_in_set_foralls = arg.into_iter()
            .chain(std::iter::once(quotient_item_def)).collect();
        self.constraints.push(Constraint::new(item_in_set_foralls, vec![vec![repr_true]]));
        // Generate the trigger that sets the repr to TRUE and add the item to the set.
        let mut trigger_conds = quotient.conditions;
        trigger_conds.push(counter_leq_zero);
        self.triggers.push(TriggerDef {
            foralls: forall_vars, conditions: trigger_conds, code: trigger_code,
        });
    }

    /// Creates a boolean choice that indicates if an object represents a givne class.
    fn create_repr_choice(&mut self, name: RcStr,
                          set: &ir::SetDef,
                          arg: Option<VarDef>,
                          item_name: RcStr) {
        let bool_str: RcStr = "Bool".into();
        let def = ir::ChoiceDef::Enum(bool_str.clone());
        let mut vars = Vec::new();
        if let Some(arg) = arg.as_ref() {
            vars.push((arg.name.clone(), set.arg().unwrap().clone()));
        }
        vars.push((item_name, set.superset().unwrap().clone()));
        let args = ir::ChoiceArguments::new(vars, false, false);
        let mut repr = ir::Choice::new(name, None, args, def);
        let false_value_set = std::iter::once("FALSE".into()).collect();
        repr.add_fragile_values(ir::ValueSet::enum_values(bool_str, false_value_set));
        self.ir_desc.add_choice(repr);
    }

    /// Creates a counter for the number of objects that can represent another object in
    /// a quotient set. Returns the name of the counter.
    fn create_repr_counter(&mut self, set_name: RcStr,
                           repr_name: &str,
                           arg: Option<VarDef>,
                           item_name: RcStr,
                           vars: Vec<VarDef>,
                           equiv_choice_name: RcStr,
                           equiv_values: Vec<RcStr>) -> RcStr {
        // Create the increment condition
        self.checks.push(Check::IsSymmetric {
            choice: equiv_choice_name.clone(), values: equiv_values.clone()
        });
        let rhs_name = RcStr::new(format!("{}_repr", item_name));
        let rhs_set = SetRef {
            name: set_name,
            var: arg.as_ref().map(|d| d.name.clone())
        };
        let equiv_choice = ChoiceInstance {
            name: equiv_choice_name,
            vars: vec![item_name, rhs_name.clone()],
        };
        let condition = Condition::Is { lhs: equiv_choice, rhs: equiv_values, is: true };
        // Create the counter.
        let name = RcStr::new(format!("{}_class_counter", repr_name));
        let visibility = ir::CounterVisibility::HiddenMax;
        let body = CounterBody {
            base: "0".to_string(),
            conditions: vec![condition],
            iter_vars: vec![VarDef { name: rhs_name, set: rhs_set }],
            kind: ir::CounterKind::Add,
            value: CounterVal::Code("1".to_string()),
        };
        self.choice_defs.push(ChoiceDef::CounterDef(CounterDef {
            name: name.clone(), doc: None, visibility, vars, body,
        }));
        name
    }

    /// Registers an enum definition.
    fn register_enum(&mut self, name: String, doc: Option<String>, vars: Vec<VarDef>,
                     statements: Vec<EnumStatement>) {
        trace!("defining enum {}", name);
        let doc = doc.map(RcStr::new);
        let enum_name = RcStr::new(::to_type_name(&name));
        let choice_name = RcStr::new(name);
        let mut stmts = EnumStatements::default();
        for s in statements { stmts.add_statement(s); }
        // Register constraints
        for (value, constraint) in stmts.constraints {
            let choice = choice_name.clone();
            self.register_value_constraint(choice, vars.clone(), value, constraint);
        }
        // Typechek the anti-symmetry mapping.
        let (symmetric, inverse) = match stmts.symmetry {
            None => (false, false),
            Some(Symmetry::Symmetric) => (true, false),
            Some(Symmetry::AntiSymmetric(..)) => (true, true),
        };
        let mut var_map = VarMap::default();
        let vars = vars.into_iter().map(|v| {
            let name = v.name.clone();
            (name, var_map.decl_argument(&self.ir_desc, v))
        }).collect();
        let arguments = ir::ChoiceArguments::new(vars, symmetric, inverse);
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
        } else { None };
        let mut enum_ = ir::Enum::new(enum_name.clone(), doc.clone(), inverse);
        // Register values and aliases
        for (name, doc) in stmts.values { enum_.add_value(name, doc); }
        for name in stmts.aliases.keys().cloned().collect_vec() {
            assert!(!enum_.values().contains_key(&name));
            let mut expanded_values = HashSet::default();
            let mut values = stmts.aliases.get_mut(&name).unwrap().1.drain().collect_vec();
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
        for (name, (doc, values)) in stmts.aliases { enum_.add_alias(name, values, doc); }
        // Register the enum and the choice.
        self.ir_desc.add_enum(enum_);
        let choice_def = ir::ChoiceDef::Enum(enum_name);
        self.ir_desc.add_choice(ir::Choice::new(choice_name, doc, arguments, choice_def));
    }

    /// Defines an integer choice.
    fn define_integer(&mut self, def: IntegerDef) {
        let choice_name = RcStr::new(def.name);
        let doc = def.doc.map(RcStr::new);
        let mut var_map = VarMap::default();
        let vars = def.variables.into_iter().map(|v| {
            let name = v.name.clone();
            (name, var_map.decl_argument(&self.ir_desc, v))
        }).collect();
        let arguments = ir::ChoiceArguments::new(vars, false, false);
        let universe = type_check_code(def.code.into(), &var_map);
        let choice_def = ir::ChoiceDef::Number { universe };
        self.ir_desc.add_choice(ir::Choice::new(choice_name, doc, arguments, choice_def));
    }

    /// Register a constraint on an enum value.
    fn register_value_constraint(&mut self, choice: RcStr, args: Vec<VarDef>,
                                 value: RcStr, mut constraint: Constraint) {
        let choice_args = args.iter().map(|def| def.name.clone()).collect();
        let self_instance = ChoiceInstance { name: choice, vars: choice_args };
        let condition = Condition::Is { lhs: self_instance, rhs: vec![value], is: false };
        constraint.forall_vars.extend(args);
        for disjunction in &mut constraint.disjunctions {
            disjunction.push(condition.clone());
        }
        self.constraints.push(constraint);
    }

    /// Registers a counter in the ir description.
    fn register_counter(&mut self,
                        counter_name: RcStr,
                        doc: Option<String>,
                        visibility: ir::CounterVisibility,
                        untyped_vars: Vec<VarDef>,
                        body: CounterBody) {
        trace!("defining counter {}", counter_name);
        let mut var_map = VarMap::default();
        // Type-check the base.
        let kind = body.kind;
        let all_var_defs = untyped_vars.iter().chain(&body.iter_vars).cloned().collect();
        let vars = untyped_vars.into_iter().map(|def| {
            (def.name.clone(), var_map.decl_argument(&self.ir_desc, def))
        }).collect_vec();
        let base = type_check_code(RcStr::new(body.base), &var_map);
        // Generate the increment
        let iter_vars = body.iter_vars.into_iter().map(|def| {
            (def.name.clone(), var_map.decl_forall(&self.ir_desc, def))
        }).collect_vec();
        let doc = doc.map(RcStr::new);
        let (incr, incr_condition) = self.gen_increment(
            &counter_name, &vars, &iter_vars, all_var_defs, body.conditions, &var_map);
        // Type check the value.
        let value = match body.value {
            CounterVal::Code(code) =>
                ir::CounterVal::Code(type_check_code(RcStr::new(code), &var_map)),
            CounterVal::Choice(counter) => {
                let counter_name = counter_name.clone();
                let (value, action) = self.counter_val_choice(
                    &counter, visibility, counter_name, &incr, kind, vars.len(), &var_map);
                self.ir_desc.add_onchange(&counter.name, action);
                value
            },
        };
        let incr_counter = self.gen_incr_counter(
            &counter_name, vars.len(), &var_map, &incr, value.clone());
        self.ir_desc.add_onchange(&incr.choice, incr_counter);
        // Register the counter choices.
        let incr_iter = iter_vars.iter().map(|p| p.1.clone()).collect_vec();
        let counter_def = ir::ChoiceDef::Counter {
            incr_iter, kind, value, incr, incr_condition, visibility, base
        };
        let counter_args = ir::ChoiceArguments::new(vars, false, false);
        let mut counter_choice = ir::Choice::new(
            counter_name, doc, counter_args, counter_def);
        // Filter the counter itself after an update, because the filter actually acts on
        // the increments and depends on the counter value.
        let filter_self = ir::OnChangeAction {
            forall_vars: vec![],
            set_constraints: ir::SetConstraints::default(),
            action: ir::ChoiceAction::FilterSelf
        };
        counter_choice.add_onchange(filter_self);
        self.ir_desc.add_choice(counter_choice);
    }

    /// Creates a choice to store the increment condition of a counter. Returns the
    /// corresponding choice instance from the point of view of the counter and the
    /// condition on wich the counter must be incremented.
    fn gen_increment(&mut self, counter: &str,
                     counter_vars: &[(RcStr, ir::Set)],
                     iter_vars: &[(RcStr, ir::Set)],
                     all_vars_defs: Vec<VarDef>,
                     conditions: Vec<Condition>,
                     var_map: &VarMap) -> (ir::ChoiceInstance, ir::ValueSet) {
        // TODO(cleanup): the choice the counter increment is based on must be declared
        // before the increment. It should not be the case.
        match conditions[..] {
            [Condition::Is { ref lhs, ref rhs, is }] => {
                let incr = lhs.type_check(&self.ir_desc, var_map);
                // Ensure all forall values are usefull.
                let mut foralls = HashSet::default();
                for &v in &incr.vars {
                    if let ir::Variable::Forall(i) = v { foralls.insert(i); }
                }
                if foralls.len() == iter_vars.len() {
                    // Generate the increment condition.
                    let choice = self.ir_desc.get_choice(&incr.choice);
                    let enum_ = self.ir_desc.get_enum(choice.choice_def().as_enum().unwrap());
                    let values = type_check_enum_values(enum_, rhs.clone());
                    let values = if is { values } else {
                        enum_.values().keys().filter(|&v| !values.contains(v))
                            .cloned().collect()
                    };
                    return (incr, ir::ValueSet::enum_values(enum_.name().clone(), values));
                }
            },
            _ => (),
        }
        // Create the new choice.
        let bool_choice: RcStr = "Bool".into();
        let name = RcStr::new("increment_".to_string() + counter);
        let def = ir::ChoiceDef::Enum(bool_choice.clone());
        let variables = counter_vars.iter().chain(iter_vars).cloned().collect();
        let args = ir::ChoiceArguments::new(variables, false, false);
        let incr_choice = ir::Choice::new(name.clone(), None, args, def);
        self.ir_desc.add_choice(incr_choice);
        // Constraint the boolean to follow the conditions.
        let vars = counter_vars.iter().chain(iter_vars).map(|x| x.0.clone()).collect();
        let incr_instance = ChoiceInstance { name: name.clone(), vars };
        let is_false = Condition::new_is_bool(incr_instance, false);
        let mut disjunctions = conditions.iter().map(|cond| {
            vec![cond.clone(), is_false.clone()]
        }).collect_vec();
        disjunctions.push(std::iter::once(is_false).chain(conditions).map(|mut cond| {
            cond.negate();
            cond
        }).collect());
        self.constraints.push(Constraint::new(all_vars_defs, disjunctions));
        // Generate the choice instance.
        let vars = (0..counter_vars.len()).map(ir::Variable::Arg)
            .chain((0..iter_vars.len()).map(ir::Variable::Forall)).collect();
        let true_value = std::iter::once("TRUE".into()).collect();
        let condition = ir::ValueSet::enum_values(bool_choice, true_value);
        (ir::ChoiceInstance { choice: name, vars }, condition)
    }

    /// Returns the `CounterVal` referencing a choice. Registers the UpdateCounter action
    /// so that the referencing counter is updated when the referenced counter is changed.
    fn counter_val_choice(&mut self,
                          counter: &ChoiceInstance,
                          caller_visibility: ir::CounterVisibility,
                          caller: RcStr,
                          incr: &ir::ChoiceInstance,
                          kind: ir::CounterKind,
                          num_caller_vars: usize,
                          var_map: &VarMap) -> (ir::CounterVal, ir::OnChangeAction) {
        // TODO(cleanup): do not force an ordering on counter declaration.
        let value_choice = self.ir_desc.get_choice(&counter.name);
        match *value_choice.choice_def() {
            ir::ChoiceDef::Counter { visibility, kind: value_kind, .. } => {
                // TODO(cleanup): allow mul of sums. The problem is that you can multiply
                // and/or divide by zero when doing this.
                use ir::CounterKind;
                assert!(!(kind == CounterKind::Mul && value_kind == CounterKind::Add));
                assert!(caller_visibility >= visibility,
                        "Counters cannot sum on counters that expose less information");
            },
            ir::ChoiceDef::Number { .. } => (),
            ir::ChoiceDef::Enum { .. } => panic!("Enum as a counter value"),
        };
        // Type the increment counter value in the calling counter context.
        let instance = counter.type_check(&self.ir_desc, var_map);
        let (forall_vars, set_constraints, adaptator) =
            self.ir_desc.adapt_env(var_map.env(), &instance);
        let caller_vars = (0..num_caller_vars).map(ir::Variable::Arg)
            .map(|v| adaptator.variable(v)).collect();
        // Create and register the action.
        let action = ir::ChoiceAction::UpdateCounter {
            counter: ir::ChoiceInstance { choice: caller, vars: caller_vars },
            incr: incr.adapt(&adaptator),
        };
        let update_action = ir::OnChangeAction { forall_vars, set_constraints, action };
        (ir::CounterVal::Choice(instance), update_action)
    }

    /// Typecheck and registers a trigger.
    fn register_trigger(&mut self, foralls: Vec<VarDef>, conditions: Vec<Condition>,
                        code: String) {
        trace!("defining trigger '{}'", code);
        // Type check the code and the conditions.
        let ref mut var_map = VarMap::default();
        let foralls = foralls.into_iter()
            .map(|def| var_map.decl_forall(&self.ir_desc, def)).collect();
        let mut inputs = Vec::new();
        let conditions = conditions.into_iter()
            .map(|c| c.type_check(&self.ir_desc, var_map, &mut inputs))
            .collect_vec();
        let code = type_check_code(RcStr::new(code), var_map);
        // Groups similiar inputs.
        let (inputs, input_adaptator) = dedup_inputs(inputs, &self.ir_desc);
        let conditions = conditions.into_iter()
            .map(|c| c.adapt(&input_adaptator)).collect_vec();
        // Adapt the trigger to the point of view of each inputs.
        let onchange_actions = inputs.iter().enumerate().map(|(pos, input)| {
            let (foralls, set_constraints, condition, adaptator) = ir::ChoiceCondition::new(
                &self.ir_desc, inputs.clone(), pos, &conditions, var_map.env());
            let code = code.adapt(&adaptator);
            (input.choice.clone(), foralls, set_constraints, condition, code)
        }).collect_vec();
        // Add the trigger to the IR.
        let trigger = ir::Trigger { foralls, inputs, conditions, code };
        let id = self.ir_desc.add_trigger(trigger);
        // Register the triggers to to be called when each input is modified.
        for (choice, forall_vars, set_constraints, condition, code) in onchange_actions {
            let action = ir::ChoiceAction::Trigger {
                id, condition, code, inverse_self_cond: false,
            };
            let on_change = ir::OnChangeAction { forall_vars, set_constraints, action };
            self.ir_desc.add_onchange(&choice, on_change);
        }
    }

    /// Creates an action to update the a counter when incr is modified.
    fn gen_incr_counter(&self, counter: &RcStr,
                        num_counter_args: usize,
                        var_map: &VarMap,
                        incr: &ir::ChoiceInstance,
                        value: ir::CounterVal) -> ir::OnChangeAction {
        // Adapt the environement to the point of view of the increment.
        let (forall_vars, set_constraints, adaptator) =
            self.ir_desc.adapt_env(var_map.env(), incr);
        let value = value.adapt(&adaptator);
        let counter_vars = (0..num_counter_args)
            .map(|i| adaptator.variable(ir::Variable::Arg(i))).collect();
        let choice = ir::ChoiceInstance { choice: counter.clone(), vars: counter_vars };
        let action =  ir::ChoiceAction::IncrCounter { choice, value };
        ir::OnChangeAction { forall_vars, set_constraints, action }
    }
}

/// A toplevel definition or constraint.
#[derive(Debug)]
pub enum Statement {
    IntegerDef {
        name: Spanned<String>,
        doc: Option<String>,
        variables: Vec<VarDef>,
        code: String,
    },
    /// Defines an enum.
    EnumDef {
        name: Spanned<String>,
        doc: Option<String>,
        variables: Vec<VarDef>,
        statements: Vec<EnumStatement>
    },
    TriggerDef {
        foralls: Vec<VarDef>,
        conditions: Vec<Condition>,
        code: String,
    },
    CounterDef {
        name: RcStr,
        doc: Option<String>,
        visibility: ir::CounterVisibility,
        vars: Vec<VarDef>,
        body: CounterBody,
    },
    SetDef {
        name: Spanned<String>,
        doc: Option<String>,
        arg: Option<VarDef>,
        superset: Option<SetRef>,
        disjoint: Vec<String>,
        keys: Vec<(ir::SetDefKey, Option<VarDef>, String)>,
        quotient: Option<Quotient>,
    },
    Require(Constraint),
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

/// A constraint that must be enforced by the IR.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Variables for which the conditions must be respected.
    pub forall_vars: Vec<VarDef>,
    /// Conjunction of disjuction of condition that must be respected.
    pub disjunctions: Vec<Vec<Condition>>,
    /// Indicates if the constraint should restrict fragile values.
    pub restrict_fragile: bool,
}

impl Constraint {
    /// Creates a new constraint.
    pub fn new(forall_vars: Vec<VarDef>, disjunctions: Vec<Vec<Condition>>) -> Self {
        Constraint { forall_vars, disjunctions, restrict_fragile: true }
    }

    /// Type check the constraint.
    fn type_check(self, ir_desc: &ir::IrDesc) -> Vec<TypedConstraint> {
        let mut var_map = VarMap::default();
        let sets = self.forall_vars.into_iter()
            .map(|v| var_map.decl_forall(ir_desc, v)).collect_vec();
        let restrict_fragile = self.restrict_fragile;
        self.disjunctions.into_iter().map(|disjuction| {
            let mut inputs = Vec::new();
            let conditions = disjuction.into_iter()
                .map(|x| x.type_check(ir_desc, &var_map, &mut inputs)).collect();
            TypedConstraint {
                vars: sets.clone(),
                restrict_fragile, inputs, conditions,
            }
        }).collect_vec()
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
    Value(String, Option<String>, Vec<Constraint>),
    /// Defines a set of possible decisions for the enum.
    Alias(String, Option<String>, Vec<String>, Vec<Constraint>),
    /// Specifies that the enum is symmetric.
    Symmetric,
    /// Specifies that the enum is antisymmetric and given the inverse function.
    AntiSymmetric(Vec<(String, String)>),
}

impl fmt::Display for EnumStatement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EnumStatement::Alias(name, ..) => write!(f, "{}", name),
            EnumStatement::Value(name, ..) => write!(f, "{}", name),
            EnumStatement::Symmetric => write!(f, "Symmetric"),
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
struct EnumStatements {
    /// The values the enum can take, with the atached documentation.
    values: HashMap<RcStr, Option<String>>,
    /// Aliases mapped to the corresponding documentation and value set.
    aliases: HashMap<RcStr, (Option<String>, HashSet<RcStr>)>,
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

/// A toplevel integer
#[derive(Clone, Debug)]
struct IntegerDef {
    name: String,
    doc: Option<String>,
    variables: Vec<VarDef>,
    code: String, // varmap, type_check_code
}

impl PartialEq for IntegerDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
 
/// A toplevel definition or constraint.
#[derive(Clone, Debug)]
struct EnumDef {
    name: String,
    doc: Option<String>,
    variables: Vec<VarDef>,
    statements: Vec<EnumStatement>
}

impl Default for EnumDef {
    fn default() -> EnumDef {
        EnumDef {
            name: String::default(),
            doc: None,
            variables: vec![],
            statements: vec![],
        }
    }
}

impl PartialEq for EnumDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}

#[derive(Debug)]
struct TriggerDef {
    foralls: Vec<VarDef>,
    conditions: Vec<Condition>,
    code: String,
}

#[derive(Clone, Debug)]
struct CounterDef {
    name: RcStr,
    doc: Option<String>,
    visibility: ir::CounterVisibility,
    vars: Vec<VarDef>,
    body: CounterBody,
}

impl PartialEq for CounterDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}

#[derive(Clone, Debug, PartialEq)]
enum ChoiceDef {
    CounterDef(CounterDef),
    EnumDef(EnumDef),
    IntegerDef(IntegerDef),
}

impl From<Statement> for ChoiceDef {
    fn from(stmt: Statement) -> Self {
        match stmt {
            Statement::CounterDef { name, doc, visibility, vars, body } => {
                ChoiceDef::CounterDef(CounterDef {
                    name, doc, visibility, vars, body
                })
            },
            Statement::EnumDef { name: Spanned {
                beg: _,
                end: _,
                data: name,
            }, doc, variables, statements } => {
                ChoiceDef::EnumDef(EnumDef {
                    name, doc, variables, statements
                })
            },
            Statement::IntegerDef { name: Spanned {
                beg: _,
                end: _,
                data: name,
            }, doc, variables, code } => {
                ChoiceDef::IntegerDef(IntegerDef {
                    name, doc, variables, code
                })
            },
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
struct SetDef {
    name: String,
    doc: Option<String>,
    arg: Option<VarDef>,
    superset: Option<SetRef>,
    disjoint: Vec<String>,
    keys: Vec<(ir::SetDefKey, Option<VarDef>, String)>,
    quotient: Option<Quotient>,
}

impl Default for SetDef {
    fn default() -> SetDef {
        SetDef {
            name: Default::default(),
            doc: None,
            arg: None,
            superset: None,
            disjoint: vec![],
            keys: vec![],
            quotient: None,
        }
    }
}

impl PartialEq for SetDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}

impl From<Statement> for Result<SetDef, TypeError> {
    fn from(stmt: Statement) -> Self {
        match stmt {
            Statement::SetDef {
                name: Spanned {
                    beg: _,
                    end: _,
                    data: name,
                }, doc, arg, superset, disjoint, keys, quotient
            } => {
                let set_def: SetDef = SetDef {
                    name, doc, arg, superset, disjoint, keys, quotient
                };
                Ok(set_def)
            },
            _ => unreachable!(),
        }
    }
}
