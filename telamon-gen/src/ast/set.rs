use std::ops::Deref;
use std::iter::once;
use std::mem;

use super::{
    ir,
    Quotient, SetRef, VarDef, VarMap, Check, Condition, CounterVal,
    print, ChoiceInstance, CounterBody
};
use super::constrain::Constraint;
use super::context::CheckerContext;
use super::choice::{CounterDef, ChoiceDef};
use super::typing_context::TypingContext;
use super::trigger::TriggerDef;
use super::error::{TypeError, Hint};

use utils::{RcStr, HashMap};
use lexer::Spanned;
use indexmap::IndexMap;
use itertools::Itertools;

#[derive(Debug, Clone)]
pub struct SetDef {
    pub name: Spanned<String>,
    pub doc: Option<String>,
    pub arg: Option<VarDef>,
    pub superset: Option<SetRef>,
    pub disjoint: Vec<String>,
    pub keys: Vec<(Spanned<ir::SetDefKey>, Option<VarDef>, String)>,
    pub quotient: Option<Quotient>,
}

impl SetDef {
    /// This checks that thereisn't any keys doublon.
    fn check_redefinition_key(&self) -> Result<(), TypeError> {
        let mut hash: HashMap<_, Spanned<()>> = HashMap::default();
        for (key, ..) in self.keys.iter() {
            if let Some(pre) = hash.insert(key.data.to_owned(), key.with_data(())) {
                Err(TypeError::Redefinition {
                    object_kind: pre.with_data(Hint::Set),
                    object_name: key.with_data(key.data.to_string()),
                })?;
            }
        }
        Ok(())
    }

    /// This checks the presence of keys ItemType, IdType, ItemGetter,
    /// IdGetter and Iter. When there is a superset, this checks the
    /// presence of FromSuperset keyword.
    fn check_missing_entry(&self) -> Result<(), TypeError> {
        let keys = self
            .keys
            .iter()
            .map(|(k, _, _)| k.data)
            .collect::<Vec<ir::SetDefKey>>();

        for ref key in ir::SetDefKey::REQUIRED.iter() {
            if !keys.contains(&key) {
                Err(TypeError::MissingEntry {
                    object_name: self.name.data.to_owned(),
                    object_field: self.name.with_data(key.to_string()),
                })?;
            }
        }
        if self.superset.is_some() && !keys.contains(&&ir::SetDefKey::FromSuperset) {
            Err(TypeError::MissingEntry {
                object_name: self.name.data.to_owned(),
                object_field: self
                    .name
                    .with_data(ir::SetDefKey::FromSuperset.to_string()),
            })?;
        }
        Ok(())
    }

    fn check_undefined_reverse_subset(
        &self,
        context: &CheckerContext,
    ) -> Result<(), TypeError> {
        if self.arg.is_some() {
            if let Some(Some(reverse)) = self
                .keys
                .iter()
                .find(|(k, _, _)| k.data == ir::SetDefKey::Reverse)
                .map(|(_, ss, _)| ss)
            {
                if !context.check_set_define(&reverse.set) {
                    let name: &String = reverse.set.name.deref();

                    Err(TypeError::Undefined {
                        object_name: self.name.with_data(name.to_owned()),
                    })?;
                }
            }
        }
        Ok(())
    }

    /// This checks if the argument is defined in the context.
    fn check_undefined_argument(
        &self,
        context: &CheckerContext,
    ) -> Result<(), TypeError> {
        if let Some(VarDef {
            name: _,
            set: ref subset,
        }) = self.arg
        {
            if !context.check_set_define(subset) {
                let name: &String = subset.name.deref();

                Err(TypeError::Undefined {
                    object_name: self.name.with_data(name.to_owned()),
                })?;
            }
        }
        Ok(())
    }

    /// This checks if the superset is defined in the context.
    fn check_undefined_superset(
        &self,
        context: &CheckerContext,
    ) -> Result<(), TypeError> {
        if let Some(ref subset) = self.superset {
            if !context.check_set_define(subset) {
                let name: &String = subset.name.deref();

                Err(TypeError::Undefined {
                    object_name: self.name.with_data(name.to_owned()),
                })?;
            }
        }
        Ok(())
    }

    /// This checks if the disjoint is defined in the context.
    fn check_undefined_disjoint(
        &self,
        context: &CheckerContext,
    ) -> Result<(), TypeError> {
        for dis in self.disjoint.iter() {
            if !context.check_set_define(&SetRef {
                name: RcStr::new(dis.to_owned()),
                var: None,
            }) {
                Err(TypeError::Undefined {
                    object_name: self.name.with_data(dis.to_owned()),
                })?;
            }
        }
        Ok(())
    }

    /// Type checks the declare's condition.
    pub fn declare(&self, context: &mut CheckerContext) -> Result<(), TypeError> {
        context.declare_set(self.name.to_owned())
    }

    /// Creates a boolean choice that indicates if an object represents a givne class.
    fn create_repr_choice(
        &self,
        name: RcStr,
        set: &ir::SetDef,
        item_name: Spanned<RcStr>,
        tc: &mut TypingContext,
    ) {
        let arg = self.arg.clone();
        let bool_str: RcStr = "Bool".into();
        let def = ir::ChoiceDef::Enum(bool_str.clone());
        let mut vars = Vec::new();
        if let Some(arg) = self.arg.as_ref() {
            vars.push((arg.name.clone(), set.arg().unwrap().clone()));
        }
        vars.push((item_name, set.superset().unwrap().clone()));
        let args = ir::ChoiceArguments::new(
            vars.into_iter().map(|(n, s)| (n.data, s)).collect(),
            false,
            false,
        );
        let mut repr = ir::Choice::new(name, None, args, def);
        let false_value_set = once("FALSE".into()).collect();
        repr.add_fragile_values(ir::ValueSet::enum_values(bool_str, false_value_set));
        tc.ir_desc.add_choice(repr);
    }

    /// Creates a counter for the number of objects that can represent another object in
    /// a quotient set. Returns the name of the counter.
    fn create_repr_counter(
        &self,
        set_name: RcStr,
        repr_name: &str,
        item_name: RcStr,
        vars: Vec<VarDef>,
        equiv_choice_name: RcStr,
        equiv_values: Vec<RcStr>,
        tc: &mut TypingContext,
    ) -> RcStr {
        // Create the increment condition
        tc.checks.push(Check::IsSymmetric {
            choice: equiv_choice_name.clone(),
            values: equiv_values.clone(),
        });
        let arg = self.arg.clone();
        let rhs_name = RcStr::new(format!("{}_repr", item_name));
        let rhs_set = SetRef {
            name: set_name,
            var: arg.as_ref().map(|d| d.name.data.clone()),
        };
        let equiv_choice = ChoiceInstance {
            name: equiv_choice_name,
            vars: vec![item_name, rhs_name.clone()],
        };
        let condition = Condition::Is {
            lhs: equiv_choice,
            rhs: equiv_values,
            is: true,
        };
        // Create the counter.
        let name = RcStr::new(format!("{}_class_counter", repr_name));
        let visibility = ir::CounterVisibility::HiddenMax;
        let body = CounterBody {
            base: "0".to_string(),
            conditions: vec![condition],
            iter_vars: vec![VarDef {
                name: Spanned {
                    data: rhs_name,
                    beg: Default::default(),
                    end: Default::default(),
                },
                set: rhs_set,
            }],
            kind: ir::CounterKind::Add,
            value: CounterVal::Code("1".to_string()),
        };
        tc.choice_defs.push(ChoiceDef::CounterDef(CounterDef {
            name: Spanned {
                data: name.clone(),
                ..Default::default()
            },
            doc: None,
            visibility,
            vars,
            body,
        }));
        name
    }

    /// Creates the choices that implement the quotient set.
    fn create_quotient(&self, set: &ir::SetDef, tc: &mut TypingContext) {
        let quotient = self.quotient.clone().unwrap();

        // assert!(set.attributes().contains_key(&ir::SetDefKey::AddToSet));
        let repr_name = quotient.representant;
        // Create decisions to back the quotient set
        self.create_repr_choice(repr_name.clone(), set, quotient.item.name.clone(), tc);
        let item_name = quotient.item.name.clone();
        let arg_name = self.arg.as_ref().map(|x| x.name.clone());
        let forall_vars = self
            .arg
            .clone()
            .into_iter()
            .chain(once(quotient.item))
            .collect_vec();
        let counter_name = self.create_repr_counter(
            set.name().clone(),
            &repr_name,
            item_name.data.clone(),
            forall_vars.clone(),
            RcStr::new(quotient.equiv_relation.0),
            quotient.equiv_relation.1,
            tc,
        );
        // Generate the code that set an item as representant.
        let trigger_code = print::add_to_quotient(
            set,
            &repr_name,
            &counter_name,
            &item_name.data,
            &arg_name.clone().map(|n| n.data),
        );
        // Constraint the representative value.
        let forall_names = forall_vars.iter().map(|x| x.name.clone()).collect_vec();
        let repr_instance = ChoiceInstance {
            name: repr_name,
            vars: forall_names
                .iter()
                .map(|n| n.data.clone())
                .collect::<Vec<_>>(),
        };
        let counter_instance = ChoiceInstance {
            name: counter_name,
            vars: forall_names
                .iter()
                .map(|n| n.data.clone())
                .collect::<Vec<_>>(),
        };
        let not_repr = Condition::new_is_bool(repr_instance.clone(), false);
        let counter_leq_zero = Condition::CmpCode {
            lhs: counter_instance,
            rhs: "0".into(),
            op: ir::CmpOp::Leq,
        };
        // Add the constraints `repr is FALSE || dividend is true` and
        // `repr is FALSE || counter <= 0`.
        let mut disjunctions = quotient
            .conditions
            .iter()
            .map(|c| vec![not_repr.clone(), c.clone()])
            .collect_vec();
        disjunctions.push(vec![not_repr, counter_leq_zero.clone()]);
        let repr_constraints = Constraint::new(forall_vars.clone(), disjunctions);
        tc.constraints.push(repr_constraints);
        // Add the constraint `repr is TRUE || counter > 0 || dividend is false`.
        let repr_true = Condition::new_is_bool(repr_instance, true);
        let mut counter_gt_zero = counter_leq_zero.clone();
        counter_gt_zero.negate();
        let mut repr_true_conditions = vec![repr_true.clone(), counter_gt_zero];
        for mut cond in quotient.conditions.iter().cloned() {
            cond.negate();
            repr_true_conditions.push(cond);
        }
        tc.constraints.push(Constraint {
            forall_vars: forall_vars.clone(),
            disjunctions: vec![repr_true_conditions],
            restrict_fragile: false,
        });
        // Add the constraint `item in set => repr is TRUE`.
        let quotient_item_def = VarDef {
            name: item_name,
            set: SetRef {
                name: set.name().clone(),
                var: arg_name.map(|n| n.data),
            },
        };
        let item_in_set_foralls = self
            .arg
            .clone()
            .into_iter()
            .chain(once(quotient_item_def))
            .collect();
        tc.constraints
            .push(Constraint::new(item_in_set_foralls, vec![vec![repr_true]]));
        // Generate the trigger that sets the repr to TRUE and add the item to the set.
        let mut trigger_conds = quotient.conditions;
        trigger_conds.push(counter_leq_zero);
        tc.triggers.push(TriggerDef {
            foralls: forall_vars,
            conditions: trigger_conds,
            code: trigger_code,
        });
    }

    /// Type checks the define's condition.
    pub fn define(
        self,
        context: &CheckerContext,
        tc: &mut TypingContext,
    ) -> Result<(), TypeError> {
        self.check_undefined_argument(context)?;
        self.check_undefined_superset(context)?;
        self.check_undefined_disjoint(context)?;
        self.check_undefined_reverse_subset(context)?;
        self.check_redefinition_key()?;
        self.check_missing_entry()?;

        trace!("defining set {}", self.name);
        let mut var_map = VarMap::default();
        let arg_name = self
            .arg
            .as_ref()
            .map(|var| "$".to_string() + &var.name.data);
        let arg = self
            .arg
            .clone()
            .map(|arg| var_map.decl_argument(&tc.ir_desc, arg));
        let superset = self
            .superset
            .as_ref()
            .map(|set| set.type_check(&tc.ir_desc, &var_map));
        for disjoint in &self.disjoint {
            tc.ir_desc.get_set_def(disjoint);
        }
        let mut keymap: IndexMap<ir::SetDefKey, String> = IndexMap::default();
        let mut reverse = None;
        for (key, var, mut value) in self
            .keys
            .iter()
            .map(|(k, v, s)| (k.data, v, s))
            .collect::<Vec<_>>()
        {
            let mut v = value.to_owned();
            let mut env = key.env();

            // Add the set argument to the environement.
            if let Some(ref arg_name) = arg_name {
                // TODO(cleanup): use ir::Code to avoid using a dummy name.
                // Currently, we may have a collision on the $var name.
                if key.is_arg_in_env() {
                    v = v.replace(arg_name, "$var");
                    env.push("var");
                }
            }
            // Handle the optional forall.
            if key == ir::SetDefKey::Reverse {
                let var_def = var.as_ref().unwrap();
                let var_name = "$".to_string() + &var_def.name.data;
                v = v.replace(&var_name, "$var");
                env.push("var");
            } else {
                assert!(var.is_none());
            }
            if key == ir::SetDefKey::Reverse {
                let set = var
                    .clone()
                    .unwrap()
                    .set
                    .type_check(&tc.ir_desc, &VarMap::default());
                assert!(superset.as_ref().unwrap().is_subset_of_def(&set));
                assert!(
                    mem::replace(&mut reverse, Some((set, v.to_owned()))).is_none()
                );
            } else {
                assert!(keymap.insert(key, v).is_none());
            }
        }
        let def = ir::SetDef::new(
            self.name.data.to_owned(),
            arg,
            superset,
            reverse,
            keymap,
            self.disjoint.to_owned(),
        );
        if let Some(ref quotient) = self.quotient {
            self.create_quotient(&def, tc);
        }
        tc.ir_desc.add_set_def(def);

        tc.set_defs.push(self);
        Ok(())
    }
}

impl PartialEq for SetDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
