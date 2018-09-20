use std::iter;

use ast::constrain::Constraint;
use ast::context::CheckerContext;
use ast::error::TypeError;
use ast::typing_context::TypingContext;
use ast::{
    type_check_code, type_check_enum_values, ChoiceDef, ChoiceInstance, Condition,
    CounterBody, CounterVal, HashSet, VarDef, VarMap,
};
use ir::{self, Adaptable};
use itertools::Itertools;
use lexer::Spanned;
use utils::RcStr;

#[derive(Clone, Debug)]
pub struct CounterDef {
    pub name: Spanned<RcStr>,
    pub doc: Option<String>,
    pub visibility: ir::CounterVisibility,
    pub vars: Vec<VarDef>,
    pub body: CounterBody,
}

impl CounterDef {
    /// Creates an action to update the a counter when incr is modified.
    fn gen_incr_counter(
        &self,
        counter: &RcStr,
        num_counter_args: usize,
        var_map: &VarMap,
        incr: &ir::ChoiceInstance,
        incr_condition: &ir::ValueSet,
        value: ir::CounterVal,
        tc: &mut TypingContext,
    ) -> ir::OnChangeAction {
        // Adapt the environement to the point of view of the increment.
        let (forall_vars, set_constraints, adaptator) =
            tc.ir_desc.adapt_env(var_map.env(), incr);
        let value = value.adapt(&adaptator);
        let counter_vars = (0..num_counter_args)
            .map(|i| adaptator.variable(ir::Variable::Arg(i)))
            .collect();
        let action = ir::ChoiceAction::IncrCounter {
            counter: ir::ChoiceInstance {
                choice: counter.clone(),
                vars: counter_vars,
            },
            value: value.adapt(&adaptator),
            incr_condition: incr_condition.adapt(&adaptator),
        };
        ir::OnChangeAction {
            forall_vars,
            set_constraints,
            action,
        }
    }

    /// Returns the `CounterVal` referencing a choice. Registers the UpdateCounter action
    /// so that the referencing counter is updated when the referenced counter is changed.
    fn counter_val_choice(
        &self,
        counter: &ChoiceInstance,
        caller_visibility: ir::CounterVisibility,
        caller: RcStr,
        incr: &ir::ChoiceInstance,
        incr_condition: &ir::ValueSet,
        kind: ir::CounterKind,
        num_caller_vars: usize,
        var_map: &VarMap,
        tc: &mut TypingContext,
    ) -> (ir::CounterVal, ir::OnChangeAction) {
        // TODO(cleanup): do not force an ordering on counter declaration.
        let value_choice = tc.ir_desc.get_choice(&counter.name);
        match *value_choice.choice_def() {
            ir::ChoiceDef::Counter {
                visibility,
                kind: value_kind,
                ..
            } => {
                // TODO(cleanup): allow mul of sums. The problem is that you can multiply
                // and/or divide by zero when doing this.
                use ir::CounterKind;
                assert!(!(kind == CounterKind::Mul && value_kind == CounterKind::Add));
                assert!(
                    caller_visibility >= visibility,
                    "Counters cannot sum on counters that expose less information"
                );
            }
            ir::ChoiceDef::Number { .. } => (),
            ir::ChoiceDef::Enum { .. } => panic!("Enum as a counter value"),
        };
        // Type the increment counter value in the calling counter context.
        let instance = counter.type_check(&tc.ir_desc, var_map);
        let (forall_vars, set_constraints, adaptator) =
            tc.ir_desc.adapt_env(var_map.env(), &instance);
        let caller_vars = (0..num_caller_vars)
            .map(ir::Variable::Arg)
            .map(|v| adaptator.variable(v))
            .collect();
        // Create and register the action.
        let action = ir::ChoiceAction::UpdateCounter {
            counter: ir::ChoiceInstance {
                choice: caller,
                vars: caller_vars,
            },
            incr: incr.adapt(&adaptator),
            incr_condition: incr_condition.adapt(&adaptator),
        };
        let update_action = ir::OnChangeAction {
            forall_vars,
            set_constraints,
            action,
        };
        (ir::CounterVal::Choice(instance), update_action)
    }

    /// Creates a choice to store the increment condition of a counter. Returns the
    /// corresponding choice instance from the point of view of the counter and the
    /// condition on wich the counter must be incremented.
    fn gen_increment(
        &self,
        counter: &str,
        counter_vars: &[(RcStr, ir::Set)],
        iter_vars: &[(RcStr, ir::Set)],
        all_vars_defs: Vec<VarDef>,
        conditions: Vec<Condition>,
        var_map: &VarMap,
        tc: &mut TypingContext,
    ) -> (ir::ChoiceInstance, ir::ValueSet) {
        // TODO(cleanup): the choice the counter increment is based on must be declared
        // before the increment. It should not be the case.
        match conditions[..] {
            [Condition::Is {
                ref lhs,
                ref rhs,
                is,
            }] => {
                let incr = lhs.type_check(&tc.ir_desc, var_map);
                // Ensure all forall values are usefull.
                let mut foralls = HashSet::default();
                for &v in &incr.vars {
                    if let ir::Variable::Forall(i) = v {
                        foralls.insert(i);
                    }
                }
                if foralls.len() == iter_vars.len() {
                    // Generate the increment condition.
                    let choice = tc.ir_desc.get_choice(&incr.choice);
                    let enum_ =
                        tc.ir_desc.get_enum(choice.choice_def().as_enum().unwrap());
                    let values = type_check_enum_values(enum_, rhs.clone());
                    let values = if is {
                        values
                    } else {
                        enum_
                            .values()
                            .keys()
                            .filter(|&v| !values.contains(v))
                            .cloned()
                            .collect()
                    };
                    return (
                        incr,
                        ir::ValueSet::enum_values(enum_.name().clone(), values),
                    );
                }
            }
            _ => (),
        }
        // Create the new choice.
        let bool_choice: RcStr = "Bool".into();
        let name = RcStr::new("increment_".to_string() + counter);
        let def = ir::ChoiceDef::Enum(bool_choice.clone());
        let variables = counter_vars.iter().chain(iter_vars).cloned().collect();
        let args = ir::ChoiceArguments::new(variables, false, false);
        let incr_choice = ir::Choice::new(name.clone(), None, args, def);
        tc.ir_desc.add_choice(incr_choice);
        // Constraint the boolean to follow the conditions.
        let vars = counter_vars
            .iter()
            .chain(iter_vars)
            .map(|x| x.0.clone())
            .collect();
        let incr_instance = ChoiceInstance {
            name: name.clone(),
            vars,
        };
        let is_false = Condition::new_is_bool(incr_instance, false);
        let mut disjunctions = conditions
            .iter()
            .map(|cond| vec![cond.clone(), is_false.clone()])
            .collect_vec();
        disjunctions.push(
            iter::once(is_false)
                .chain(conditions)
                .map(|mut cond| {
                    cond.negate();
                    cond
                })
                .collect(),
        );
        tc.constraints
            .push(Constraint::new(all_vars_defs, disjunctions));
        // Generate the choice instance.
        let vars = (0..counter_vars.len())
            .map(ir::Variable::Arg)
            .chain((0..iter_vars.len()).map(ir::Variable::Forall))
            .collect();
        let true_value = iter::once("TRUE".into()).collect();
        let condition = ir::ValueSet::enum_values(bool_choice, true_value);
        (ir::ChoiceInstance { choice: name, vars }, condition)
    }

    /// Registers a counter in the ir description.
    pub fn register_counter(
        &self,
        counter_name: RcStr,
        doc: Option<String>,
        visibility: ir::CounterVisibility,
        untyped_vars: Vec<VarDef>,
        body: CounterBody,
        tc: &mut TypingContext,
    ) {
        trace!("defining counter {}", counter_name);
        println!("defining counter {}", counter_name);

        let mut var_map = VarMap::default();
        // Type-check the base.
        let kind = body.kind;
        let all_var_defs = untyped_vars
            .iter()
            .chain(&body.iter_vars)
            .cloned()
            .collect();
        let vars = untyped_vars
            .into_iter()
            .map(|def| (def.name.clone(), var_map.decl_argument(&tc.ir_desc, def)))
            .collect_vec();
        let base = type_check_code(RcStr::new(body.base), &var_map);
        // Generate the increment
        let iter_vars = body
            .iter_vars
            .into_iter()
            .map(|def| (def.name.clone(), var_map.decl_forall(&tc.ir_desc, def)))
            .collect_vec();
        let doc = doc.map(RcStr::new);
        let (incr, incr_condition) = self.gen_increment(
            &counter_name,
            vars.iter()
                .cloned()
                .map(|(n, s)| (n.data, s))
                .collect::<Vec<_>>()
                .as_slice(),
            iter_vars
                .iter()
                .cloned()
                .map(|(n, s)| (n.data, s))
                .collect::<Vec<_>>()
                .as_slice(),
            all_var_defs,
            body.conditions,
            &var_map,
            tc,
        );
        // Type check the value.
        let value = match body.value {
            CounterVal::Code(code) => {
                ir::CounterVal::Code(type_check_code(RcStr::new(code), &var_map))
            }
            CounterVal::Choice(counter) => {
                let counter_name = counter_name.clone();
                let (value, action) = self.counter_val_choice(
                    &counter,
                    visibility,
                    counter_name,
                    &incr,
                    &incr_condition,
                    kind,
                    vars.len(),
                    &var_map,
                    tc,
                );
                tc.ir_desc.add_onchange(&counter.name, action);
                value
            }
        };
        let incr_counter = self.gen_incr_counter(
            &counter_name,
            vars.len(),
            &var_map,
            &incr,
            &incr_condition,
            value.clone(),
            tc,
        );
        tc.ir_desc.add_onchange(&incr.choice, incr_counter);
        // Register the counter choices.
        let incr_iter = iter_vars.iter().map(|p| p.1.clone()).collect_vec();
        let counter_def = ir::ChoiceDef::Counter {
            incr_iter,
            kind,
            value,
            incr,
            incr_condition,
            visibility,
            base,
        };
        let counter_args = ir::ChoiceArguments::new(
            vars.into_iter().map(|(n, s)| (n.data, s)).collect(),
            false,
            false,
        );
        let mut counter_choice =
            ir::Choice::new(counter_name, doc, counter_args, counter_def);
        // Filter the counter itself after an update, because the filter actually acts on
        // the increments and depends on the counter value.
        let filter_self = ir::OnChangeAction {
            forall_vars: vec![],
            set_constraints: ir::SetConstraints::default(),
            action: ir::ChoiceAction::FilterSelf,
        };
        counter_choice.add_onchange(filter_self);
        tc.ir_desc.add_choice(counter_choice);
    }

    pub fn define(
        self,
        context: &mut CheckerContext,
        tc: &mut TypingContext,
    ) -> Result<(), TypeError> {
        tc.choice_defs.push(ChoiceDef::CounterDef(self));
        Ok(())
    }
}

impl PartialEq for CounterDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
