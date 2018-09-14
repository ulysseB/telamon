use super::*;
use ir::{self, Adaptable};

#[derive(Default)]
pub struct TypingContext {
    pub ir_desc: ir::IrDesc,
    pub set_defs: Vec<SetDef>,
    pub choice_defs: Vec<ChoiceDef>,
    pub triggers: Vec<TriggerDef>,
    pub constraints: Vec<Constraint>,
    pub checks: Vec<Check>,
}

impl TypingContext {
    /// Adds a statement to the typing context.
    pub fn add_statement(&mut self, statement: Statement) {
        match statement {
            stmt @ Statement::ChoiceDef(ChoiceDef::CounterDef(..)) => {
                self.choice_defs.push(ChoiceDef::from(stmt))
            }
            Statement::TriggerDef {
                foralls,
                conditions,
                code,
            } => self.triggers.push(TriggerDef {
                foralls: foralls,
                conditions: conditions,
                code: code,
            }),
            Statement::Require(constraint) => self.constraints.push(constraint),
            _ => {}
        }
    }

    /// Type-checks the statements in the correct order.
    pub fn finalize(mut self) -> (ir::IrDesc, Vec<TypedConstraint>) {
        for choice_def in std::mem::replace(&mut self.choice_defs, vec![]) {
            match choice_def {
                ChoiceDef::CounterDef(CounterDef {
                    name,
                    doc,
                    visibility,
                    vars,
                    body,
                }) => self.register_counter(name.data, doc, visibility, vars, body),
                _ => {}
            }
        }
        for trigger in std::mem::replace(&mut self.triggers, vec![]) {
            self.register_trigger(trigger.foralls, trigger.conditions, trigger.code);
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

    /// Registers a counter in the ir description.
    fn register_counter(
        &mut self,
        counter_name: RcStr,
        doc: Option<String>,
        visibility: ir::CounterVisibility,
        untyped_vars: Vec<VarDef>,
        body: CounterBody,
    ) {
        trace!("defining counter {}", counter_name);
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
            .map(|def| (def.name.clone(), var_map.decl_argument(&self.ir_desc, def)))
            .collect_vec();
        let base = type_check_code(RcStr::new(body.base), &var_map);
        // Generate the increment
        let iter_vars = body
            .iter_vars
            .into_iter()
            .map(|def| (def.name.clone(), var_map.decl_forall(&self.ir_desc, def)))
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
                );
                self.ir_desc.add_onchange(&counter.name, action);
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
        );
        self.ir_desc.add_onchange(&incr.choice, incr_counter);
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
        self.ir_desc.add_choice(counter_choice);
    }

    /// Creates a choice to store the increment condition of a counter. Returns the
    /// corresponding choice instance from the point of view of the counter and the
    /// condition on wich the counter must be incremented.
    fn gen_increment(
        &mut self,
        counter: &str,
        counter_vars: &[(RcStr, ir::Set)],
        iter_vars: &[(RcStr, ir::Set)],
        all_vars_defs: Vec<VarDef>,
        conditions: Vec<Condition>,
        var_map: &VarMap,
    ) -> (ir::ChoiceInstance, ir::ValueSet) {
        // TODO(cleanup): the choice the counter increment is based on must be declared
        // before the increment. It should not be the case.
        match conditions[..] {
            [Condition::Is {
                ref lhs,
                ref rhs,
                is,
            }] => {
                let incr = lhs.type_check(&self.ir_desc, var_map);
                // Ensure all forall values are usefull.
                let mut foralls = HashSet::default();
                for &v in &incr.vars {
                    if let ir::Variable::Forall(i) = v {
                        foralls.insert(i);
                    }
                }
                if foralls.len() == iter_vars.len() {
                    // Generate the increment condition.
                    let choice = self.ir_desc.get_choice(&incr.choice);
                    let enum_ = self
                        .ir_desc
                        .get_enum(choice.choice_def().as_enum().unwrap());
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
        self.ir_desc.add_choice(incr_choice);
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
            std::iter::once(is_false)
                .chain(conditions)
                .map(|mut cond| {
                    cond.negate();
                    cond
                }).collect(),
        );
        self.constraints
            .push(Constraint::new(all_vars_defs, disjunctions));
        // Generate the choice instance.
        let vars = (0..counter_vars.len())
            .map(ir::Variable::Arg)
            .chain((0..iter_vars.len()).map(ir::Variable::Forall))
            .collect();
        let true_value = std::iter::once("TRUE".into()).collect();
        let condition = ir::ValueSet::enum_values(bool_choice, true_value);
        (ir::ChoiceInstance { choice: name, vars }, condition)
    }

    /// Returns the `CounterVal` referencing a choice. Registers the UpdateCounter action
    /// so that the referencing counter is updated when the referenced counter is changed.
    fn counter_val_choice(
        &mut self,
        counter: &ChoiceInstance,
        caller_visibility: ir::CounterVisibility,
        caller: RcStr,
        incr: &ir::ChoiceInstance,
        incr_condition: &ir::ValueSet,
        kind: ir::CounterKind,
        num_caller_vars: usize,
        var_map: &VarMap,
    ) -> (ir::CounterVal, ir::OnChangeAction) {
        // TODO(cleanup): do not force an ordering on counter declaration.
        let value_choice = self.ir_desc.get_choice(&counter.name);
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
        let instance = counter.type_check(&self.ir_desc, var_map);
        let (forall_vars, set_constraints, adaptator) =
            self.ir_desc.adapt_env(var_map.env(), &instance);
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

    /// Typecheck and registers a trigger.
    fn register_trigger(
        &mut self,
        foralls: Vec<VarDef>,
        conditions: Vec<Condition>,
        code: String,
    ) {
        trace!("defining trigger '{}'", code);
        // Type check the code and the conditions.
        let ref mut var_map = VarMap::default();
        let foralls = foralls
            .into_iter()
            .map(|def| var_map.decl_forall(&self.ir_desc, def))
            .collect();
        let mut inputs = Vec::new();
        let conditions = conditions
            .into_iter()
            .map(|c| c.type_check(&self.ir_desc, var_map, &mut inputs))
            .collect_vec();
        let code = type_check_code(RcStr::new(code), var_map);
        // Groups similiar inputs.
        let (inputs, input_adaptator) = dedup_inputs(inputs, &self.ir_desc);
        let conditions = conditions
            .into_iter()
            .map(|c| c.adapt(&input_adaptator))
            .collect_vec();
        // Adapt the trigger to the point of view of each inputs.
        let onchange_actions = inputs
            .iter()
            .enumerate()
            .map(|(pos, input)| {
                let (foralls, set_constraints, condition, adaptator) =
                    ir::ChoiceCondition::new(
                        &self.ir_desc,
                        inputs.clone(),
                        pos,
                        &conditions,
                        var_map.env(),
                    );
                let code = code.adapt(&adaptator);
                (
                    input.choice.clone(),
                    foralls,
                    set_constraints,
                    condition,
                    code,
                )
            }).collect_vec();
        // Add the trigger to the IR.
        let trigger = ir::Trigger {
            foralls,
            inputs,
            conditions,
            code,
        };
        let id = self.ir_desc.add_trigger(trigger);
        // Register the triggers to to be called when each input is modified.
        for (choice, forall_vars, set_constraints, condition, code) in onchange_actions {
            let action = ir::ChoiceAction::Trigger {
                id,
                condition,
                code,
                inverse_self_cond: false,
            };
            let on_change = ir::OnChangeAction {
                forall_vars,
                set_constraints,
                action,
            };
            self.ir_desc.add_onchange(&choice, on_change);
        }
    }

    /// Creates an action to update the a counter when incr is modified.
    fn gen_incr_counter(
        &self,
        counter: &RcStr,
        num_counter_args: usize,
        var_map: &VarMap,
        incr: &ir::ChoiceInstance,
        incr_condition: &ir::ValueSet,
        value: ir::CounterVal,
    ) -> ir::OnChangeAction {
        // Adapt the environement to the point of view of the increment.
        let (forall_vars, set_constraints, adaptator) =
            self.ir_desc.adapt_env(var_map.env(), incr);
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
}
