//! The constraint description for the ir.
use std;

use indexmap::IndexMap;
use itertools::Itertools;
use utils::*;

mod adaptator;
mod choice;
mod filter;
mod set;

pub use self::adaptator::*;
pub use self::choice::*;
pub use self::filter::*;
pub use self::set::*;

use serde::Serialize;

/// Describes the choices that constitute the IR.
#[derive(Debug)]
pub struct IrDesc {
    choices: IndexMap<RcStr, Choice>,
    enums: IndexMap<RcStr, Enum>,
    set_defs: IndexMap<RcStr, (std::rc::Rc<SetDef>, set::OnNewObject)>,
    triggers: Vec<Trigger>,
}

impl IrDesc {
    /// Adds a `Choice` to the IR description.
    pub fn add_choice(&mut self, choice: Choice) {
        assert!(self.choices.insert(choice.name().clone(), choice).is_none());
    }

    /// Adds an `Enum` to the IR desctiption.
    pub fn add_enum(&mut self, enum_: Enum) {
        assert!(self.enums.insert(enum_.name().clone(), enum_).is_none());
    }

    /// List the choice definitions.
    pub fn choices<'a>(&'a self) -> impl Iterator<Item = &'a Choice> {
        self.choices.values()
    }

    /// List the enum definitions.
    pub fn enums<'a>(&'a self) -> impl Iterator<Item = &'a Enum> {
        self.enums.values()
    }

    /// Returns the enum with the given name.
    pub fn get_enum<'a>(&'a self, name: &str) -> &'a Enum {
        &self.enums[name]
    }

    /// Returns the choice with the given name.
    pub fn get_choice<'a>(&'a self, name: &str) -> &'a Choice {
        unwrap!(self.choices.get(name), "choice '{}' is not declared", name)
    }

    /// Iterates over all the sets.
    pub fn set_defs(
        &self,
    ) -> impl Iterator<Item = &(std::rc::Rc<SetDef>, OnNewObject)> + '_ {
        self.set_defs.values()
    }

    /// Register a set definition.
    pub fn add_set_def(&mut self, def: std::rc::Rc<SetDef>) {
        let name = def.name().clone();
        assert!(self
            .set_defs
            .insert(name, (def, OnNewObject::default()))
            .is_none());
    }

    /// Returns the set definition associated with a name.
    pub fn get_set_def<'a>(&'a self, name: &str) -> &'a std::rc::Rc<SetDef> {
        &self
            .set_defs
            .get(name)
            .unwrap_or_else(|| panic!("Undefined set {}", name))
            .0
    }

    /// Adds a filter to a choice.
    pub fn add_filter(
        &mut self,
        choice: RcStr,
        filter: Filter,
        forall_vars: Vec<Set>,
        set_constraints: SetConstraints,
    ) {
        // Register the filter and add the `OnChangeAction`s.
        let filter_ref = if filter.inputs.is_empty() {
            if let SubFilter::Rules(rules) = filter.rules {
                FilterRef::Inline(rules)
            } else {
                panic!()
            }
        } else {
            let inputs = filter.inputs.clone();
            let filter_id = self.choices.get_mut(&choice).unwrap().add_filter(filter);
            // Ensure the filter is called when other choices are restricted.
            for input in &inputs {
                self.register_filter_call(
                    &choice,
                    filter_id,
                    input,
                    &forall_vars,
                    &set_constraints,
                );
            }
            // Call the filter on initialization.
            let arguments = self
                .get_choice(&choice)
                .arguments()
                .iter()
                .enumerate()
                .map(|(i, _)| Variable::Arg(i))
                .chain((0..forall_vars.len()).map(|i| Variable::Forall(i)))
                .collect();
            FilterRef::Function {
                choice: choice.clone(),
                id: filter_id,
                args: arguments,
            }
        };
        self.register_filter_on_new_objs(
            &filter_ref,
            &choice,
            &forall_vars,
            &set_constraints,
        );
        // Add the init call.
        let filter = FilterCall {
            forall_vars,
            filter_ref,
        };
        let filter_action = FilterAction {
            filter,
            set_constraints,
        };
        self.choices
            .get_mut(&choice)
            .unwrap()
            .add_filter_action(filter_action);
    }

    /// Registers to run `filter` when a new object is added to a set in `foralls`.
    fn register_filter_on_new_objs(
        &mut self,
        filter: &FilterRef,
        choice: &RcStr,
        foralls: &[Set],
        set_constraints: &SetConstraints,
    ) {
        let choice = &self.choices[choice];
        let mut choice_args = choice.arguments().sets().cloned().collect_vec();
        // Apply the set constraints the the choice arguments.
        for (var, set_constraint) in set_constraints.constraints() {
            if let Variable::Arg(i) = *var {
                choice_args[i] = set_constraint.clone();
            } else {
                panic!("expected an argument variable");
            }
        }
        for (i, mut set) in foralls.iter().cloned().enumerate() {
            let var = Variable::Forall(i);
            // If the set if a reverse set, it has no proper definition so we cannot
            // register it. Instead we re-reverse it to obtain the initial set.
            let arg_set;
            let reverse = if (&set).def().name().is_empty() {
                let (rev_arg_set, rev_set) = set
                    .reverse(Variable::Forall(i), (&set).def().arg().unwrap())
                    .unwrap();
                arg_set = Some(rev_arg_set);
                set = rev_set;
                true
            } else {
                arg_set = (&set).arg().map(|var| match var {
                    Variable::Forall(id) => foralls[id].clone(),
                    Variable::Arg(id) => choice_args[id].clone(),
                });
                false
            };
            // Constraint the set of the argument.
            let (new_foralls, adaptator) =
                adapt_to_var_context(&choice_args, foralls, var, reverse);
            let set_constraints = arg_set
                .filter(|arg_set| !(&set).def().arg().unwrap().is_subset_of(arg_set))
                .map(|arg_set| {
                    let arg_set = arg_set.adapt(&adaptator);
                    SetConstraints::new(vec![(Variable::Arg(1), arg_set)])
                })
                .unwrap_or_default();
            let remote_call = RemoteFilterCall {
                choice: ChoiceInstance::self_choice(choice).adapt(&adaptator),
                filter: FilterCall {
                    // `forall_vars` are supposed to iterate on multiple filters for the
                    // same choice.  Our foralls iterate on multiple choices so we pass
                    // them externally.
                    forall_vars: vec![],
                    filter_ref: filter.adapt(&adaptator),
                },
            };
            let set_def = self.set_defs.get_mut((&set).def().name()).unwrap();
            set_def
                .1
                .filter
                .push((new_foralls, set_constraints, remote_call));
        }
    }

    pub fn add_onchange(&mut self, choice: &str, action: OnChangeAction) {
        let inverse = if action.applies_to_symmetric()
            && self.get_choice(choice).arguments().is_symmetric()
        {
            Some(action.inverse(self))
        } else {
            None
        };
        let choice = unwrap!(self.choices.get_mut(choice));
        choice.add_onchange(action);
        if let Some(action) = inverse {
            choice.add_onchange(action);
        }
    }

    /// Registers an filter call action for a filter on a choice.
    fn register_filter_call(
        &mut self,
        filtered: &RcStr,
        filter_id: usize,
        changed: &ChoiceInstance,
        foralls: &[Set],
        set_constraints: &SetConstraints,
    ) {
        let env = self
            .get_choice(&filtered)
            .arguments()
            .sets()
            .enumerate()
            .map(|(i, set)| (Variable::Arg(i), set))
            .chain(
                foralls
                    .iter()
                    .enumerate()
                    .map(|(i, set)| (Variable::Forall(i), set)),
            )
            .map(|(v, set)| (v, set_constraints.find_set(v).unwrap_or(set).clone()))
            .collect::<FxHashMap<_, _>>();
        // If the changed choice is symmetric, the inverse filter should also be called.
        if self.get_choice(&changed.choice).arguments().is_symmetric() {
            let vars = vec![changed.vars[1], changed.vars[0]];
            let ref changed = ChoiceInstance {
                choice: changed.choice.clone(),
                vars,
            };
            let action = self.gen_filter_call_action(
                filtered.clone(),
                filter_id,
                changed,
                env.clone(),
                foralls.len(),
            );
            self.add_onchange(&changed.choice, action);
        }
        let action = self.gen_filter_call_action(
            filtered.clone(),
            filter_id,
            &changed,
            env,
            foralls.len(),
        );
        self.add_onchange(&changed.choice, action);
    }

    /// Generates an `OnChangeAction`.
    fn gen_filter_call_action(
        &self,
        filtered: RcStr,
        id: usize,
        changed: &ChoiceInstance,
        env: FxHashMap<Variable, Set>,
        num_foralls: usize,
    ) -> OnChangeAction {
        // TODO(cc_perf): no need to iterate on the full domain for symmetric choices.
        let (choice_foralls, filter_foralls, set_constraints, adaptator) =
            self.adapt_env_ext(env, changed);
        // Generate the action.
        let num_args = self.get_choice(&filtered).arguments().len();
        let args = (0..num_args)
            .map(Variable::Arg)
            .chain((0..num_foralls).map(Variable::Forall))
            .map(|v| adaptator.variable(v))
            .collect();
        let filter_ref = FilterRef::Function {
            choice: filtered.clone(),
            id,
            args,
        };
        let filtered_args = (0..num_args).map(|i| adaptator.variable(Variable::Arg(i)));
        let remote_filter = RemoteFilterCall {
            choice: ChoiceInstance {
                choice: filtered,
                vars: filtered_args.collect(),
            },
            filter: FilterCall {
                forall_vars: filter_foralls,
                filter_ref,
            },
        };
        OnChangeAction {
            forall_vars: choice_foralls,
            set_constraints,
            action: ChoiceAction::RemoteFilter(remote_filter),
        }
    }

    /// Adds a trigger to a choice.
    pub fn add_trigger(&mut self, trigger: Trigger) -> usize {
        let id = self.triggers.len();
        self.triggers.push(trigger);
        id
    }

    /// Iterates on the triggers.
    pub fn triggers(&self) -> impl Iterator<Item = &Trigger> {
        self.triggers.iter()
    }

    /// Generates the list of sets to iterate and to constraints to iterate on the given
    /// context, but from the point of view of the given choice instance.
    pub fn adapt_env(
        &self,
        vars: FxHashMap<Variable, Set>,
        choice_instance: &ChoiceInstance,
    ) -> (Vec<Set>, SetConstraints, Adaptator) {
        let (mut arg_foralls, other_foralls, set_constraints, adaptator) =
            self.adapt_env_ext(vars, choice_instance);
        arg_foralls.extend(other_foralls);
        (arg_foralls, set_constraints, adaptator)
    }

    /// Generates the foralls and the set constraints to iterate on the given environment,
    /// from the point of view of the given choice. Returns the foralls issued from arguments
    /// in a different list than the foralls issued from foralls in the original environment.
    pub fn adapt_env_ext(
        &self,
        mut vars: FxHashMap<Variable, Set>,
        choice_instance: &ChoiceInstance,
    ) -> (Vec<Set>, Vec<Set>, SetConstraints, Adaptator) {
        let mut arg_foralls = Vec::new();
        let mut other_foralls = Vec::new();
        let mut set_constraints = Vec::new();
        let mut adaptator = Adaptator::default();
        // Set the mapping of the target choice arguments.
        let target = self.get_choice(&choice_instance.choice);
        let src_vars = choice_instance
            .vars
            .iter()
            .zip_eq(target.arguments().sets());
        for (arg_id, (&mapped_var, given_set)) in src_vars.enumerate() {
            let expected_set = vars.remove(&mapped_var).unwrap();
            adaptator.set_variable(mapped_var, Variable::Arg(arg_id));
            if &expected_set != given_set {
                set_constraints.push((Variable::Arg(arg_id), expected_set.clone()));
            }
        }
        // Add the remaining variables as foralls. We keep the foralls in the order they were
        // given, so the sets are still defined after their parameters.
        let vars = vars.into_iter().sorted_by_key(|x| x.0);
        for (forall_id, (mapped_var, set)) in vars.into_iter().enumerate() {
            adaptator.set_variable(mapped_var, Variable::Forall(forall_id));
            match mapped_var {
                Variable::Arg(_) => arg_foralls.push(set),
                Variable::Forall(_) => other_foralls.push(set),
            }
        }
        // Adapt the set constraints to the new environement.
        for set in set_constraints
            .iter_mut()
            .map(|x| &mut x.1)
            .chain(&mut arg_foralls)
            .chain(&mut other_foralls)
        {
            *set = set.adapt(&adaptator);
        }
        // Reverse the set constraints when the set parameter is defined in the foralls.
        // TODO(cleanup): make the reversing code readable
        // TODO(cleanup): reimplemente `drain_filter` when stable rust will be ready.
        for (var, ref mut set) in set_constraints.iter_mut() {
            // Reverse the set if its parameter if defined after the constraints.
            if let Some(Variable::Forall(forall_id)) = (&*set).arg() {
                // Assign the reverse set to foralls.
                let forall = if forall_id < arg_foralls.len() {
                    &mut arg_foralls[forall_id]
                } else {
                    &mut other_foralls[forall_id - arg_foralls.len()]
                };
                let (superset, reverse_set) = set.reverse(*var, &forall).unwrap();
                *forall = reverse_set;
                // Use the superset as the constraint is enforced by the forall.
                assert!((&superset).arg().is_none());
                *set = superset;
            }
        }
        set_constraints.retain(|&(var, ref set)| {
            if let Some(Variable::Forall(_)) = (&*set).arg() {
                if let Variable::Arg(arg_id) = var {
                    let given_set = target.arguments().get(arg_id).1;
                    !given_set.is_subset_of_def(set)
                } else {
                    panic!()
                }
            } else {
                true
            }
        });
        (
            arg_foralls,
            other_foralls,
            SetConstraints::new(set_constraints),
            adaptator,
        )
    }
}

impl Default for IrDesc {
    fn default() -> Self {
        let mut ir_desc = IrDesc {
            choices: IndexMap::default(),
            enums: IndexMap::default(),
            set_defs: IndexMap::default(),
            triggers: Vec::new(),
        };
        let mut bool_enum = Enum::new("Bool".into(), None, None);
        bool_enum.add_value("TRUE".into(), None);
        bool_enum.add_value("FALSE".into(), None);
        ir_desc.add_enum(bool_enum);
        ir_desc
    }
}

/// Adapt the environement to the point of view of a variable. In practice, this means, mapping
/// `var` to `ir::Variable::Arg(0)`, its argument to `ir::Variable::Arg(1)` (if any) and other sets
/// to forall variables. If reverse is `true`, Arg(0) and Arg(1) are inversed.
fn adapt_to_var_context(
    args: &[Set],
    foralls: &[Set],
    var: Variable,
    reverse: bool,
) -> (Vec<Set>, Adaptator) {
    let mut adaptator = Adaptator::default();
    adaptator.set_variable(var, Variable::Arg(if reverse { 1 } else { 0 }));
    let arg_var = match var {
        Variable::Arg(i) => (&args[i]).arg(),
        Variable::Forall(i) => (&foralls[i]).arg(),
    };
    if let Some(arg) = arg_var {
        adaptator.set_variable(arg, Variable::Arg(if reverse { 0 } else { 1 }));
    } else {
        assert!(!reverse);
    }
    let mut forall_index = 0;
    let args = args
        .iter()
        .enumerate()
        .map(|(i, set)| (Variable::Arg(i), set));
    let foralls = foralls
        .iter()
        .enumerate()
        .map(|(i, set)| (Variable::Forall(i), set));
    let new_foralls = args
        .chain(foralls)
        .filter(|&(id, _)| id != var && Some(id) != arg_var)
        .map(|(old_id, set)| {
            adaptator.set_variable(old_id, Variable::Forall(forall_index));
            forall_index += 1;
            set.adapt(&adaptator)
        })
        .collect();
    (new_foralls, adaptator)
}

/// Indicates whether a counter sums or adds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[repr(C)]
pub enum CounterKind {
    Add,
    Mul,
}

impl CounterKind {
    /// Returns the neutral element of the operand.
    pub fn zero(&self) -> u32 {
        match *self {
            CounterKind::Add => 0,
            CounterKind::Mul => 1,
        }
    }
}

/// A choice that can take a few predefined values.
#[derive(Debug, Clone)]
pub struct Enum {
    name: RcStr,
    doc: Option<RcStr>,
    values: IndexMap<RcStr, Option<String>>,
    aliases: IndexMap<RcStr, (FxHashSet<RcStr>, Option<String>)>,
    inverse: Option<Vec<(RcStr, RcStr)>>,
}

impl Enum {
    /// Creates a new enum definition.
    pub fn new(
        name: RcStr,
        doc: Option<RcStr>,
        inverse: Option<Vec<(RcStr, RcStr)>>,
    ) -> Self {
        Enum {
            name: name,
            doc: doc,
            inverse: inverse,
            values: IndexMap::default(),
            aliases: IndexMap::default(),
        }
    }

    /// Returns the name of the enum.
    pub fn name(&self) -> &RcStr {
        &self.name
    }

    /// Adds a possible value to the enum.
    pub fn add_value(&mut self, name: RcStr, doc: Option<String>) {
        assert!(!self.aliases.contains_key(&name));
        assert!(self.values.insert(name, doc).is_none());
    }

    /// Adds an alias to the enum possible values.
    pub fn add_alias(
        &mut self,
        name: RcStr,
        vals: FxHashSet<RcStr>,
        doc: Option<String>,
    ) {
        assert!(!self.values.contains_key(&name));
        assert!(self.aliases.insert(name, (vals, doc)).is_none());
    }

    /// Lists the aliases.
    pub fn aliases(&self) -> &IndexMap<RcStr, (FxHashSet<RcStr>, Option<String>)> {
        &self.aliases
    }

    /// Returns the documentation associated with the enum.
    pub fn doc(&self) -> Option<&str> {
        self.doc.as_ref().map(|x| x as &str)
    }

    /// Returns the values the enum can take, and their associated comment.
    pub fn values(&self) -> &IndexMap<RcStr, Option<String>> {
        &self.values
    }

    /// Replaces aliases by the corresponding values.
    pub fn expand<IT: IntoIterator<Item = RcStr>>(&self, set: IT) -> FxHashSet<RcStr> {
        let mut new_set = FxHashSet::default();
        for alias in set {
            if let Some(&(ref alias_set, _)) = self.aliases.get(&alias) {
                new_set.extend(alias_set.iter().cloned());
            } else {
                new_set.insert(alias);
            }
        }
        new_set
    }

    /// Inverse an antisymmetric value.
    pub fn inverse<'a>(&'a self, value: &'a RcStr) -> &'a RcStr {
        if let Some(ref mapping) = self.inverse {
            for &(ref lhs, ref rhs) in mapping {
                if lhs == value {
                    return rhs;
                }
                if rhs == value {
                    return lhs;
                }
            }
        }
        value
    }

    /// Returns the mapping to apply to obtain the symmetric of a value.
    pub fn inverse_mapping(&self) -> Option<&[(RcStr, RcStr)]> {
        self.inverse.as_ref().map(|x| &x[..])
    }
}

/// A piece of host code called when a list of conditions are met.
#[derive(Clone, Debug)]
pub struct Trigger {
    pub foralls: Vec<Set>,
    pub inputs: Vec<ChoiceInstance>,
    pub conditions: Vec<Condition>,
    pub code: Code,
}

/// A context in which a sub-filter can be evaluated.
#[cfg(test)]
pub mod test {
    use super::*;
    use itertools::Itertools;
    use std;

    pub use super::filter::test::*;

    /// A context in which a sub-filter can be evaluated.
    pub struct EvalContext<'a> {
        pub ir_desc: &'a IrDesc,
        pub enum_: &'a Enum,
        pub var_sets: FxHashMap<Variable, Set>,
        pub inputs_def: &'a [ChoiceInstance],
        pub input_values: Vec<ValueSet>,
        pub static_conds: FxHashMap<&'a StaticCond, bool>,
    }

    impl<'a> EvalContext<'a> {
        /// Evaluate a condition.
        pub fn eval_cond(&self, cond: &Condition) -> bool {
            let mapping = self.input_values.iter().enumerate().collect();
            cond.evaluate(&self.inputs_def, &mapping, self.ir_desc)
                .is_true()
        }

        /// Filters the list of valid values according to the given rule.
        pub fn eval_rule(&self, rule: &'a Rule, out: &mut ValueSet) {
            let mapping = self.input_values.iter().enumerate().collect();
            let set = rule.alternatives.instantiate(&mapping, self.ir_desc);
            self.eval_rule_aux(&rule.conditions, &rule.set_constraints, set, out);
        }

        pub fn eval_rule_aux(
            &self,
            conditions: &'a [Condition],
            set_constraints: &'a SetConstraints,
            alternatives: ValueSet,
            valid_values: &mut ValueSet,
        ) {
            for &(var, ref t) in set_constraints.constraints() {
                if !(&self.var_sets[&var]).is_subset_of(t) {
                    return;
                }
            }
            if conditions.iter().all(|c| self.eval_cond(c)) {
                valid_values.intersect(alternatives);
            }
        }

        /// Returns the list of valid values according the given rules.
        pub fn eval_rules<IT>(&self, rules: IT) -> ValueSet
        where
            IT: IntoIterator<Item = &'a Rule>,
        {
            let enum_values = self.enum_.values().keys().cloned().collect();
            let enum_name = self.enum_.name().clone();
            let mut value_set = ValueSet::enum_values(enum_name, enum_values);
            for rule in rules {
                self.eval_rule(rule, &mut value_set);
            }
            value_set
        }

        /// Returns the list of values autorized by the given subfilter.
        pub fn eval_subfilter(&self, filter: &'a SubFilter) -> ValueSet {
            match *filter {
                SubFilter::Rules(ref rules) => self.eval_rules(rules),
                SubFilter::Switch { switch, ref cases } => {
                    let t = ValueType::Enum(self.enum_.name().clone());
                    let mut value_set = ValueSet::empty(&t);
                    for &(ref guard, ref filter) in cases {
                        if self.input_values[switch].is(guard).maybe_true() {
                            value_set.extend(self.eval_subfilter(filter));
                        }
                    }
                    value_set
                }
            }
        }

        /// Returns an iterator over all possible totaly specified contexts.
        pub fn iter_contexts(
            ir_desc: &'a IrDesc,
            enum_: &'a Enum,
            inputs_def: &'a [ChoiceInstance],
            static_conds: &'a [StaticCond],
        ) -> impl Iterator<Item = EvalContext<'a>> + 'a {
            let values = inputs_def
                .iter()
                .map(
                    |input| match ir_desc.get_choice(&input.choice).value_type() {
                        ValueType::Enum(ref name) => {
                            ir_desc.get_enum(name).values().keys().collect_vec()
                        }
                        _ => panic!(),
                    },
                )
                .collect_vec();
            let num_values = values.iter().map(|x| x.len()).collect_vec();
            NDRange::new(&num_values)
                .map(|indexes| {
                    values
                        .iter()
                        .zip_eq(indexes)
                        .map(|(x, y)| x[y])
                        .collect_vec()
                })
                .flat_map(|values| {
                    let cond_vec = static_conds.iter().map(|_| 2).collect_vec();
                    NDRange::new(&cond_vec)
                        .map(|cond_values| {
                            let cond_values = cond_values.into_iter().map(|x| match x {
                                0 => false,
                                1 => true,
                                _ => panic!(),
                            });
                            let input_values = values
                                .iter()
                                .map(|&v| {
                                    let v = std::iter::once(v.clone()).collect();
                                    ValueSet::enum_values(enum_.name().clone(), v)
                                })
                                .collect();
                            EvalContext {
                                ir_desc: ir_desc,
                                enum_: enum_,
                                var_sets: FxHashMap::default(),
                                inputs_def: inputs_def,
                                input_values: input_values,
                                static_conds: static_conds
                                    .iter()
                                    .zip_eq(cond_values)
                                    .collect(),
                            }
                        })
                        .collect_vec()
                })
                .collect_vec()
                .into_iter()
        }
    }

    /// A condition with a value fixed before exploration.
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    pub enum StaticCond {
        Code(Code),
    }

    /// Generates an enum.
    pub fn gen_enum(name: &str, num_values: usize, ir_desc: &mut IrDesc) {
        let mut e = Enum::new(RcStr::new(format!("Enum{}", name)), None, None);
        for i in 0..num_values {
            e.add_value(RcStr::new(format!("{}_{}", name, i)), None);
        }
        let choice_name = RcStr::new(format!("enum_{}", name.to_lowercase()));
        let choice_type = ChoiceDef::Enum(e.name().clone());
        let args = ChoiceArguments::Plain { vars: vec![] };
        ir_desc.add_choice(Choice::new(choice_name, None, args, choice_type));
        ir_desc.add_enum(e);
    }
}
