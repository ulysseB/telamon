//! Describe decisions that must be specified.
use itertools::{Either, Itertools};
use ir::{self, Adaptable};
use std;
use utils::*;

pub fn dummy_choice() -> Choice {
    let def = ChoiceDef::Enum("Bool".into());
    let args = ChoiceArguments::Plain { vars: vec![] };
    Choice::new("DUMMY".into(), None, args, def)
}

/// A decision to specify.
#[derive(Debug)]
pub struct Choice {
    name: RcStr,
    doc: Option<RcStr>,
    arguments: ChoiceArguments,
    choice_def: ChoiceDef,
    on_change: Vec<OnChangeAction>,
    filter_actions: Vec<FilterAction>,
    filters: Vec<ir::Filter>,
    no_propagate_values: ir::ValueSet,
}

impl Choice {
    /// Creates a new `Choice`.
    pub fn new(name: RcStr, doc: Option<RcStr>, arguments: ChoiceArguments,
               def: ChoiceDef) -> Self {
        let value_type = def.value_type();
        Choice {
            name: name,
            doc: doc,
            arguments: arguments,
            choice_def: def,
            on_change: Vec::new(),
            filter_actions: Vec::new(),
            filters: Vec::new(),
            no_propagate_values: ir::ValueSet::empty(value_type),
        }
    }

    /// Returns the name of the choice, in snake_case.
    pub fn name(&self) -> &RcStr { &self.name }

    /// Returns the documentation associated with the `Choice`.
    pub fn doc(&self) -> Option<&str> { self.doc.as_ref().map(|x| x as &str) }

    /// Returns the parameters for which the `Choice` is defined.
    pub fn arguments(&self) -> &ChoiceArguments { &self.arguments }

    /// Returns the type representing the values the `Choice` can take.
    pub fn value_type(&self) -> ValueType { self.choice_def.value_type() }

    /// Returns the definition of the `Choice.
    pub fn choice_def(&self) -> &ChoiceDef { &self.choice_def }

    /// Returns the actions to perform when the `Choice` is constrained.
    pub fn on_change(&self) -> std::slice::Iter<OnChangeAction> {
        self.on_change.iter()
    }

    /// Returns the actions to run to get the valid alternatives of the choice.
    pub fn filter_actions(&self) -> std::slice::Iter<FilterAction> {
        self.filter_actions.iter()
    }

    /// Returns the filters operating on the `Choice`.
    pub fn filters(&self) -> std::slice::Iter<ir::Filter> { self.filters.iter() }

    /// Adds a filter to run on initialization.
    pub fn add_filter_action(&mut self, action: FilterAction) {
        self.filter_actions.push(action)
    }

    /// Adds an action to perform when the `Choice` is constrained.
    pub fn add_onchange(&mut self, action: OnChangeAction) {
        // TODO(cc_perf): normalize and merge forall vars when possible
        if self.arguments().is_symmetric() && action.applies_to_symmetric() {
            self.on_change.push(action.inverse());
        }
        self.on_change.push(action);
    }

    /// Adds a filter to the `Choice`, returns an ID to indentify it.
    pub fn add_filter(&mut self, filter: ir::Filter) -> usize {
        self.filters.push(filter);
        self.filters.len() - 1
    }

    /// Returns the values that should not be automatically restricted by filters.
    pub fn fragile_values(&self) -> &ir::ValueSet { &self.no_propagate_values }

    /// Extends the list of values that should not be automatically propagated by filters.
    pub fn add_fragile_values(&mut self, values: ir::ValueSet) {
        self.no_propagate_values.extend(values);
    }
}

/// Defines the parameters for which the `Choice` is defined.
#[derive(Debug)]
pub enum ChoiceArguments {
    /// The `Choice` is defined for all comibnation of variables of the given sets
    /// Each variable can only appear once.
    Plain { vars: Vec<(RcStr, ir::Set)> },
    /// The `Choice` is defined on a triangular space. The rests is obtained by symmetry.
    Symmetric { names: [RcStr; 2], t: ir::Set, inverse: bool },
}

impl ChoiceArguments {
    /// Creates a new `ChoiceArguments`.
    pub fn new(mut vars: Vec<(RcStr, ir::Set)>, symmetric: bool, inverse: bool) -> Self {
        if symmetric {
            assert_eq!(vars.len(), 2);
            let (rhs, t1) = vars.pop().unwrap();
            let (lhs, t0) = vars.pop().unwrap();
            assert_eq!(t0, t1);
            ChoiceArguments::Symmetric { names: [lhs, rhs], t: t0, inverse: inverse }
        } else {
            assert!(!inverse);
            ChoiceArguments::Plain { vars: vars }
        }
    }

    /// Returns the name of the arguments.
    pub fn names<'a>(&'a self) -> impl Iterator<Item=&'a RcStr> {
        match *self {
            ChoiceArguments::Plain { ref vars } =>
                Either::Left(vars.iter().map(|x| &x.0)),
            ChoiceArguments::Symmetric { ref names, .. } =>
                Either::Right(names.iter()),
        }
    }

    /// Returns the sets of the arguments.
    pub fn sets<'a>(&'a self) -> impl Iterator<Item=&'a ir::Set> + 'a {
        match *self {
            ChoiceArguments::Plain { ref vars } =>
                Either::Left(vars.iter().map(|x| &x.1)),
            ChoiceArguments::Symmetric { ref t, ..} =>
                Either::Right(vec![t, t].into_iter()),
        }
    }

    /// Returns the name and set of the argument at the given position.
    pub fn get(&self, index: usize) -> (&RcStr, &ir::Set) {
        match *self {
            ChoiceArguments::Plain { ref vars } => (&vars[index].0, &vars[index].1),
            ChoiceArguments::Symmetric { ref names, ref t, .. } => (&names[index], t),
        }
    }

    /// Iterates over the arguments, with their sets and names.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item=(&'a RcStr, &'a ir::Set)> + 'a {
        self.names().zip_eq(self.sets())
    }

    /// Indicates if the arguments iteration domain is triangular.
    pub fn is_symmetric(&self) -> bool {
        if let ChoiceArguments::Symmetric { .. } = *self { true } else { false }
    }

    /// Returns the number of arguments.
    pub fn len(&self) -> usize {
        match *self {
            ChoiceArguments::Plain { ref vars } => vars.len(),
            ChoiceArguments::Symmetric { .. } => 2,
        }
    }
}

/// Specifies how the `Choice` is defined.
#[derive(Clone, Debug)]
pub enum ChoiceDef {
    /// The `Choice` can take a small set of predefined values.
    Enum(RcStr),
    /// An integer abstracted by an interval.
    Counter {
        kind: ir::CounterKind,
        value: CounterVal,
        incr_iter: Vec<ir::Set>,
        incr: ir::ChoiceInstance,
        incr_condition: ir::ValueSet,
        visibility: CounterVisibility,
        base: ir::Code,
    },
    /// The `Choice` can take a small set of dynamically defined numeric values.
    Number { universe: ir::Code },
}

/// Indicates how a counter exposes how its maximum value. The variants are ordered by
/// increasing amount of information available.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[repr(C)]
pub enum CounterVisibility {
    /// Only the minimal value is computed and stored.
    NoMax,
    /// Both the min and max are stored, but only the min is exposed.
    HiddenMax,
    /// Both the min and the max value are exposed.
    Full,
}

impl ChoiceDef {
    /// Returns the underlying value type.
    pub fn value_type(&self) -> ValueType {
        match *self {
            ChoiceDef::Enum(ref name) => ValueType::Enum(name.clone()),
            ChoiceDef::Counter { visibility: CounterVisibility::NoMax, .. } =>
                ValueType::HalfRange,
            ChoiceDef::Counter { .. } => ValueType::Range,
            ChoiceDef::Number { ref universe, .. } =>
                ValueType::NumericSet(universe.clone()),
        }
    }

    /// Indicates if the choice is a counter.
    pub fn is_counter(&self) -> bool {
        if let ChoiceDef::Counter { .. } = *self { true } else { false }
    }

    /// Returns the name of the `Enum` the `Choice` is based on.
    pub fn as_enum(&self) -> Option<&RcStr> {
        if let ChoiceDef::Enum(ref name) = *self { Some(name) } else { None }
    }

    /// Indicates the comparison operators that can be applied to the decision.
    pub fn is_valid_operator(&self, op: ir::CmpOp) -> bool {
        match *self {
            ChoiceDef::Enum(..) => op == ir::CmpOp::Eq || op == ir::CmpOp::Neq,
            ChoiceDef::Counter { visibility: CounterVisibility::Full, .. } => true,
            ChoiceDef::Counter { .. } => op == ir::CmpOp::Lt || op == ir::CmpOp::Leq,
            ChoiceDef::Number { .. } => unimplemented!(), // FIXME
        }
    }
}

/// The value of the increments of a counter.
#[derive(Clone, Debug)]
pub enum CounterVal { Code(ir::Code), Counter(ir::ChoiceInstance) }

impl Adaptable for CounterVal {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        match *self {
            CounterVal::Code(ref code) => CounterVal::Code(code.adapt(adaptator)),
            CounterVal::Counter(ref choice_instance) =>
                CounterVal::Counter(choice_instance.adapt(adaptator)),
        }
    }
}

impl ValueType {
    /// Returns the full type, instead of a the trimmed one.
    pub fn full_type(self) -> Self {
        if self == ValueType::HalfRange { ValueType::Range } else { self }
    }
}

/// Specifies the type of the values a choice can take.
#[derive(Clone, Debug, PartialEq, Eq)]
// FIXME: unimplemented must adapt the value type everywhere
pub enum ValueType { Enum(RcStr), Range, HalfRange, NumericSet(ir::Code) }

impl Adaptable for ValueType {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        match *self {
            ref t @ ValueType::Enum(..) |
            ref t @ ValueType::Range |
            ref t @ ValueType::HalfRange => t.clone(),
            ValueType::NumericSet(ref uni) => ValueType::NumericSet(uni.adapt(adaptator)),
        }
    }
}

/// A call to a filter.
#[derive(Clone, Debug)]
pub struct FilterCall {
    pub forall_vars: Vec<ir::Set>,
    pub filter_ref: FilterRef,
}

impl Adaptable for FilterCall {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        FilterCall {
            forall_vars: self.forall_vars.adapt(adaptator),
            filter_ref: self.filter_ref.adapt(adaptator),
        }
    }
}

/// References a filter to call.
#[derive(Clone, Debug)]
pub enum FilterRef {
    Inline(Vec<ir::Rule>),
    Local { id: usize, args: Vec<ir::Variable> },
    Remote { choice: RcStr, id: usize, args: Vec<ir::Variable> },
}

impl Adaptable for FilterRef {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        use self::FilterRef::*;
        match *self {
            Inline(ref rules) => Inline(rules.adapt(adaptator)),
            Local { id, ref args } => Local { id, args: args.adapt(adaptator) },
            Remote { ref choice, id, ref args } =>
                Remote { choice: choice.clone(), id, args: args.adapt(adaptator) },
        }
    }
}

/// An action to perform when the choice is restricted.
#[derive(Clone, Debug)]
pub struct OnChangeAction {
    pub forall_vars: Vec<ir::Set>,
    pub set_constraints: ir::SetConstraints,
    pub action: ChoiceAction,
}

impl OnChangeAction {
    /// Indicates if the action sould also be registered for the symmetric of the choice,
    /// if applicable.
    fn applies_to_symmetric(&self) -> bool { self.action.applies_to_symmetric() }

    /// Returns the action for the symmetric of the choice.
    fn inverse(&self) -> Self {
        let ref mut adaptator = ir::Adaptator::default();
        adaptator.set_variable(ir::Variable::Arg(0), ir::Variable::Arg(1));
        adaptator.set_variable(ir::Variable::Arg(1), ir::Variable::Arg(0));
        let mut out = self.adapt(adaptator);
        out.action.inverse_self();
        out
    }
}

impl Adaptable for OnChangeAction {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        OnChangeAction {
            forall_vars: self.forall_vars.adapt(adaptator),
            set_constraints: self.set_constraints.adapt(adaptator),
            action: self.action.adapt(adaptator),
        }
    }
}

/// An action to perform,
#[derive(Clone, Debug)]
pub enum ChoiceAction {
    FilterSelf,
    Filter { choice: ir::ChoiceInstance, filter: FilterCall },
    IncrCounter { choice: ir::ChoiceInstance, value: ir::CounterVal, },
    UpdateCounter { counter: ir::ChoiceInstance, incr: ir::ChoiceInstance, to_half: bool },
    Trigger {
        id: usize,
        condition: ir::ChoiceCondition,
        code: ir::Code,
        inverse_self_cond: bool,
    },
}

impl ChoiceAction {
    /// Indicates if the action sould also be registered for the symmetric of the choice,
    /// if applicable.
    fn applies_to_symmetric(&self) -> bool {
        match *self {
            // Filters for the symmetric are already produced by constraint translation.
            ChoiceAction::FilterSelf | ChoiceAction::Filter { .. } => false,
            _ => true,
        }
    }

    /// Returns the list of variables to allocate.
    pub fn variables<'a>(&'a self) -> Box<Iterator<Item=&'a ir::Set> + 'a> {
        match *self {
            ChoiceAction::Filter { ref filter, .. } =>
                Box::new(filter.forall_vars.iter()) as Box<_>,
            ChoiceAction::Trigger { .. } |
            ChoiceAction::IncrCounter { .. } |
            ChoiceAction::UpdateCounter { .. } |
            ChoiceAction::FilterSelf => Box::new(std::iter::empty()) as Box<_>,
        }
    }

    /// Returns the list of inputs used by the action.
    pub fn inputs(&self) -> &[ir::ChoiceInstance] {
        match *self {
            ChoiceAction::Trigger { ref condition, .. } => &condition.inputs,
            _ => &[],
        }
    }

    /// Inverse references to the value of the choice the action is registered in.
    pub fn inverse_self(&mut self) {
        match *self {
            ChoiceAction::Trigger { ref mut inverse_self_cond, .. } =>
                *inverse_self_cond = !*inverse_self_cond,
            _ => (),
        }
    }
}

impl Adaptable for ChoiceAction {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        use self::ChoiceAction::*;
        match *self {
            FilterSelf => FilterSelf,
            Filter { ref choice, ref filter } => Filter {
                choice: choice.adapt(adaptator),
                filter: filter.adapt(adaptator),
            },
            IncrCounter { ref choice, ref value } => IncrCounter {
                choice: choice.adapt(adaptator),
                value: value.adapt(adaptator),
            },
            UpdateCounter { ref counter, ref incr, to_half } => UpdateCounter {
                counter: counter.adapt(adaptator),
                incr: incr.adapt(adaptator),
                to_half
            },
            Trigger { id, ref condition, ref code, inverse_self_cond } => Trigger {
                // FIXME: must inverse self condition
                condition: condition.adapt(adaptator),
                code: code.adapt(adaptator),
                id, inverse_self_cond,
            },
        }
    }
}

/// A condition from the point of view of a choice.
#[derive(Clone, Debug)]
pub struct ChoiceCondition {
    pub inputs: Vec<ir::ChoiceInstance>,
    pub self_condition: ir::ValueSet,
    pub others_conditions: Vec<ir::Condition>,
}

impl ChoiceCondition {
    /// Adapt the list of conditions to be from the point of view of the given choice.
   pub fn new(ir_desc: &ir::IrDesc,
              mut inputs: Vec<ir::ChoiceInstance>, self_id: usize,
              conditions: &[ir::Condition],
              env: HashMap<ir::Variable, ir::Set>)
        -> (Vec<ir::Set>, ir::SetConstraints, Self, ir::Adaptator)
    {
        // Create the new evironement.
        let (foralls, set_constraints, mut adaptator) =
            ir_desc.adapt_env(env, &inputs[self_id]);
        let choice = inputs.swap_remove(self_id).choice;
        adaptator.set_input(inputs.len(), self_id);
        let inputs = inputs.into_iter().map(|i| i.adapt(&adaptator)).collect();
        // Extract the constraints on the considered input.
        let choice = ir_desc.get_choice(&choice);
        let mut self_condition = ir::ValueSet::empty(choice.value_type());
        let others_conditions = conditions.iter().flat_map(|condition| {
            let alternatives = condition.alternatives_of(self_id, choice, ir_desc);
            if let Some(alternatives) = alternatives {
                self_condition.extend(alternatives.adapt(&adaptator));
                None
            } else { Some(condition.adapt(&adaptator)) }
        }).collect();
        let condition = ChoiceCondition { inputs, self_condition, others_conditions };
        (foralls, set_constraints, condition, adaptator)
    }
}

impl Adaptable for ChoiceCondition {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        ChoiceCondition {
            inputs: self.inputs.adapt(&adaptator),
            self_condition: self.self_condition.adapt(&adaptator),
            others_conditions: self.others_conditions.adapt(&adaptator),
        }
    }
}

/// Restricts the set of valid values.
#[derive(Clone, Debug)]
pub struct FilterAction {
    pub set_constraints: ir::SetConstraints,
    pub filter: FilterCall,
}
