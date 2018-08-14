//! Filter on choices.
use ir::{self, SetRef, Adaptable};
use itertools::Itertools;
use std;
use std::collections::BTreeSet;
use utils::*;

/// Filters the set valid values.
#[derive(Debug)]
pub struct Filter {
    /// The variables on which the filter depends.
    pub arguments: Vec<ir::Set>,
    /// The choices that the filter depends on.
    pub inputs: Vec<ChoiceInstance>,
    /// The filter rules.
    pub rules: SubFilter,
}

/// Filters the set of valid values, given some inputs.
#[derive(Debug, Clone)]
pub enum SubFilter {
    /// Enumerate the possible values that a input can take, and lists the possible values
    /// for each.
    Switch { switch: usize, cases: Vec<(ValueSet, SubFilter)> },
    /// Applies a set of negative rules to filter the possible values.
    Rules(Vec<Rule>),
}

/// Specifies a conditional restriction on the set of valid values.
#[derive(Debug, Clone)]
pub struct Rule {
    /// The conditions that must be true for the rule to trigger.
    pub conditions: Vec<Condition>,
    /// The values allowed for the enum if the rule is triggered.
    pub alternatives: ValueSet,
    /// The condition on subsets for the rule to apply.
    pub set_constraints: SetConstraints,
}

impl Rule {
    /// Instantiates the rule for a given assignment of the inputs.
    pub fn instantiate(&self, inputs: &[ChoiceInstance],
                       input_mapping: &HashMap<usize, &ir::ValueSet>,
                       ir_desc: &ir::IrDesc) -> Option<Rule> {
        let mut conditions = Vec::new();
        for condition in &self.conditions {
            match condition.instantiate(inputs, input_mapping, ir_desc) {
                Condition::Bool(true) => (),
                Condition::Bool(false) => return None,
                c => conditions.push(c),
            }
        }
        let alternatives = self.alternatives.instantiate(input_mapping, ir_desc);
        let set_constraints = self.set_constraints.clone();
        Some(Rule { conditions, alternatives, set_constraints })
    }

    /// Normalizes the `Rule`.
    pub fn normalize(&mut self, inputs: &[ir::ChoiceInstance], ir_desc: &ir::IrDesc) {
        for cond in &mut self.conditions {
            cond.normalize(inputs, ir_desc);
        }
        self.conditions.sort();
    }
}

impl Adaptable for Rule {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        Rule {
            conditions: self.conditions.iter().map(|c| c.adapt(adaptator)).collect(),
            set_constraints: self.set_constraints.adapt(adaptator),
            alternatives: self.alternatives.adapt(adaptator),
        }
    }
}

/// A list of constraints on the set each variable belongs to. It must be built using
/// `SetConstraints::new` so the constraints are in the right order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SetConstraints {
    constraints: Vec<(Variable, ir::Set)>,
}

impl SetConstraints {
    /// Create a new list of set constraints. The constraints are put in a normalized and
    /// legal order.
    pub fn new(mut constraints: Vec<(Variable, ir::Set)>) -> Self {
        constraints.sort_by_key(|&(v, ref s)| (s.def().def_order(), v));
        SetConstraints { constraints }
    }

    /// Returns the constraints in a legal order.
    pub fn constraints(&self) -> &[(Variable, ir::Set)] { &self.constraints }

    /// Indicates if the set of constraints is empty.
    pub fn is_empty(&self) -> bool { self.constraints.is_empty() }

    /// Returns the set the given variable is constrained to, if any.
    pub fn find_set(&self, var: Variable) -> Option<&ir::Set> {
        self.constraints.iter().find(|x| x.0 == var).map(|x| &x.1)
    }
}

impl Adaptable for SetConstraints {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        let constraints = self.constraints.iter().map(|&(v, ref s)| {
            (adaptator.variable(v), s.adapt(adaptator))
        }).collect();
        SetConstraints::new(constraints)
    }
}

impl IntoIterator for SetConstraints {
    type Item = (Variable, ir::Set);
    type IntoIter = std::vec::IntoIter<(Variable, ir::Set)>;

    fn into_iter(self) -> Self::IntoIter { self.constraints.into_iter() }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Variable { Arg(usize), Forall(usize) }

impl Adaptable for Variable {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        adaptator.variable(*self)
    }
}

impl std::fmt::Debug for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Variable::Arg(id) => write!(f, "$arg_{}", id),
            Variable::Forall(id) => write!(f, "$var_{}", id),
        }
    }
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Variable::Arg(id) => write!(f, "arg_{}", id),
            Variable::Forall(id) => write!(f, "var_{}", id),
        }
    }
}

/// An choice instantiated with the given variables.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ChoiceInstance {
    pub choice: RcStr,
    pub vars: Vec<Variable>,
}

impl ChoiceInstance {
    /// Normalizes the `ChoiceInstance` and indicates if the corresponding input should be
    /// inversed.
    pub fn normalize(&mut self, ir_desc: &ir::IrDesc) -> bool {
        match *ir_desc.get_choice(&self.choice).arguments() {
            ir::ChoiceArguments::Plain {..} => false,
            ir::ChoiceArguments::Symmetric { inverse, .. } => {
                assert_eq!(self.vars.len(), 2);
                if self.vars[0] > self.vars[1] {
                    self.vars.swap(0, 1);
                    inverse
                } else {
                    false
                }
            },
        }
    }
}

impl Adaptable for ChoiceInstance {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        ChoiceInstance {
            choice: self.choice.clone(),
            vars: self.vars.iter().map(|v| adaptator.variable(*v)).collect(),
        }
    }
}

/// A piece of rust code.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Code {
    pub code: RcStr,
    pub vars: Vec<(Variable, RcStr)>,
}

impl Code {
    /// Normalizes the `Code.
    pub fn normalize(&mut self) {
        let code = &mut self.code;
        self.vars = self.vars.iter().map(|&(var, ref old_name)| {
            let old_pattern = "$".to_string() + old_name;
            let new_name = RcStr::new(var.to_string());
            let new_pattern = "$".to_string() + &new_name;
            *code = RcStr::new(code.replace(&old_pattern, &new_pattern));
            (var, new_name)
        }).sorted_by(|x, y| std::cmp::Ord::cmp(&x.0, &y.0));
        self.vars.dedup();
    }
}

impl Adaptable for Code {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        let vars = self.vars.iter()
            .map(|&(var, ref name)| (adaptator.variable(var), name.clone()))
            .collect();
        Code { code: self.code.clone(), vars }
    }
}

/// A condition producing a boolean.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Condition {
    /// Triggers if the given piece of code returns `true`, given the mapping of variables.
    Code { code: Code, negate: bool },
    /// Triggers if the choice can only take the given values.
    Enum { input: usize, values: BTreeSet<RcStr>, negate: bool, inverse: bool },
    /// Always or never triggers.
    Bool(bool),
    /// Triggers if the one inputs respects the comparison operator with some rust code.
    CmpCode { lhs: usize, rhs: Code, op: CmpOp },
    /// Triggers if the two inputs respects the comparison operator.
    CmpInput { lhs: usize, rhs: usize, op: CmpOp, inverse: bool },
}

impl Condition {
    /// Negates the condition.
    pub fn negate(&mut self) {
        match *self {
            Condition::Code { ref mut negate, .. } |
            Condition::Bool(ref mut negate) |
            Condition::Enum { ref mut negate, .. } => *negate = !*negate,
            Condition::CmpCode { ref mut op,  .. } |
            Condition::CmpInput { ref mut op, .. } => op.negate(),
        }
    }

    /// Returns allowed alternatives for the given input. Returns None if the condition
    /// is not on the given input.
    pub fn alternatives_of(&self, input_id: usize, t: &ir::ValueType,
                           ir_desc: &ir::IrDesc) -> Option<ValueSet> {
        match *self {
            Condition::Enum { input, ref values, negate, inverse } if input_id == input => {
                let enum_ = ir_desc.get_enum(unwrap!(t.as_enum()));
                Some(normalized_enum_set(values, negate, inverse, enum_))
            },
            Condition::CmpCode { lhs, op, ref rhs } if lhs == input_id => {
                let cmp_code = std::iter::once((op, rhs.clone())).collect();
                let cmp_inputs = BTreeSet::default();
                let universe = t.clone().full_type();
                Some(ValueSet::Integer { is_full: false, cmp_inputs, cmp_code, universe })
            },
            Condition::CmpCode { .. } => None,
            Condition::CmpInput { lhs, rhs, op, inverse } => {
                if input_id == lhs && input_id == rhs {
                    let is_eq = op.allows_eq();
                    Some(ValueSet::from_properties(t, is_eq, inverse, ir_desc))
                } else if input_id == lhs {
                    Some(ValueSet::from_input(t, rhs, op, inverse))
                } else if input_id == rhs {
                    Some(ValueSet::from_input(t, lhs, op.inverse(), inverse))
                } else {
                    None
                }
            },
            _ => None,
        }
    }

    /// Instantiate the condition in the given context.
    pub fn instantiate(&self, inputs: &[ChoiceInstance],
                       input_mapping: &HashMap<usize, &ir::ValueSet>,
                       ir_desc: &ir::IrDesc) -> Condition {
        self.evaluate(inputs, input_mapping, ir_desc).as_bool()
            .map(|b| Condition::Bool(b)).unwrap_or(self.clone())
    }

    /// Evaluates the condition. Requires the mapping to be instantiated.
    pub fn evaluate(&self, inputs: &[ChoiceInstance],
                    input_mapping: &HashMap<usize, &ir::ValueSet>,
                    ir_desc: &ir::IrDesc) -> Trivalent {
        match *self {
            Condition::Bool(true) => Trivalent::True,
            Condition::Bool(false) => Trivalent::False,
            Condition::Code { .. } |
            Condition::CmpCode { .. } => Trivalent::Maybe,
            Condition::Enum { input, ref values, negate, inverse } => {
                let choice = ir_desc.get_choice(&inputs[input].choice);
                let enum_ = ir_desc.get_enum(choice.choice_def().as_enum().unwrap());
                let values = normalized_enum_set(values, negate, inverse, enum_);
                let input_val = input_mapping.get(&input);
                input_val.map(|i| i.is(&values)).unwrap_or(Trivalent::Maybe)
            },
            Condition::CmpInput { lhs, rhs, op, inverse } => {
                let lhs = input_mapping.get(&lhs);
                if let (Some(&lhs), Some(&rhs)) = (lhs, input_mapping.get(&rhs)) {
                    let mut lhs = lhs.clone();
                    if inverse { lhs.inverse(ir_desc); }
                    op.evaluate(&lhs, rhs)
                } else {
                    Trivalent::Maybe
                }
            },
        }
    }

    /// Returns the `StaticCond` representing the condition, if it exists. Indicates if
    /// the condition is negated.
    #[cfg(test)]
    pub fn as_static_cond(&self) -> Option<(ir::test::StaticCond, bool)> {
        match *self {
            Condition::Enum { .. } |
            Condition::Bool(_) |
            Condition::CmpCode { .. } |
            Condition::CmpInput { .. } => None,
            Condition::Code { ref code, negate } =>
                Some((ir::test::StaticCond::Code(code.clone()), negate)),
        }
    }

    /// Normalizes the condition to make it easier to apply equality on it.
    pub fn normalize(&mut self, inputs: &[ChoiceInstance], ir_desc: &ir::IrDesc) {
        match *self {
            Condition::Bool(..) => (),
            Condition::CmpInput { ref mut lhs, ref mut rhs, ref mut op, .. } => {
                if lhs > rhs {
                    std::mem::swap(lhs, rhs);
                    *op = op.inverse();
                }
            },
            Condition::CmpCode { ref mut rhs, .. } => rhs.normalize(),
            Condition::Code { ref mut code, .. } => code.normalize(),
            Condition::Enum { input, ref mut values, ref mut negate, ref mut inverse} => {
                let choice = ir_desc.get_choice(&inputs[input].choice);
                let enum_ = ir_desc.get_enum(choice.choice_def().as_enum().unwrap());
                *values = normalize_values(&*values, *negate, *inverse, enum_)
                    .into_iter().cloned().collect();
                *negate = false;
                *inverse = false;
            },
        }
    }
}

impl Adaptable for Condition {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        match *self {
            Condition::Bool(..) => self.clone(),
            Condition::Enum { input, ref values, negate, inverse } => {
                let (new_input, inversed) = adaptator.input(input);
                Condition::Enum {
                    input: new_input,
                    values: values.clone(),
                    negate: negate,
                    inverse: inverse ^ inversed,
                }
            },
            Condition::Code { ref code, negate } =>
                Condition::Code { code: code.adapt(adaptator), negate },
            Condition::CmpCode { lhs, ref rhs, op } => {
                let (new_lhs, lhs_inversed) = adaptator.input(lhs);
                assert!(!lhs_inversed);
                Condition::CmpCode { lhs: new_lhs, rhs: rhs.adapt(adaptator), op: op }
            },
            Condition::CmpInput { lhs, rhs, op, inverse } => {
                let (new_lhs, lhs_inversed) = adaptator.input(lhs);
                let (new_rhs, rhs_inversed) = adaptator.input(rhs);
                Condition::CmpInput {
                    lhs: new_lhs,
                    rhs: new_rhs,
                    op: op,
                    inverse: inverse ^ lhs_inversed ^ rhs_inversed,
                }
            },
        }
    }
}

/// A compariason operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub enum CmpOp { Lt, Gt, Leq, Geq, Eq, Neq }

impl CmpOp {
    /// Negates the operator.
    pub fn negate(&mut self) {
        *self = match *self {
            CmpOp::Lt => CmpOp::Geq,
            CmpOp::Gt => CmpOp::Leq,
            CmpOp::Leq => CmpOp::Gt,
            CmpOp::Geq => CmpOp::Lt,
            CmpOp::Eq => CmpOp::Neq,
            CmpOp::Neq => CmpOp::Eq,
        }
    }

    /// Returns the equivalent operator for when the operator are inversed.
    pub fn inverse(&self) -> Self {
        match *self {
            CmpOp::Lt => CmpOp::Gt,
            CmpOp::Gt => CmpOp::Lt,
            CmpOp::Leq => CmpOp::Geq,
            CmpOp::Geq => CmpOp::Leq,
            CmpOp::Eq => CmpOp::Eq,
            CmpOp::Neq => CmpOp::Neq,
        }
    }

    /// Indicates if the operator returns true when both operands are equals.
    pub fn allows_eq(&self) -> bool {
        match *self {
            CmpOp::Lt | CmpOp::Gt | CmpOp::Neq => false,
            CmpOp::Leq | CmpOp::Geq | CmpOp::Eq => true,
        }
    }

    /// Evaluates the operator on the given `ValueSet`s.
    pub fn evaluate(&self, lhs: &ValueSet, rhs: &ValueSet) -> Trivalent {
        match *self {
            CmpOp::Lt | CmpOp::Gt | CmpOp::Leq | CmpOp::Geq => Trivalent::Maybe,
            CmpOp::Eq if lhs.is_constrained().is_true() => lhs.is(rhs) & rhs.is(lhs),
            CmpOp::Eq => lhs.is(rhs) & rhs.is(lhs) & Trivalent::Maybe,
            CmpOp::Neq => !lhs.is(rhs) | !rhs.is(lhs),
        }
    }
}

/// Normalizes a list of values.
fn normalize_values<'a, IT>(values: IT, negate: bool, inverse: bool,
                            choice: &'a ir::Enum) -> HashSet<&'a RcStr>
    where IT: IntoIterator<Item=&'a RcStr>
{
    let inverser = |x| if inverse { choice.inverse(x) } else { x };
    let mut values: HashSet<_> = values.into_iter().map(inverser).collect();
    if negate {
        values = choice.values().keys().filter(|&x| !values.contains(x)).collect();
    }
    values
}


/// Creates a `ValueSet` from the list of enum values.
pub fn normalized_enum_set<'a, IT>(values: IT, negate: bool, inverse: bool,
                               choice: &'a ir::Enum) -> ValueSet
    where IT: IntoIterator<Item=&'a RcStr>
{
    let values = normalize_values(values, negate, inverse, choice)
        .into_iter().map(|x| x.clone()).collect();
    ValueSet::enum_values(choice.name().clone(), values)
}

/// Represents a set of values a choice can take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueSet {
    // TODO(cc_perf): detect when an input and its negation are included.
    Enum {
        /// The enum type.
        enum_name: RcStr,
        /// A fixed set of values. Does not contains aliases.
        values: BTreeSet<RcStr>,
        /// A set of inputs whose values can be take by the choice. Two flags indicate if the
        /// value should be negated or inversed.
        inputs: BTreeSet<(usize, bool, bool)>,
    },
    Integer {
        is_full: bool,
        cmp_inputs: BTreeSet<(CmpOp, usize)>,
        cmp_code: BTreeSet<(CmpOp, Code)>,
        universe: ir::ValueType,
    },
}

impl ValueSet {
    /// Creates an enmpty `ValueSet` of the given type.
    pub fn empty(t: &ir::ValueType) -> Self {
        match t {
            ir::ValueType::Enum(name) => {
                ValueSet::Enum {
                    enum_name: name.clone(),
                    values: BTreeSet::default(),
                    inputs: BTreeSet::default()
                }
            },
            t @ ir::ValueType::Range |
            t @ ir::ValueType::HalfRange |
            t @ ir::ValueType::NumericSet(..) => {
                ValueSet::Integer {
                    is_full: false,
                    cmp_inputs: BTreeSet::default(),
                    cmp_code: BTreeSet::default(),
                    universe: t.clone().full_type(),
                }
            },
        }

    }

    /// Computes a `ValueSet` from the properties it must respect.
    pub fn from_properties(t: &ir::ValueType, is_eq: bool, is_inv: bool,
                           ir_desc: &ir::IrDesc) -> Self {
        match t {
            ir::ValueType::Enum(name) => {
                let enum_ = ir_desc.get_enum(name);
                if is_inv {
                    let values = enum_.values().keys()
                        .filter(|&v| enum_.inverse(v) == v);
                    normalized_enum_set(values, !is_eq, false, enum_)
                } else {
                    normalized_enum_set(vec![], is_eq, false, enum_)
                }
            },
            t @ ir::ValueType::Range |
            t @ ir::ValueType::HalfRange |
            t @ ir::ValueType::NumericSet(..) => {
                assert!(!is_inv);
                if is_eq {
                    ValueSet::Integer {
                        is_full: true,
                        cmp_inputs: BTreeSet::default(),
                        cmp_code: BTreeSet::default(),
                        universe: t.clone().full_type(),
                    }
                } else { ValueSet::empty(t) }
            },
        }
    }

    /// Creates a `ValueSet` from a normalized set of values.
    pub fn enum_values(enum_: RcStr, values: BTreeSet<RcStr>) -> Self {
        ValueSet::Enum { enum_name: enum_, values: values, inputs: Default::default() }
    }

    /// Creates a `ValueSet` from the given input.
    pub fn from_input(t: &ir::ValueType, input: usize, op: CmpOp, inverse: bool) -> Self {
        match t {
            ir::ValueType::Enum(enum_name) => ValueSet::Enum {
                enum_name: enum_name.clone(),
                values: Default::default(),
                inputs: std::iter::once((input, op == CmpOp::Neq, inverse)).collect(),
            },
            t @ ir::ValueType::Range | t @ ir::ValueType::NumericSet(..) => {
                assert!(!inverse);
                ValueSet::Integer {
                    is_full: false,
                    cmp_inputs: std::iter::once((op, input)).collect(),
                    cmp_code: BTreeSet::default(),
                    universe: t.clone().full_type(),
                }
            },
            ir::ValueType::HalfRange => panic!("Cannot compare HalfRanges to inputs"),
        }
    }

    /// Indicates if the set of values is empty.
    pub fn is_empty(&self) -> bool {
        match *self {
            ValueSet::Enum { ref values, ref inputs, .. } =>
                values.is_empty() && inputs.is_empty(),
            ValueSet::Integer { is_full, ref cmp_inputs, ref cmp_code, universe: _ } =>
                !is_full && cmp_inputs.is_empty() && cmp_code.is_empty()
        }
    }

    /// Indicates if the set contains all the values. This functions is pessimistic: the
    /// set may contain all the values and the function still return false.
    pub fn is_full(&self, ir_desc: &ir::IrDesc) -> bool {
        match *self {
            ValueSet::Enum { ref values, ref enum_name, .. } =>
                ir_desc.get_enum(enum_name).values().len() == values.len(),
            ValueSet::Integer { is_full, .. } => is_full,
        }
    }

    /// Indicates if the set contains a single value.
    pub fn is_constrained(&self) -> Trivalent {
        match *self {
            ValueSet::Enum { ref values, ref inputs, .. } if inputs.is_empty() => {
                if values.len() == 1 { Trivalent::True } else { Trivalent::False }
            },
            ValueSet::Enum { ref values, ref inputs, .. } if values.is_empty() => {
                if inputs.len() == 1 { Trivalent::True } else { Trivalent::False }
            },
            ValueSet::Enum { .. } | ValueSet::Integer { .. } => Trivalent::Maybe,
        }
    }

    /// Extends the `ValueSet` with the values of anther set.
    pub fn extend(&mut self, other: ValueSet) {
        match *self {
            ValueSet::Enum { ref mut values, ref mut inputs, .. } => match other {
                ValueSet::Enum { values: other_values, inputs: other_inputs, .. } => {
                    values.extend(other_values);
                    inputs.extend(other_inputs);
                },
                ValueSet::Integer { .. } => panic!(),
            },
            ValueSet::Integer {
                ref mut is_full,
                ref mut cmp_inputs,
                ref mut cmp_code,
                universe: _,
            } => {
                if let ValueSet::Integer {
                    is_full: other_full,
                    cmp_inputs: other_inputs,
                    cmp_code: other_code,
                    universe: _,
                } = other {
                    if *is_full || other_full { *is_full = true } else {
                        cmp_inputs.extend(other_inputs);
                        cmp_code.extend(other_code);
                    }
                } else { panic!() }
            }
        }
    }

    /// Intersects the `ValueSet` with the given values. Indicates if the intersection was
    /// successful or if the sets should be kept separate.
    pub fn intersect(&mut self, other: ValueSet) -> bool {
        match (self, other) {
            (&mut ValueSet::Enum { values: ref mut vals0, inputs: ref ins0, .. },
             ValueSet::Enum { values: ref mut vals1, inputs: ref ins1, .. }) => {
                if ins0.is_empty() && ins1.is_empty() {
                    *vals0 = vals0.iter().cloned().filter(|v| vals1.contains(v)).collect();
                    true
                } else { false }
            },
            (s @ &mut ValueSet::Integer { is_full: true, .. }, other) => {
                *s = other;
                true
            },
            (_, ValueSet::Integer { is_full: true, .. }) => true,
            (&mut ValueSet::Integer { ref mut cmp_inputs, ref mut cmp_code, .. },
             ValueSet::Integer { cmp_inputs: ref mut ins1, cmp_code: ref mut code1, .. }) => {
                 if ins1.is_subset(cmp_inputs) && code1.is_subset(cmp_code) {
                     std::mem::swap(cmp_inputs, ins1);
                     std::mem::swap(cmp_code, code1);
                     true
                 } else {
                     cmp_inputs.is_subset(ins1) && cmp_code.is_subset(code1)
                 }
             },
             _ => panic!()
        }
    }

    /// Instantiates the `ValueSet` for a given input assignment.
    pub fn instantiate(&self, input_mapping: &HashMap<usize, &ir::ValueSet>,
                       ir_desc: &ir::IrDesc) -> Self {
        match *self {
            ValueSet::Enum { ref enum_name, ref values, ref inputs } => {
                let mut new_values = values.clone();
                let enum_ = ir_desc.get_enum(enum_name);
                let mut new_inputs = BTreeSet::default();
                for &(input, negate, inverse) in inputs {
                    if let Some(mapping) = input_mapping.get(&input).cloned() {
                        let mut mapping = mapping.clone();
                        if inverse { mapping.inverse(ir_desc); }
                        match mapping {
                            ValueSet::Enum { ref values, ref inputs, .. } => {
                                if negate {
                                    let values = enum_.values().keys()
                                        .filter(|&v| !values.contains(v)).cloned();
                                    new_values.extend(values);
                                } else {
                                    new_values.extend(values.into_iter().cloned());
                                }
                                for &(new_input, neg2, inv) in inputs {
                                    new_inputs.insert((new_input, negate ^ neg2, inv));
                                }
                            },
                            _ => panic!(),
                        }
                    } else {
                        new_inputs.insert((input, negate, inverse));
                    }
                }
                ValueSet::Enum {
                    enum_name: enum_name.clone(),
                    values: new_values,
                    inputs: new_inputs,
                }
            },
            ValueSet::Integer { .. } => self.clone(), // Cannot be instantiated
        }
    }

    /// Inverse the `ValueSet`. The choice must be antisymmetric.
    pub fn inverse(&mut self, ir_desc: &ir::IrDesc) {
        match *self {
            ValueSet::Enum { ref enum_name, ref mut values, ref mut inputs } => {
                let enum_ = ir_desc.get_enum(enum_name);
                *values = values.iter().map(|v| enum_.inverse(&v).clone()).collect();
                *inputs = inputs.iter().map(|&(id, neg, inverse)| (id, neg, !inverse))
                    .collect();
            },
            ValueSet::Integer { .. } => panic!(),
        }
    }

    /// Indicates if the `ValueSet` will be contained into anoter after instantiation.
    /// Requires both `self` and `other` to be instantiated.
    pub fn is(&self, other: &ValueSet) -> Trivalent {
        match (self, other) {
            (&ValueSet::Enum { ref values, ref inputs, .. },
             &ValueSet::Enum { values: ref values2, inputs: ref inputs2, .. }) => {
                assert!(inputs.is_empty() && inputs2.is_empty());
                if values.is_subset(values2) {
                    Trivalent::True
                } else if values.is_disjoint(values2) {
                    Trivalent::False
                } else {
                    Trivalent::Maybe
                }
            },
            _ => panic!()
        }
    }

    /// Returns the type of the values.
    pub fn t(&self) -> ir::ValueType {
        match *self {
            ValueSet::Enum { ref enum_name, .. } => ir::ValueType::Enum(enum_name.clone()),
            ValueSet::Integer { ref universe, ..} => universe.clone(),
        }
    }
}

impl Adaptable for ValueSet {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        match *self {
            ValueSet::Enum { ref inputs, ref enum_name, ref values } => {
                let values = values.clone();
                let inputs = inputs.iter().cloned().map(|(input, negate, inverse)| {
                    let (new_input, inversed) = adaptator.input(input);
                    (new_input, negate, inverse ^ inversed)
                }).collect();
                ValueSet::Enum { inputs, enum_name: enum_name.clone(), values }
            },
            ValueSet::Integer { ref cmp_inputs, ref cmp_code, is_full, ref universe } => {
                let cmp_inputs = cmp_inputs.iter().cloned().map(|(op, input)| {
                    let (new_input, inversed) = adaptator.input(input);
                    assert!(!inversed);
                    (op, new_input)
                }).collect();
                let cmp_code = cmp_code.iter().map(|(op, code)| {
                    (*op, code.adapt(adaptator))
                }).collect();
                ValueSet::Integer {
                    is_full, cmp_inputs, cmp_code,
                    universe: universe.adapt(adaptator),
                }
            },
        }
    }
}


#[cfg(test)]
pub mod test {
    use super::*;

    /// Creates a set from of values.
    pub fn mk_enum_values_set(enum_: &str, values: &[&str]) -> ir::ValueSet {
        let values = values.iter().map(|&x| x.into()).collect();
        ir::ValueSet::enum_values(enum_.into(), values)
    }

    /// Enusure `ValueSet::is` is working.
    #[test]
    fn value_set_is() {
        let set = mk_enum_values_set("enum", &["A", "B"]);
        let disjoint = mk_enum_values_set("enum", &["C"]);
        let subset = mk_enum_values_set("enum", &["A"]);
        let intersect = mk_enum_values_set("enum", &["B", "C"]);
        assert_eq!(set.is(&disjoint), Trivalent::False);
        assert_eq!(set.is(&subset), Trivalent::Maybe);
        assert_eq!(subset.is(&set), Trivalent::True);
        assert_eq!(set.is(&intersect), Trivalent::Maybe);
    }

    /// Test `ValueSet::instantiate".
    #[test]
    fn value_set_instantiate() {
        // Declare the Enum
        let mut ir_desc = ir::IrDesc::default();
        let enum_name: RcStr = "Foo".into();
        let a: RcStr = "A".into();
        let b: RcStr = "B".into();
        let c: RcStr = "C".into();
        let mapping = vec![(a.clone(), b.clone())];
        let mut enum_ = ir::Enum::new(enum_name.clone(), None, Some(mapping));
        enum_.add_value(a.clone(), None);
        enum_.add_value(b.clone(), None);
        enum_.add_value(c.clone(), None);
        ir_desc.add_enum(enum_);
        let t = ir::ValueType::Enum(enum_name.clone());
        // Declare reference sets.
        let a_set = ValueSet::enum_values(enum_name.clone(),
            std::iter::once(a.clone()).collect());
        let b_set = ValueSet::enum_values(enum_name.clone(),
            std::iter::once(b.clone()).collect());
        let c_set = ValueSet::enum_values(enum_name.clone(),
            std::iter::once(c.clone()).collect());
        let ac_set = ValueSet::enum_values(enum_name.clone(),
            vec![a.clone(), c.clone()].into_iter().collect());
        let bc_set = ValueSet::enum_values(enum_name.clone(),
            vec![b.clone(), c.clone()].into_iter().collect());
        let a_mapping = std::iter::once((0, &a_set)).collect();
        let c_mapping = std::iter::once((0, &c_set)).collect();
        // Eq, not inversed.
        let set = ValueSet::from_input(&t, 0, CmpOp::Eq, false);
        assert_eq!(set.instantiate(&a_mapping, &ir_desc), a_set);
        // Eq, 'a' inversed.
        let set = ValueSet::from_input(&t, 0, CmpOp::Eq, true);
        assert_eq!(set.instantiate(&a_mapping, &ir_desc), b_set);
        // Eq, 'c' inversed.
        let set = ValueSet::from_input(&t, 0, CmpOp::Eq, true);
        assert_eq!(set.instantiate(&c_mapping, &ir_desc), c_set);
        // NEq, not inversed.
        let set = ValueSet::from_input(&t, 0, CmpOp::Neq, false);
        assert_eq!(set.instantiate(&a_mapping, &ir_desc), bc_set);
        // NEq, inversed.
        let set = ValueSet::from_input(&t, 0, CmpOp::Neq, true);
        assert_eq!(set.instantiate(&a_mapping, &ir_desc), ac_set);
    }
}
