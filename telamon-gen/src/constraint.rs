//! Constraint representation and manipulation.
use crate::flat_filter::FlatFilter;
use crate::ir::SetRef;
use crate::ir::{self, Adaptable};
use std;
use utils::*;

/// A constraints on IR choices.
#[derive(Debug)]
pub struct Constraint {
    pub vars: Vec<ir::Set>,
    pub inputs: Vec<ir::ChoiceInstance>,
    pub conditions: Vec<ir::Condition>,
    /// Indicates if the constraint should restricct fragile values.
    pub restrict_fragile: bool,
}

impl Constraint {
    /// Remove duplicates among inputs.
    pub fn dedup_inputs(&mut self, ir_desc: &ir::IrDesc) {
        let old_inputs = std::mem::replace(&mut self.inputs, Vec::new());
        let (new_inputs, adaptator) = dedup_inputs(old_inputs, ir_desc);
        self.inputs = new_inputs;
        self.conditions = self
            .conditions
            .iter()
            .map(|x| x.adapt(&adaptator))
            .collect();
    }

    /// Generate filters to enforce the constraint.
    pub fn gen_filters(&self, ir_desc: &ir::IrDesc) -> Vec<(RcStr, FlatFilter)> {
        let mut filters = vec![];
        for (input_id, input) in self.inputs.iter().enumerate() {
            let filter = if let Some(filter) = gen_flat_filter(self, input_id, ir_desc) {
                filter
            } else {
                continue;
            };
            let choice = ir_desc.get_choice(&input.choice);
            match *choice.arguments() {
                ir::ChoiceArguments::Plain { .. } => (),
                ir::ChoiceArguments::Symmetric { inverse, .. } => {
                    let filter = filter.inverse(inverse, ir_desc);
                    filters.push((input.choice.clone(), filter));
                }
            }
            filters.push((input.choice.clone(), filter));
        }
        filters
    }
}

/// Normalizes a list of inputs.
pub fn dedup_inputs(
    mut inputs: Vec<ir::ChoiceInstance>,
    ir_desc: &ir::IrDesc,
) -> (Vec<ir::ChoiceInstance>, ir::Adaptator) {
    let mut adaptator = ir::Adaptator::default();
    // Normalize inputs.
    for (pos, input) in inputs.iter_mut().enumerate() {
        if input.normalize(ir_desc) {
            adaptator.set_inversed(pos);
        }
    }
    let mut new_input_defs;
    // Assign new positions.
    {
        let mut input_map = HashMap::default();
        for (old_pos, input) in inputs.iter().enumerate() {
            let next_pos = input_map.len();
            let new_pos = *input_map.entry(input).or_insert(next_pos);
            adaptator.set_input(old_pos, new_pos);
        }
        new_input_defs = Vec::with_capacity(input_map.len());
    }
    // Build the new input vector.
    for (old_pos, input) in inputs.into_iter().enumerate() {
        let (new_pos, _) = adaptator.input(old_pos);
        if new_pos == new_input_defs.len() {
            new_input_defs.push(input);
        }
    }
    (new_input_defs, adaptator)
}

/// Generates a filter for the given input from from the given constraint.
fn gen_flat_filter(
    constraint: &Constraint,
    input_id: usize,
    ir_desc: &ir::IrDesc,
) -> Option<FlatFilter> {
    let choice = ir_desc.get_choice(&constraint.inputs[input_id].choice);
    // Extract the conditions on the given input.
    let env = constraint
        .vars
        .iter()
        .enumerate()
        .map(|(i, set)| (ir::Variable::Forall(i), set.clone()))
        .collect::<HashMap<_, _>>();
    let (foralls, all_set_constraints, mut conditions, _) = ir::ChoiceCondition::new(
        ir_desc,
        constraint.inputs.clone(),
        input_id,
        &constraint.conditions,
        env,
    );
    if !constraint.restrict_fragile {
        conditions
            .self_condition
            .extend(choice.fragile_values().clone());
    }
    if conditions.self_condition.is_full(ir_desc) {
        return None;
    }
    for cond in &mut conditions.others_conditions {
        cond.negate();
    }
    // Detect the set constraints that must be applied before fetching the inputs.
    let mut input_set_constraints = Vec::new();
    let mut inner_set_constraints = Vec::new();
    for (var, subset) in all_set_constraints.into_iter().rev() {
        let given_set = match var {
            ir::Variable::Arg(i) => choice.arguments().get(i).1,
            _ => panic!(),
        };
        let mut is_used = false;
        for input in &conditions.inputs {
            // 1. check if it is used as argument to a choice.
            if let Some(pos) = input.vars.iter().position(|&v| v == var) {
                let choice = ir_desc.get_choice(&input.choice);
                if !given_set.is_subset_of_def(&choice.arguments().get(pos).1) {
                    is_used = true;
                    break;
                }
            }
        }
        // 2. Check if it used in another constraint
        is_used |= input_set_constraints
            .iter()
            .map(|x: &(_, _)| &x.1)
            .chain(&foralls)
            .map(|set| set.as_ref())
            .chain(foralls.iter().flat_map(|set| set.reverse_constraint()))
            .filter(|s| s.arg().map(|v| v == var).unwrap_or(false))
            .map(|s| unwrap!(s.def().arg()))
            .any(|s| !given_set.is_subset_of(s));
        if is_used {
            input_set_constraints.push((var, subset));
        } else {
            inner_set_constraints.push((var, subset));
        }
    }
    inner_set_constraints.reverse();
    input_set_constraints.reverse();
    // Build the flat filter from the different parts.
    let set_constraints = ir::SetConstraints::new(inner_set_constraints);
    let rule = ir::Rule {
        conditions: conditions.others_conditions,
        alternatives: conditions.self_condition,
        set_constraints,
    };
    let input_set_constraints = ir::SetConstraints::new(input_set_constraints);
    Some(FlatFilter::new(
        foralls,
        conditions.inputs,
        vec![rule],
        input_set_constraints,
        ir_desc,
    ))
}
