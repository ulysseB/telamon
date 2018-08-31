//! Filter generation.
use ir::{self, Adaptable};
use utils::*;

/// Merge flat filters that can be merged in the given list.
pub fn merge(mut filters: Vec<FlatFilter>, ir_desc: &ir::IrDesc) -> Vec<FlatFilter> {
    // Filters can only be merged into filters with more inputs. Thus we only try to
    // merge them with filter with more inputs.
    filters.sort_by_key(|x| x.inputs.len());
    let mut merged_filters: Vec<FlatFilter> = Vec::new();
    for filter in filters.into_iter().rev() {
        let mut merged = false;
        for other_filter in &mut merged_filters {
            if other_filter.try_merge(&filter, ir_desc) {
                merged = true;
            }
        }
        if !merged {
            merged_filters.push(filter);
        }
    }
    merged_filters
}

/// A filter with only negative rules.
#[derive(Debug, Clone)]
pub struct FlatFilter {
    vars: Vec<ir::Set>,
    inputs: Vec<ir::ChoiceInstance>,
    rules: Vec<ir::Rule>,
    set_constraints: ir::SetConstraints,
}

impl FlatFilter {
    /// Builds a `FlatFilter`. The inputs are normalized.
    pub fn new(
        vars: Vec<ir::Set>,
        mut inputs: Vec<ir::ChoiceInstance>,
        rules: Vec<ir::Rule>,
        set_constraints: ir::SetConstraints,
        ir_desc: &ir::IrDesc,
    ) -> Self {
        let mut adaptator = ir::Adaptator::default();
        for (pos, input) in inputs.iter_mut().enumerate() {
            if input.normalize(ir_desc) {
                adaptator.set_inversed(pos);
            }
        }
        let rules = rules.iter().map(|r| r.adapt(&adaptator)).collect();
        FlatFilter {
            vars,
            inputs,
            rules,
            set_constraints,
        }
    }

    /// Returns the composants of the 'FlatFilter'
    pub fn deconstruct(
        self,
    ) -> (
        Vec<ir::Set>,
        Vec<ir::ChoiceInstance>,
        Vec<ir::Rule>,
        ir::SetConstraints,
    ) {
        (self.vars, self.inputs, self.rules, self.set_constraints)
    }

    /// Try to merge another filter into `Self`.
    fn try_merge(&mut self, other: &FlatFilter, ir_desc: &ir::IrDesc) -> bool {
        // FIXME: do not set the merged flag if the set of foralls is bigger
        if self.inputs.is_empty() != other.inputs.is_empty() {
            return false;
        }
        if self.set_constraints != other.set_constraints {
            return false;
        }
        let mut merged = false;
        let input_map: HashMap<_, _> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(x, y)| (y, x))
            .collect();
        let var_maps = PartialPermutations::new(0..self.vars.len(), other.vars.len());
        // Try every mapping of variable and merge each time it is possible
        'var_maps: for var_map in var_maps {
            let mut adaptator = ir::Adaptator::default();
            // Ensure variables are compatible.
            for (old_id, &new_id) in var_map.iter().enumerate() {
                if other.vars[old_id] != self.vars[new_id] {
                    continue 'var_maps;
                }
                let old_var = ir::Variable::Forall(old_id);
                adaptator.set_variable(old_var, ir::Variable::Forall(new_id));
            }
            // Find the input mapping.
            for (old_id, input) in other.inputs.iter().enumerate() {
                let mut new_input = input.adapt(&adaptator);
                if new_input.normalize(ir_desc) {
                    adaptator.set_inversed(old_id);
                }
                if let Some(&new_id) = input_map.get(&new_input) {
                    adaptator.set_input(old_id, new_id);
                } else {
                    continue 'var_maps;
                }
            }
            // Adapt and merge the rules.
            self.rules
                .extend(other.rules.iter().map(|r| r.adapt(&adaptator)));
            // If the number of variable is different, don't set the merged flag: otherwise, the
            // filter might not be run if their is not enought to objects to run the filter with
            // more variables.
            if self.vars.len() == other.vars.len() {
                // Early exit for static filters.
                if self.inputs.is_empty() {
                    return true;
                }
                merged = true;
            }
        }
        merged
    }

    /// Inverse a filter on a symmetric or anti-symmetric relation.
    pub fn inverse(&self, antisymmetric: bool, ir_desc: &ir::IrDesc) -> Self {
        let mut adaptator = ir::Adaptator::default();
        adaptator.set_variable(ir::Variable::Arg(0), ir::Variable::Arg(1));
        adaptator.set_variable(ir::Variable::Arg(1), ir::Variable::Arg(0));
        let rules = self
            .rules
            .iter()
            .map(|old_rule| {
                let mut rule = old_rule.adapt(&adaptator);
                if antisymmetric {
                    rule.alternatives.inverse(ir_desc);
                }
                rule
            })
            .collect();
        let inputs = self
            .inputs
            .iter()
            .map(|input| input.adapt(&adaptator))
            .collect();
        let constraints = self.set_constraints.adapt(&adaptator);
        FlatFilter::new(self.vars.clone(), inputs, rules, constraints, ir_desc)
    }
}
