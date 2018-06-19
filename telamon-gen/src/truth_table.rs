//! Truth tables.
use ir;
use itertools::Itertools;
use std;
use std::collections::hash_map;
use utils::*;

/// Lists the rules to apply for each combinaison of input.
#[derive(Debug)]
struct TruthTable {
    values: Vec<(usize, Vec<ir::ValueSet>)>,
    rules: NDArray<Cell>,
}

impl TruthTable {
    /// Creates the truth table from the set of rules.
    fn build(inputs: &[ir::ChoiceInstance], rules: &[ir::Rule],
             ir_desc: &ir::IrDesc) -> Self {
        let values = inputs.iter().enumerate().flat_map(|(id, input)| {
            let choice = ir_desc.get_choice(&input.choice);
            if !choice.fragile_values().is_empty() { return None; }
            match *choice.choice_def() {
                ir::ChoiceDef::Enum(ref name) => {
                    let sets = ir_desc.get_enum(name).values().keys().map(|v| {
                        let values = std::iter::once(v.clone()).collect();
                        ir::ValueSet::enum_values(name.clone(), values)
                    }).collect_vec();
                    Some((id, sets))
                },
                ir::ChoiceDef::Counter { .. } |
                ir::ChoiceDef::Number { .. } => None,
            }
        }).collect_vec();
        let sizes = values.iter().map(|&(_, ref v)| v.len()).collect_vec();
        let data = NDRange::new(&sizes).map(|index| {
            let input_mapping = index.iter().zip_eq(&values)
                .map(|(&idx, &(input, ref values))| (input, &values[idx]))
                .collect();
            let rules = rules.iter()
                .flat_map(|x| x.instantiate(inputs, &input_mapping, ir_desc))
                .collect_vec();
            Cell::build(rules, inputs, ir_desc)
        }).collect_vec();
        TruthTable { values, rules: NDArray::new(sizes, data), }
    }

    /// Returns a view on the entire table.
    fn view(&mut self) -> TableView {
        TableView {
            values: &self.values,
            dim_map: (0..self.values.len()).collect(),
            view: self.rules.view_mut(),
        }
    }
}

/// The condition and set constraints of a rule.
type RuleConds = (Vec<ir::Condition>, ir::SetConstraints);

/// The rules that applies to a particular combination of choices.
#[derive(Debug, PartialEq, Eq)]
struct Cell {
    rules: HashMap<RuleConds, Vec<ir::ValueSet>>,
}

impl Cell {
    /// Creates a cell from the given rules.
    fn build(rules: Vec<ir::Rule>,
             inputs: &[ir::ChoiceInstance],
             ir_desc: &ir::IrDesc) -> Cell {
        let mut rule_map: HashMap<_, Vec<ir::ValueSet>> = HashMap::default();
        for mut rule in rules {
            rule.normalize(inputs, ir_desc);
            match rule_map.entry((rule.conditions, rule.set_constraints)) {
                hash_map::Entry::Occupied(mut entry) => {
                    let mut success = false;
                    for set in entry.get_mut() {
                        if set.intersect(rule.alternatives.clone()) {
                            success = true;
                            break;
                        }
                    }
                    if !success { entry.get_mut().push(rule.alternatives); }
                },
                hash_map::Entry::Vacant(entry) => {
                    entry.insert(vec![rule.alternatives]);
                },
            };
        }
        Cell { rules: rule_map }
    }

    /// Extracts the rules from the cell. Leaves the cell empty.
    fn extract_rules(&mut self) -> Vec<ir::Rule> {
        self.rules.drain().flat_map(|((conds, set_conds), alternatives)| {
            alternatives.into_iter().map(move |alts| {
                ir::Rule {
                    conditions: conds.clone(),
                    alternatives: alts,
                    set_constraints: set_conds.clone()
                }
            })
        }).collect_vec()
    }

    /// Indicates if the cell allows any alternative,
    fn is_empty(&self) -> bool {
        self.rules.get(&(vec![], ir::SetConstraints::default()))
            .map(|sets| sets.iter().any(|set| set.is_empty()))
            .unwrap_or(false)
    }
}

/// A `TruthTable` with some input fixed.
struct TableView<'a> {
    values: &'a Vec<(usize, Vec<ir::ValueSet>)>,
    /// Maps the id of a view dimension to the id of dimension of the original table.
    dim_map: Vec<usize>,
    view: ndarray::ViewMut<'a, Cell>,
}

impl<'a> TableView<'a> {
    /// Returns the number of inputs of the table.
    fn num_inputs(&self) -> usize { self.dim_map.len() }

    /// Returns the input_id associated to a position.
    fn input_from_pos(&self, input: usize) -> usize { self.values[self.dim_map[input]].0 }

    /// Instantiate the truth table for each value of a given dimension.
    fn instantiate(&mut self, pos: usize) -> Vec<(&'a ir::ValueSet, TableView)> {
        let dim_map = &self.dim_map;
        let values = &self.values;
        self.values[self.dim_map[pos]].1.iter().zip_eq(self.view.split(pos))
            .map(|(value, view)| {
                let mut dim_map = dim_map.clone();
                dim_map.remove(pos);
                (value, TableView { values, dim_map, view })
            }).collect()
    }

    /// Indicates if the two view have the same cells.
    fn equal_content(&self, other: &Self) -> bool {
        ::itertools::equal(self.into_iter(), other.into_iter())
    }

    /// Indicates if al the cells are empty.
    fn is_empty(&self) -> bool { self.into_iter().all(|c| c.is_empty()) }
}

impl<'a, 'b> IntoIterator for &'b TableView<'a> where 'a: 'b {
    type Item = &'b Cell;
    type IntoIter = ndarray::ViewMutIter<'b, Cell>;

    fn into_iter(self) -> Self::IntoIter {
        (&self.view).into_iter()
    }
}

impl<'a, 'b> IntoIterator for &'b mut TableView<'a> where 'a: 'b {
    type Item = &'b mut Cell;
    type IntoIter = ndarray::ViewIterMut<'a, 'b, Cell>;

    fn into_iter(self) -> Self::IntoIter {
        (&mut self.view).into_iter()
    }
}

/// Optimize the given rules into a sub_filter.
pub fn opt_rules(inputs: &[ir::ChoiceInstance], rules: Vec<ir::Rule>,
                      ir_desc: &ir::IrDesc) -> ir::SubFilter {
    if rules.len() == 1 {
        ir::SubFilter::Rules(rules)
    } else {
        let mut table = TruthTable::build(inputs, &rules, ir_desc);
        debug!("truth table: {:?}", table);
        truth_table_to_filter(&mut table.view())
    }
}

/// Implements a truth table as a filter.
fn truth_table_to_filter(table: &mut TableView) -> ir::SubFilter {
    if table.num_inputs() == 0 {
        let cell = table.into_iter().next().unwrap();
        let rules = cell.extract_rules();
        return ir::SubFilter::Rules(rules);
    }
    match table_min_split(table).unwrap() {
        TableSplit::Forward { mut sub_view } => truth_table_to_filter(&mut sub_view),
        TableSplit::Switch { input, cases } => {
            let sub_filters = cases.into_iter().map(|(values, mut view)| {
                (values, truth_table_to_filter(&mut view))
            }).collect();
            ir::SubFilter::Switch { switch: input, cases: sub_filters }
        },
    }
}

enum TableSplit<'a> {
    Switch { input: usize, cases: Vec<(ir::ValueSet, TableView<'a>)> },
    Forward { sub_view: TableView<'a> }
}

/// Find the input instantiation with the minimal number of instances.
fn table_min_split<'a, 'b>(table: &'a mut TableView<'b>) -> Option<TableSplit<'a>> {
    let mut min_split: Option<(usize, Vec<_>)> = None;
    // Find the best splitting configuration
    for pos in 0..table.num_inputs() {
        let mut forward = true;
        {
            let mut instances: Vec<(ir::ValueSet, usize, TableView)> = Vec::new();
            let split = table.instantiate(pos);
            // Merge similar branches.
            'instances: for (pos, (value, sub_table)) in split.into_iter().enumerate() {
                if sub_table.is_empty() {
                    forward = false;
                    continue 'instances;
                }
                for &mut (ref mut other_values, _, ref other_table) in &mut instances {
                    if sub_table.equal_content(other_table) {
                        other_values.extend(value.clone());
                        continue 'instances;
                    }
                }
                instances.push((value.clone(), pos, sub_table));
            }
            forward &= instances.len() == 1;
            // Update the best split.
            if min_split.as_ref().map(|x| x.1.len() > instances.len()).unwrap_or(true) {
                let config = instances.into_iter().map(|(values, vpos, _)| (values, vpos));
                min_split = Some((pos, config.collect_vec()));
            }
        }
        // Early exit if the table is not split.
        if forward {
            let sub_view = table.instantiate(pos).pop().unwrap().1;
            return Some(TableSplit::Forward { sub_view });
        }
    }
    // Replicate the best splitting
    min_split.map(move |(pos, config)| {
        let input = table.input_from_pos(pos);
        let mut views = table.instantiate(pos).into_iter().enumerate();
        let cases = config.into_iter().map(|(values, pos)| {
            let view = views.find(|&(other_pos, _)| pos == other_pos).unwrap().1;
            assert!(!view.1.is_empty());
            (values, view.1)
        }).collect();
        TableSplit::Switch { input, cases }
    })
}

#[cfg(test)]
pub mod test {
    use super::*;
    use constraint::Constraint;
    use ir;
    use itertools::Itertools;
    use ir::test::{EvalContext, mk_enum_values_set};

    /// Returns the values allowed by the given `Cell`.
    fn eval_cell(cell: &Cell, context: &ir::test::EvalContext)
            -> ir::ValueSet {
        let enum_name = context.enum_.name().clone();
        let values = context.enum_.values().keys().cloned().collect();
        let mut valid_values = ir::ValueSet::enum_values(enum_name, values);
        for (&(ref conds, ref set_constraints), value_sets) in &cell.rules {
            for set in value_sets.clone() {
                context.eval_rule_aux(conds, set_constraints, set, &mut valid_values);
            }
        }
        valid_values
    }

    /// Returns the valid alternatives according to a given table.
    fn eval_table(table: &TruthTable, context: &ir::test::EvalContext)
            -> ir::ValueSet {
        let valid_indexes = table.values.iter().map(|&(input, ref table_values)| {
            let ctx_values = &context.input_values[input];
            table_values.iter().enumerate()
                .filter(|&(_, value)| ctx_values.is(value).maybe_true())
                .map(|(pos, _)| pos).collect_vec()
        }).collect_vec();
        let num_indexes = valid_indexes.iter().map(|x| x.len()).collect_vec();
        let t = ir::ValueType::Enum(context.enum_.name().clone());
        let mut value_set = ir::ValueSet::empty(&t);
        for indexes in NDRange::new(&num_indexes) {
            let table_index = indexes.iter().zip_eq(&valid_indexes)
                .map(|(&idx, table_indexes)| table_indexes[idx])
                .collect_vec();
            value_set.extend(eval_cell(&table.rules[&table_index[..]], context));
        }
        value_set
    }

    /// Ensures the generation of filters works when no rule is present.
    #[test]
    fn no_rules() {
        let _ = ::env_logger::try_init();
        let mut ir_desc = ir::IrDesc::default();
        ir::test::gen_enum("A", 3, &mut ir_desc);
        ir::test::gen_enum("B", 3, &mut ir_desc);
        ir::test::gen_enum("C", 3, &mut ir_desc);
        let enum_ = ir_desc.get_enum("EnumA");
        let inputs = [mk_input("enum_b"), mk_input("enum_c")];
        test_filter(&inputs, &[], enum_, &ir_desc);
    }

    /// Ensures the generation of static condition works correctly.
    #[test]
    fn no_inputs_filter() {
        let _ = ::env_logger::try_init();
        let mut ir_desc = ir::IrDesc::default();
        ir::test::gen_enum("A", 4, &mut ir_desc);
        let enum_ = ir_desc.get_enum("EnumA");
        let rule0 = mk_rule(vec![], "EnumA", &["A_0", "A_1", "A_2"]);
        let rule1 = mk_rule(vec![mk_code_cond("code_0")], "EnumA", &["A_1", "A_2", "A_3"]);
        let rules = [rule0, rule1];
        test_filter(&[], &rules, enum_, &ir_desc);
    }

    /// Ensures the generation of filters with a single input works correctly.
    #[test]
    fn single_input_filter() {
        let _ = ::env_logger::try_init();
        let mut ir_desc = ir::IrDesc::default();
        ir::test::gen_enum("A", 4, &mut ir_desc);
        ir::test::gen_enum("B", 4, &mut ir_desc);
        let enum_a = ir_desc.get_enum("EnumA");
        let rule0 = mk_rule(vec![], "EnumA", &["A_0", "A_1", "A_2"]);
        let rule1 = mk_rule(vec![mk_enum_cond(0, &["B_0", "B_1"])], "EnumA", &["A_0", "A_1"]);
        let rule2 = mk_rule(vec![mk_enum_cond(0, &["B_0", "B_2"]),
                            mk_code_cond("code_0")], "EnumA", &["A_0", "A_2"]);
        let rule3 = mk_rule(vec![mk_enum_cond(0, &["B_3"])], "EnumA", &[]);
        let rules = [rule0, rule1, rule2, rule3];
        test_filter(&[mk_input("enum_b")], &rules, enum_a, &ir_desc)
    }

    /// Snsures the generation of filters with multiple inputs works correctly.
    #[test]
    fn two_inputs_filter() {
        let _ = ::env_logger::try_init();
        let mut ir_desc = ir::IrDesc::default();
        ir::test::gen_enum("A", 4, &mut ir_desc);
        ir::test::gen_enum("B", 3, &mut ir_desc);
        ir::test::gen_enum("C", 3, &mut ir_desc);
        let enum_a = ir_desc.get_enum("EnumA");
        let cond_b1 = mk_enum_cond(0, &["B_1"]);
        let cond_c01 = mk_enum_cond(1, &["C_0", "C_1"]);
        let cond_b12 = mk_enum_cond(0, &["B_1", "B_2"]);
        let cond_c12 = mk_enum_cond(1, &["C_1", "C_2"]);
        let cond_code0 = mk_code_cond("code_0");
        let rules = [
            mk_rule(vec![mk_enum_cond(0, &["B_0"])], "EnumA", &["A_0", "A_1",]),
            mk_rule(vec![cond_b1, cond_c01], "EnumA", &["A_1", "A_2"]),
            mk_rule(vec![cond_b12, cond_c12, cond_code0], "EnumA", &["A_2", "A_3"]),
        ];
        let inputs = [mk_input("enum_b"), mk_input("enum_c")];
        test_filter(&inputs, &rules, enum_a, &ir_desc)
    }

    fn test_filter(inputs: &[ir::ChoiceInstance], rules: &[ir::Rule],
                   enum_: &ir::Enum, ir_desc: &ir::IrDesc) {
        let static_conds = rules.iter().flat_map(|rule| {
            rule.conditions.iter().flat_map(|x| x.as_static_cond()).map(|x| x.0)
        }).unique().collect_vec();
        // Test the table correctness.
        let mut table = TruthTable::build(inputs, &rules, ir_desc);
        for ctx in EvalContext::iter_contexts(ir_desc, enum_, inputs, &static_conds[..]) {
            let table_res = eval_table(&table, &ctx);
            let rules_res = ctx.eval_rules(rules);
            debug!("Context{}", ctx);
            debug!("table res: {:?}", table_res);
            debug!("rules res: {:?}", rules_res);
            assert_eq!(table_res, rules_res);
        }
        // Test the generated filter correctness.
        let filter = truth_table_to_filter(&mut table.view());
        for ctx in EvalContext::iter_contexts(ir_desc, enum_, inputs, &static_conds[..]) {
            let filter_res = ctx.eval_subfilter(&filter);
            let rules_res = ctx.eval_rules(rules);
            debug!("Context{}", ctx);
            debug!("filter res: {:?}", filter_res);
            debug!("rules res: {:?}", rules_res);
            assert_eq!(filter_res, rules_res);
        }
    }

    /// Ensures similar inputs are correctly merged.
    #[test]
    fn normalize_equal_inputs() {
        let mut constraint = Constraint {
            restrict_fragile: true,
            vars: vec![],
            inputs: vec![mk_input("enum_b"), mk_input("enum_b")],
            conditions: vec![mk_enum_cond(0, &["B_0"]), mk_enum_cond(1, &["B_1"])],
        };
        let mut ir_desc = ir::IrDesc::default();
        ir::test::gen_enum("B", 2, &mut ir_desc);
        constraint.dedup_inputs(&ir_desc);
        assert_eq!(constraint.inputs.len(), 1);
    }

    /// Creates a code condition.
    fn mk_code_cond(code: &str) -> ir::Condition {
        ir::Condition::Code {
            code: ir::Code { code: code.into(), vars: vec![] },
            negate: false,
        }
    }

    /// Create an enum condition.
    fn mk_enum_cond(input: usize, values: &[&str]) -> ir::Condition {
        let values = values.iter().map(|&s| s.into()).collect();
        ir::Condition::Enum {
            input: input,
            values: values,
            negate: false,
            inverse: false,
        }
    }

    /// Creates a rule.
    fn mk_rule(conds: Vec<ir::Condition>, enum_: &str, alternatives: &[&str]) -> ir::Rule {
        ir::Rule {
            conditions: conds,
            alternatives: mk_enum_values_set(enum_, alternatives),
            set_constraints: ir::SetConstraints::default(),
        }
    }

    /// Create a input definition.
    fn mk_input(name: &str) -> ir::ChoiceInstance {
        ir::ChoiceInstance { choice: name.into(), vars: Vec::new() }
    }
}
