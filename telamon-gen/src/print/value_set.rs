//! Prints sets of values.
use ir;
use itertools::Itertools;
use print::ast::{self, Context};
use std::collections::BTreeSet;
use utils::*;

/// Prints a `ValueSet`.
// FIXME: unimplemented!() two possibilities:
// 1) cast the set to the target each time it is referenced
// FIXME: unimplemented!() limit the universe to the existing domain
pub fn print(set: &ir::ValueSet, ctx: &Context) -> String {
    if set.is_empty() {
        format!("{}::FAILED", set.t())
    } else {
        match *set {
            ir::ValueSet::Enum { ref enum_name, ref values, ref inputs } =>
                enum_set(enum_name, values, inputs, ctx),
            ir::ValueSet::Integer { is_full: true, ref universe, .. } =>
                universe_fun(universe, "all", ctx),
            ir::ValueSet::Integer { ref cmp_inputs, ref cmp_code, ref universe, .. } => {
                let inputs = cmp_inputs.iter().map(|&(op, input)| {
                    (op, ctx.input_name(input).to_string())
                }).collect_vec();
                let code = cmp_code.iter().map(|&(op, ref code)| {
                    (op, ast::code(code, ctx))
                }).collect_vec();
                // FIXME: May not restrict enough
                // -> must intersect each component of the range with he current set ?
                //    - not enough, must rerun in some cases
                inputs.into_iter().map(|(op, val)| match op {
                    ir::CmpOp::Lt => format!("Range::new_lt({}.max)", val),
                    ir::CmpOp::Gt => format!("Range::new_gt({}.min)", val),
                    ir::CmpOp::Leq => format!("Range::new_leq({}.max)", val),
                    ir::CmpOp::Geq => format!("Range::new_geq({}.min)", val),
                    ir::CmpOp::Eq => format!("{}", val),
                    ir::CmpOp::Neq => format!("Range::ALL"), // FIXME: may not restrict enough
                }).chain(code.into_iter().map(|(op, val)| match op {
                    ir::CmpOp::Lt => format!("Range::new_lt({})", val),
                    ir::CmpOp::Gt => format!("Range::new_gt({})", val),
                    ir::CmpOp::Leq => format!("Range::new_leq({})", val),
                    ir::CmpOp::Geq => format!("Range::new_geq({})", val),
                    ir::CmpOp::Eq => format!("Range::new_eq({})", val),
                    ir::CmpOp::Neq => format!("Range::ALL"), // FIXME: may not restrict enough
                })).format("|").to_string()
                // FIXME: Is Or defined ?
            },
        }
    }
}

/// Prints a set of enum values.
fn enum_set(name: &str,
            values: &BTreeSet<RcStr>,
            inputs: &BTreeSet<(usize, bool, bool)>,
            ctx: &Context) -> String {
    let values = values.iter().map(|x| {
        format!("{}::{}", name, x)
    }).collect_vec();
    let inputs = inputs.iter().map(|&(input, negate, inverse)| {
        let neg_str = if negate { "!" } else { "" };
        let inv_str = if inverse { ".inverse()" } else { "" };
        let var = ctx.input_name(input);
        format!("{}{}{}", neg_str, var, inv_str)
    }).collect_vec();
    values.into_iter().chain(inputs).format("|").to_string()
}

fn universe_fun(t: &ir::ValueType, fun: &str, ctx: &Context) -> String {
    let universe = match *t {
        ir::ValueType::NumericSet(ref universe) =>
            format!("{}, ", ast::code(universe, ctx)),
        _ => String::new(),
    };
    format!("{}::{}({})", t, fun, universe)
}
