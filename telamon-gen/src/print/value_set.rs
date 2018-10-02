//! Prints sets of values.
use ir;
use itertools::Itertools;
use print::ast::{self, Context};
use std::collections::BTreeSet;
use utils::*;

/// Prints a `ValueSet`.
pub fn print(set: &ir::ValueSet, ctx: &Context) -> String {
    let t = ast::ValueType::new(set.t(), ctx);
    if set.is_empty() {
        format!("{}::FAILED", t)
    } else {
        // FIXME: The set might not be restriceted enough when combining range sets.
        // - should limit to the current domain.
        // TODO(cc_perf): might be faster to limit to the current domain rather than the
        // full universe.
        match *set {
            ir::ValueSet::Enum {
                ref enum_name,
                ref values,
                ref inputs,
            } => enum_set(enum_name, values, inputs, ctx),
            ir::ValueSet::Integer { is_full: true, .. } => {
                render!(value_type / full_domain, t)
            }
            ir::ValueSet::Integer {
                ref cmp_inputs,
                ref cmp_code,
                ..
            } => cmp_inputs
                .iter()
                .map(|&(op, input)| (op, ctx.input_name(input).to_string()))
                .chain(
                    cmp_code
                        .iter()
                        .map(|&(op, ref code)| (op, ast::code(code, ctx))),
                )
                .map(|(op, val)| universe_fun(&t, cmp_op_fun_name(op), &val))
                .format("|")
                .to_string(),
        }
    }
}

/// Prints a set of enum values.
fn enum_set(
    name: &str,
    values: &BTreeSet<RcStr>,
    inputs: &BTreeSet<(usize, bool, bool)>,
    ctx: &Context,
) -> String
{
    let values = values
        .iter()
        .map(|x| format!("{}::{}", name, x))
        .collect_vec();
    let inputs = inputs
        .iter()
        .map(|&(input, negate, inverse)| {
            let neg_str = if negate { "!" } else { "" };
            let inv_str = if inverse { ".inverse()" } else { "" };
            let var = ctx.input_name(input);
            format!("{}{}{}", neg_str, var, inv_str)
        })
        .collect_vec();
    values.into_iter().chain(inputs).format("|").to_string()
}

/// Returns the function to call to implement the given operator.
fn cmp_op_fun_name(op: ir::CmpOp) -> &'static str {
    match op {
        ir::CmpOp::Lt => "new_lt",
        ir::CmpOp::Gt => "new_gt",
        ir::CmpOp::Leq => "new_leq",
        ir::CmpOp::Geq => "new_geq",
        ir::CmpOp::Eq => "new_eq",
        ir::CmpOp::Neq => panic!("neq operations on numeric domains are not supported"),
    }
}

fn universe_fun(t: &ast::ValueType, fun: &str, arg: &str) -> String {
    match t {
        ast::ValueType::NumericSet(universe) => {
            format!("{}::{}({},{})", t, fun, universe, arg)
        }
        t => format!("{}::{}(&{}::ALL,{})", t, fun, t, arg),
    }
}
