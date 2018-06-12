//! Prints sets of values.
use ir;
use itertools::Itertools;
use print::ast::{self, Context};
use std::collections::BTreeSet;
use utils::*;

/// Prints a `ValueSet`.
pub fn print(set: &ir::ValueSet, ctx: &Context) -> String {
    if set.is_empty() {
        format!("{}::FAILED", set.t())
    } else {
        // FIXME: May not restrict enough, mut rerun in some cases
        match *set {
            ir::ValueSet::Enum { ref enum_name, ref values, ref inputs } =>
                enum_set(enum_name, values, inputs, ctx),
            ir::ValueSet::Integer { is_full: true, ref universe, .. } => {
                // FIXME(unimplemented): take current domain into account
                full_universe(universe, ctx)
            },
            ir::ValueSet::Integer { ref cmp_inputs, ref cmp_code, .. } => {
                cmp_inputs.iter().map(|&(op, input)| {
                    (op, ctx.input_name(input).to_string())
                }).chain(cmp_code.iter().map(|&(op, ref code)| {
                    (op, ast::code(code, ctx))
                })).map(|(op, val)| {
                    // FIXME(unimplemented): take current domain into account
                    universe_fun(&set.t(), cmp_op_fun_name(op), &val, ctx)
                }).format("|").to_string()
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

fn universe_fun(t: &ir::ValueType, fun: &str, arg: &str, ctx: &Context) -> String {
    let universe = match *t {
        ir::ValueType::NumericSet(ref universe) => ast::code(universe, ctx),
        _ => format!("&{}::ALL", t),
    };
    format!("{}::{}({},{})", t, fun, universe, arg)
}

fn full_universe(t: &ir::ValueType, ctx: &Context) -> String {
    match *t {
        ir::ValueType::NumericSet(ref universe) =>
            format!("NumericSet::all({})", ast::code(universe, ctx)),
        _ => format!("{}::all()", t),
    }
}
