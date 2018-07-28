//! Prints sets of values.
use ir;
use itertools::Itertools;
use print;
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
            ir::ValueSet::Enum { ref enum_name, ref values, ref inputs } =>
                enum_set(enum_name, values, inputs, ctx),
            ir::ValueSet::Integer { is_full: true, .. } => {
                render!(value_type/full_domain, t)
            },
            ir::ValueSet::Integer { ref cmp_inputs, ref cmp_code, .. } => {
                cmp_inputs.iter().map(|&(op, input)| {
                    let input_type = ctx.input(input).value_type(&ctx.ir_desc);
                    (op, ctx.input_name(input).to_string(), input_type)
                }).chain(cmp_code.iter().map(|&(op, ref code)| {
                    (op, ast::code(code, ctx), ir::ValueType::Constant)
                })).map(|(op, arg, arg_t)| {
                    // TODO(span): parse in the lexer rather than here
                    let from = unwrap!(arg.parse());
                    print::value::integer_domain_constructor(
                        &set.t(), op, from, &arg_t, ctx)
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
