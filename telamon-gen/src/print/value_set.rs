//! Prints sets of values.
use ir;
use itertools::Itertools;
use print;
use print::ast::{self, Context};
use proc_macro2::TokenStream;
use std::collections::BTreeSet;
use std::borrow::Cow;
use utils::*;

/// Prints a `ValueSet`.
pub fn print(set: &ir::ValueSet, ctx: &Context) -> TokenStream {
    let t = ast::ValueType::new(set.t(), ctx);
    if set.is_empty() {
        // TODO(cleanup): return a token stream instead of parsing
        unwrap!(format!("{}::FAILED", t).parse())
    } else {
        match *set {
            ir::ValueSet::Enum { ref enum_name, ref values, ref inputs } => {
                // TODO(cleanup): return a token stream instead of parsing
                unwrap!(enum_set(enum_name, values, inputs, ctx).parse())
            }
            ir::ValueSet::Integer { is_full: true, .. } => {
                // TODO(cleanup): return a token stream instead of parsing
                unwrap!(render!(value_type/full_domain, t).parse())
            },
            ir::ValueSet::Integer { ref cmp_inputs, ref cmp_code, .. } => {
                // FIXME: The set might not be restriceted enough when combining range
                // sets: should limit to the current domain.
                let parts = cmp_inputs.iter().map(|&(op, input)| {
                    (op, Cow::Borrowed(ctx.input(input)))
                }).chain(cmp_code.iter().map(|&(op, ref code)| {
                    (op, Cow::Owned(print::Value::new_const(code, ctx)))
                })).map(|(op, from)| {
                    print::value::integer_domain_constructor(op, &from, set.t(), ctx)
                });
                quote!(#(#parts)|*)
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
