//! Prints sets of values.
use crate::ir;
use crate::print;
use crate::print::ast::{self, Context};
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::quote;
use std::collections::BTreeSet;
use utils::*;

/// Prints a `ValueSet`.
pub fn print(set: &ir::ValueSet, ctx: &Context) -> TokenStream {
    if set.is_empty() {
        let t = set.t();
        quote! { #t::FAILED }
    } else {
        match *set {
            ir::ValueSet::Enum {
                ref enum_name,
                ref values,
                ref inputs,
            } => {
                // TODO(cleanup): return a token stream instead of parsing
                unwrap!(enum_set(enum_name, values, inputs, ctx).parse())
            }
            ir::ValueSet::Integer { is_full: true, .. } => {
                // TODO(cleanup): return a token stream instead of parsing
                let t = ast::ValueType::new(set.t(), ctx);
                unwrap!(render!(value_type / full_domain, t).parse())
            }
            ir::ValueSet::Integer {
                ref cmp_inputs,
                ref cmp_code,
                ..
            } => {
                // FIXME: The set might not be restricted enough when combining range
                // sets: should limit to the current domain.
                let parts = cmp_inputs
                    .iter()
                    .map(|&(op, input)| (op, ctx.input(input).clone().into()))
                    .chain(
                        cmp_code.iter().map(|&(op, ref code)| {
                            (op, print::Value::new_const(code, ctx))
                        }),
                    )
                    .map(|(op, from)| {
                        print::value::integer_domain_constructor(op, &from, set.t(), ctx)
                    });
                quote!(#(#parts)|*)
            }
        }
    }
}

/// Prints a set of enum values.
fn enum_set(
    name: &str,
    values: &BTreeSet<RcStr>,
    inputs: &BTreeSet<(usize, bool, bool)>,
    ctx: &Context,
) -> String {
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
