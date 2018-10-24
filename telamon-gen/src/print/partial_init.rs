/// Prints code that handles the insertion of new objects during the search.
use ir::{self, SetRef};
use print;
use proc_macro2::{Ident, Span, TokenStream};
use quote::ToTokens;

/// Runs necessary filters after new objects are allocated. This only runs new filters on
/// old objects.
pub fn filters(ir_desc: &ir::IrDesc) -> TokenStream {
    ir_desc
        .set_defs()
        .map(|(set_def, on_new_object)| {
            if on_new_object.filter.is_empty() {
                return quote!();
            }
            let obj = Ident::new("obj", Span::call_site());
            let arg = set_def.arg().map(|_| Ident::new("arg", Span::call_site()));
            // We assume the argument variable is Arg(1) since this is what is generated
            // by `ir::adapt_to_var_context`.
            let set = ir::Set::new(set_def, set_def.arg().map(|_| ir::Variable::Arg(1)));
            // We must use the old template facilities to print the filter call so
            // generate variables for the old printer.
            let mut vars = vec![(print::ast::Variable::with_name("obj"), &set)];
            if let Some(arg) = set_def.arg() {
                vars.push((print::ast::Variable::with_name("arg"), arg));
            }
            let body = on_new_object
                .filter
                .iter()
                .map(|(foralls, constraints, filter_call)| {
                    let ctx =
                        print::Context::new_outer(ir_desc, &vars, foralls);
                    let mut conflicts = vars.iter()
                        .map(|(var, set)| print::ast::Conflict::new(var.clone(), set))
                        .collect();
                    let loop_vars = (0..foralls.len()).map(ir::Variable::Forall);
                    let loop_nest = print::ast::LoopNest::new(
                        loop_vars, &ctx, &mut conflicts, false);
                    let filter_call = print::choice::RemoteFilterCall::new(
                        filter_call, conflicts, foralls.len(), &ctx);
                    let set_constraints = print::ast::SetConstraint::new(
                        constraints, &ctx);
                    let printed_code = render!(partial_init_filters, <'a>,
                        loop_nest: print::ast::LoopNest<'a> = loop_nest,
                        set_constraints: Vec<print::ast::SetConstraint<'a>> = set_constraints,
                        filter_call: print::choice::RemoteFilterCall<'a> = filter_call);
                    let tokens: TokenStream = printed_code.parse().unwrap();
                    tokens
                }).collect();
            iter_new_objects(set_def, &obj, arg.as_ref(), &body)
        }).collect()
}

/// Iterates on the new objects of `set` and executes `body` for each of them.
///
/// Warning: this method differs from the partial_iterators in `print::store`: it may
/// call multiple times the same filter. This is because this makes it easier to handle
/// reversed sets: we can register them in the non-reversed set instead. This doesn't
/// work if the non-reversed set iterator skips some objects because it thinks they are
/// handled by the reverse set. To factorize those functions, we would need to consider
/// relations instead of parametric sets, thus removing the need to have reversed sets.
fn iter_new_objects(
    set: &ir::SetDef,
    obj: &Ident,
    arg: Option<&Ident>,
    body: &TokenStream,
) -> TokenStream {
    assert_eq!(set.arg().is_some(), arg.is_some());
    let new_objs_iter = print::set::iter_new_objects(&set);
    let obj_id = print::set::ObjectId::new("obj_id", set);
    let arg_id = set
        .arg()
        .map(|set| print::set::ObjectId::new("arg_id", set.def()));
    let obj_arg_pattern = arg_id
        .as_ref()
        .map(|arg| quote! { (#arg, #obj_id) })
        .unwrap_or(obj_id.clone().into_token_stream());
    let arg_def = arg_id.as_ref().map(|id| {
        let getter = id.fetch_object(None);
        quote!{ let #arg = #getter; }
    });
    let arg = arg.map(|x| x.clone().into());
    let obj_getter = obj_id.fetch_object(arg.as_ref());
    quote! {
        for &#obj_arg_pattern in #new_objs_iter.iter() {
            #arg_def
            let #obj = #obj_getter;
            #body
        }
    }
}
