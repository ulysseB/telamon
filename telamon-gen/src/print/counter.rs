//! Prints counter manipulation methods.
use ir;
use print;
use proc_macro2::TokenStream;
use quote;

/// Prints the value of a counter increment. If `use_old` is true, only takes into account
/// decisions that have been propagated.
pub fn increment_amount(
    value: &ir::CounterVal,
    use_old: bool,
    ctx: &print::Context
) -> print::Value {
    match value {
        ir::CounterVal::Code(code) => print::Value::new_const(code, ctx),
        ir::CounterVal::Choice(choice) => print::Value::from_store(choice, use_old, ctx),
    }
}

/// Returns `+=` or `*=` depending on if the counter is additive or multiplicative.
fn increment_operator(kind: ir::CounterKind) -> TokenStream {
    match kind {
        ir::CounterKind::Add => quote!(+=),
        ir::CounterKind::Mul => quote!(*=),
    }
}

/// Returns the operator performing the inverse operation of `op`.
fn inverse_operator(op: ir::CounterKind) -> TokenStream {
    match op {
        ir::CounterKind::Add => quote!(-),
        ir::CounterKind::Mul => quote!(/),
    }
}

impl quote::ToTokens for ir::CounterKind {
    fn to_tokens(&self, stream: &mut TokenStream) {
        match *self {
            ir::CounterKind::Add => quote!(+).to_tokens(stream),
            ir::CounterKind::Mul => quote!(-).to_tokens(stream),
        }
    }
}

/// Prints code to rescrict the increment amount if necessary. `delayed` indicates if the
/// actions should be taken immediately, or pushed into an action list.
// TODO(cleanup): print the full method instead of just part of its body.
// TODO(cleanup): simplify the template
pub fn restrict_incr_amount(
    incr_amount: &ir::ChoiceInstance,
    current_incr_amount: &print::Value,
    op: ir::CounterKind,
    delayed: bool,
    ctx: &print::Context
) -> TokenStream {
    let max_val = print::ValueIdent::new("max_val", ir::ValueType::Constant);
    let neg_op = inverse_operator(op);
    let restricted_value_type = incr_amount.value_type(ctx.ir_desc).full_type();
    let restricted_value = print::value::integer_domain_constructor(
        ir::CmpOp::Leq, &max_val.into(), restricted_value_type, ctx);
    let restricted_value_name = restricted_value.create_ident("value");
    let apply_restriction = print::choice::restrict(
        incr_amount, &restricted_value_name.into(), delayed, ctx);
    let min_incr_amount = current_incr_amount.get_min(ctx);
    quote! {
        else if incr_status.is_true() {
            let max_val = new_values.max #neg_op current.min #op #min_incr_amount;
            let value = #restricted_value;
            #apply_restriction
        }
    }
}

/// Prints a method that computes the value of a counter, only taking propagated actions
/// into account.
// TODO(cleanup): print the full method instead of just part of its body.
// TODO(cleanup): simplify the template
pub fn compute_counter_body(
    value: &ir::CounterVal,
    incr: &ir::ChoiceInstance,
    incr_condition: &ir::ValueSet,
    op: ir::CounterKind,
    visibility: ir::CounterVisibility,
    ctx: &print::Context,
) -> TokenStream {
    let value_getter = increment_amount(value, true, ctx);
    let value: print::Value = value_getter.create_ident("value").into();
    let value_min = value.get_min(ctx);
    let value_max = value.get_max(ctx);
    let incr_getter = print::Value::from_store(incr, true, ctx);
    let incr_condition = print::value_set::print(incr_condition, ctx);
    let op_eq = increment_operator(op);

    let update_max = if visibility == ir::CounterVisibility::NoMax {
        None
    } else {
        Some(quote! {
            if (#incr_condition).intersects(incr) { counter_val.max #op_eq #value_max; }
        })
    };
    quote! {
        let value = #value_getter;
        let incr = #incr_getter;
        #update_max
        if (#incr_condition).contains(incr) { counter_val.min #op_eq #value_min; }
    }
}

