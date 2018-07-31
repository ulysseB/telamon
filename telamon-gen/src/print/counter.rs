//! Prints counter manipulation methods.
use ir;
use print;
use proc_macro2::TokenStream;

/// Prints a method that computes the value of a counter, only taking propagated actions
/// into account.
// TODO(cleanup): print the full method instead of just part of its body.
pub fn compute_counter_body(
    value: &ir::CounterVal,
    incr: &ir::ChoiceInstance,
    incr_condition: &ir::ValueSet,
    op: ir::CounterKind,
    visibility: ir::CounterVisibility,
    ctx: &print::Context,
) -> TokenStream {
    let value_getter = counter_value(value, ctx);
    let value = print::Value::ident("value", value_getter.value_type().clone());
    let value_min = value.get_min(ctx);
    let value_max = value.get_max(ctx);
    let incr_getter = print::Value::from_store(incr, true, ctx);
    let incr_condition = print::value_set::print(incr_condition, ctx);
    let op_eq = increment_operator(op);

    let update_max = if visibility == ir::CounterVisibility::NoMax {
        TokenStream::default()
    } else {
        quote! {
            if (#incr_condition).intersects(incr) { counter_val.max #op_eq #value_max; }
        }
    };
    quote! {
        let value = #value_getter;
        let incr = #incr_getter;
        #update_max
        if (#incr_condition).contains(incr) { counter_val.min #op_eq #value_min; }
    }
}

/// Prints the value of a counter increment. Only takes into account decisions that have
/// been propagated.
fn counter_value(value: &ir::CounterVal, ctx: &print::Context) -> print::Value {
    match value {
        ir::CounterVal::Code(code) => print::Value::new_const(code, ctx),
        ir::CounterVal::Choice(choice) => print::Value::from_store(choice, true, ctx),
    }
}

/// Returns `+=` or `*=` depending on if the counter is additive or multiplicative.
fn increment_operator(kind: ir::CounterKind) -> TokenStream {
    match kind {
        ir::CounterKind::Add => quote!(+=),
        ir::CounterKind::Mul => quote!(*=),
    }
}
