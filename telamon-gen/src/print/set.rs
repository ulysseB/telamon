//! Manipulation of sets and their items.
use ir;
use proc_macro2::TokenStream;

/// Prints the ID of an object.
pub fn id<'a, S: ir::SetRef<'a>>(object: TokenStream, set: S) -> TokenStream {
    // TODO(cleanup): parse beforehand
    let expr = set.def().attributes()[&ir::SetDefKey::IdGetter]
        .replace("$fun", "ir_instance")
        .replace("$item", &object.to_string());
    unwrap!(expr.parse())
}
