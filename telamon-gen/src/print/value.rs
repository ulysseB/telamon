//! Prints the manipulation of individual domains.
use ir;
use print::ast;
use proc_macro2::{Ident, Span, TokenStream};
use quote;

/// Prints the universe of a `ValueType`.
pub fn universe(value_type: &ir::ValueType, ctx: &ast::Context) -> TokenStream {
    match value_type {
        ir::ValueType::Enum(..) => panic!("only intger domains have a universe"),
        ir::ValueType::Range { .. } | ir::ValueType::Constant => quote!(&()),
        ir::ValueType::NumericSet(universe) => {
            // TODO(cleanup): parse the piece of code during parsing.
            unwrap!(ast::code(universe, ctx).parse())
        }
    }
}

impl quote::ToTokens for ir::ValueType {
    fn to_tokens(&self, stream: &mut TokenStream) {
        let name = match self {
            ir::ValueType::Enum(name) => name,
            ir::ValueType::Range { is_half: false } => "Range",
            ir::ValueType::Range { is_half: true } => "HalfRange",
            ir::ValueType::NumericSet(..) => "NumericSet",
            ir::ValueType::Constant => {
                panic!("the type of user-provided constants cannot is unspecified")
            }
        };
        Ident::new(name, Span::call_site()).to_tokens(stream);
    }
}
