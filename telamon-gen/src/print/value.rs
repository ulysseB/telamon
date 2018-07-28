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

/// Calls a constructor for an integer domain. `constructor_op` indicates which
/// constructor should be called.
pub fn integer_domain_constructor(to_type: &ir::ValueType,
                                  constructor_op: ir::CmpOp,
                                  from: TokenStream,
                                  from_type: &ir::ValueType,
                                  ctx: &ast::Context) -> TokenStream {
    let constructor = constructor_from_op(constructor_op);
    let to_universe = universe(to_type, ctx);
    let from_universe = universe(from_type, ctx);
    quote!(#to_type::#constructor(#to_universe, #from, #from_universe))
}

/// Returns the name of te integer domain constructor to call to obatain the domain
/// that respects the condition `op` with regard to another domain.
pub fn constructor_from_op(op: ir::CmpOp) -> Ident {
    let name = match op {
        ir::CmpOp::Lt => "new_lt",
        ir::CmpOp::Gt => "new_gt",
        ir::CmpOp::Leq => "new_leq",
        ir::CmpOp::Geq => "new_geq",
        ir::CmpOp::Eq => "new_eq",
        ir::CmpOp::Neq => panic!("neq operations on numeric domains are not supported"),
    };
    // TODO(cleanup): get the real span from the lexer
    Ident::new(name, Span::call_site())
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
        // TODO(cleanup): get the real span from the lexer
        Ident::new(name, Span::call_site()).to_tokens(stream);
    }
}
