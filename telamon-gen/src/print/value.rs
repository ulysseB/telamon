//! Prints the manipulation of individual domains.
use ir;
use print;
use proc_macro2::{Ident, Span, TokenStream};
use quote;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A value that a domain can take.
#[derive(Debug, Clone)]
pub struct Value {
    tokens: TokenStream,
    value_type: ir::ValueType,
}

impl Value {
    /// Create a new value.
    pub fn new(tokens: TokenStream, value_type: ir::ValueType) -> Self {
        Value { tokens, value_type }
    }

    /// Creates a new value with a unique name.
    pub fn new_ident(prefix: &str, value_type: ir::ValueType) -> Self {
        lazy_static! { static ref NEXT_ID: AtomicUsize = AtomicUsize::new(0); }
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        // TODO(span): get the real span from the lexer
        let ident = Ident::new(&format!("{}_{}", prefix, id), Span::call_site());
        Self::new(quote!(#ident), value_type)
    }

    /// Creates a value with the given name.
    pub fn ident(name: &str, value_type: ir::ValueType) -> Self {
        // TODO(span): get the real span from the lexer
        let ident = Ident::new(name, Span::call_site());
        Self::new(quote!(#ident), value_type)
    }

    /// Creates a new user-provided constant value.
    pub fn new_const(code: &ir::Code, ctx: &print::Context) -> Self {
        // TODO(cleanup): parse in the lexer rather than here
        let tokens = unwrap!(print::ast::code(code, ctx).parse());
        Self::new(tokens, ir::ValueType::Constant)
    }

    /// Fetches a value from the store. If `get_old` is true, only take into account
    /// decisions that have been propagated.
    pub fn from_store(
        choice_instance: &ir::ChoiceInstance,
        get_old: bool,
        ctx: &print::Context
    ) -> Self {
        let getter = print::store::getter_name(&choice_instance.choice, get_old);
        let ids = print::choice::ids(choice_instance, ctx);
        let diff = if get_old { quote!(diff) } else { quote!() };
        let tokens = quote!(store.#getter(#ids#diff));
        Self::new(tokens, choice_instance.value_type(ctx.ir_desc))
    }

    /// Creates a new value that has the same type that this one, but with an identifier
    /// in place of the expression.
    pub fn with_name(&self, name: &str) -> Self {
        Self::ident(name, self.value_type().clone())
    }

    /// Returns the type taken by the value.
    pub fn value_type(&self) -> &ir::ValueType { &self.value_type }

    /// Returns the antisymmetric of the value instead of the value.
    pub fn inverse(&mut self) {
        self.tokens = {
            let tokens = &self.tokens;
            quote!((#tokens).inverse())
        };
    }

    /// Returns the complement of the value instead of the value.
    pub fn negate(&mut self) {
        self.tokens = {
            let tokens = &self.tokens;
            quote!(!(#tokens))
        };
    }

    /// Returns the minimum of an integer domain.
    pub fn get_min(&self, ctx: &print::Context) -> Self {
        let universe = universe(self.value_type(), ctx);
        let tokens = quote!(NumSet::min(&#self, #universe));
        Value::new(tokens, ir::ValueType::Constant)
    }

    /// Returns the maximum of an integer domain.
    pub fn get_max(&self, ctx: &print::Context) -> Self {
        let universe = universe(self.value_type(), ctx);
        let tokens = quote!(NumSet::max(&#self, #universe));
        Value::new(tokens, ir::ValueType::Constant)
    }
}

impl quote::ToTokens for Value {
    fn to_tokens(&self, stream: &mut TokenStream) {
        self.tokens.to_tokens(stream)
    }
}

/// Prints the universe of a `ValueType`.
pub fn universe(value_type: &ir::ValueType, ctx: &print::Context) -> Value {
    match value_type {
        ir::ValueType::Enum(..) => panic!("only intger domains have a universe"),
        ir::ValueType::Range { .. } | ir::ValueType::Constant => {
            Value::new(quote!(&()), ir::ValueType::Constant)
        }
        ir::ValueType::NumericSet(universe) => Value::new_const(universe, ctx),
    }
}

/// Calls a constructor for an integer domain. `constructor_op` indicates which
/// constructor should be called.
pub fn integer_domain_constructor(
    constructor_op: ir::CmpOp,
    from: &Value,
    to_type: ir::ValueType,
    ctx: &print::Context
) -> Value {
    let constructor = constructor_from_op(constructor_op);
    let to_universe = universe(&to_type, ctx);
    let from_universe = universe(from.value_type(), ctx);
    let tokens = quote!(#to_type::#constructor(#to_universe, #from, #from_universe));
    Value::new(tokens, to_type)
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
    // TODO(span): get the real span from the lexer
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
        // TODO(span): get the real span from the lexer
        Ident::new(name, Span::call_site()).to_tokens(stream);
    }
}
