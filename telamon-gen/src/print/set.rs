//! Manipulation of sets and their items.
use ir;
use proc_macro2::{Delimiter, Group, Ident, Span, TokenStream, TokenTree};
use quote;

/// Lists the new objects of the given set.
pub fn iter_new_objects(set: &ir::SetDef) -> TokenTree {
    // TODO(cleanup): parse in the parser instead of after substitution.
    let expr = set.attributes()[&ir::SetDefKey::NewObjs].replace("$objs", "new_objs");
    Group::new(Delimiter::Parenthesis, unwrap!(expr.parse())).into()
}

/// Represents an expression that holds the ID of an object.
#[derive(Clone)]
pub struct ObjectId<'a> {
    id: TokenTree,
    set: &'a ir::SetDef,
}

impl<'a> ObjectId<'a> {
    /// Creates a new variable to hold an object ID.
    pub fn new(name: &str, set: &'a ir::SetDef) -> Self {
        ObjectId {
            id: TokenTree::Ident(Ident::new(name, Span::call_site())),
            set,
        }
    }

    /// Creates an expression that returns the id of an object.
    pub fn from_object(object: &TokenTree, set: &'a ir::SetDef) -> Self {
        let expr = set.attributes()[&ir::SetDefKey::IdGetter]
            .replace("$fun", "ir_instance")
            .replace("$item", &object.to_string());
        let id = Group::new(Delimiter::None, unwrap!(expr.parse())).into();
        ObjectId { id, set }
    }

    /// Returns code that fetches the object corresponding to the ID.
    pub fn fetch_object(&self, arg: Option<&TokenTree>) -> TokenTree {
        // TODO(cleanup): parse in the parser instead of after substitution.
        let mut expr = self.set.attributes()[&ir::SetDefKey::ItemGetter]
            .replace("$fun", "ir_instance")
            .replace("$id", &self.id.to_string());
        if let Some(arg) = arg {
            expr = expr.replace("$var", &arg.to_string());
        }
        Group::new(Delimiter::None, unwrap!(expr.parse())).into()
    }
}

impl<'a> quote::ToTokens for ObjectId<'a> {
    fn to_tokens(&self, stream: &mut TokenStream) {
        self.id.to_tokens(stream)
    }
}
