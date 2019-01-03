//! Defines types required at exploration runtime.
mod integer_domain;
mod integer_set;
mod range;

use proc_macro2::TokenStream;
use quote::quote;

/// Returns the token stream defining the runtime.
pub fn get() -> TokenStream {
    let range = range::get();
    let integer_set = integer_set::get();
    let integer_domain = integer_domain::get();
    quote! {
        pub trait Domain: Copy + Eq {
            /// Indicates if the domain is empty.
            fn is_failed(&self) -> bool;
            /// Indicates if the domain contains a single alternative.
            fn is_constrained(&self) -> bool;
            /// Indicates if the domain contains another.
            fn contains(&self, other: Self) -> bool;
            /// Restricts the domain to the intersection with `other`.
            fn restrict(&mut self, other: Self);

            /// Indicates if the domain has an alternatve in common with `other`.
            fn intersects(&self, mut other: Self) -> bool where Self: Sized {
                other.restrict(*self);
                !other.is_failed()
            }

            /// Indicates if the domain is equal to another domain.
            fn is(&self, mut other: Self) -> Trivalent where Self: Sized {
                other.restrict(*self);
                if other.is_failed() {
                    Trivalent::False
                } else if other == *self {
                    Trivalent::True
                } else {
                    Trivalent::Maybe
                }
            }
        }

        #range
        #integer_set
        #integer_domain
    }
}
