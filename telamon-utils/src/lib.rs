//! Generic helper functions.
extern crate fnv;
extern crate itertools;
extern crate linked_hash_map;
extern crate linked_list;
extern crate num;
extern crate serde;
#[macro_use]
extern crate serde_derive;

mod cache;
mod dag;
pub mod ndarray;
mod iterator;
pub mod multimap;
mod vec_set;
#[macro_use]
pub mod unwrap;

pub use self::cache::Cache;
pub use self::dag::Dag;
pub use self::iterator::*;
pub use self::ndarray::{NDArray, NDRange};
pub use self::vec_set::VecSet;
use fnv::FnvHasher;
use num::Integer;
use std::hash::BuildHasherDefault;

/// A fast but not secure `Hasher`.
pub type DefaultHasher = BuildHasherDefault<FnvHasher>;
/// An `HashSet` based on `DefaultHasher`.
pub type HashSet<K> = std::collections::HashSet<K, DefaultHasher>;
/// An `HashMap` based on `DefaultHasher`.
pub type HashMap<K, V> = std::collections::HashMap<K, V, DefaultHasher>;
/// A `HashMap` based on `DefaultHasher`.
pub type MultiHashMap<K, V> = self::multimap::MultiHashMap<K, V, DefaultHasher>;

/// A reference counted string, compatible with `&str`.
#[derive(Default, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Debug)]
pub struct RcStr(std::rc::Rc<String>);

impl RcStr {
    /// Creates a new reference-counted string.
    pub fn new(s: String) -> Self { RcStr(std::rc::Rc::new(s)) }
}

impl<'a> From<&'a str> for RcStr {
    fn from(s: &'a str) -> Self { Self::new(s.to_string()) }
}

impl serde::Serialize for RcStr {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl std::borrow::Borrow<str> for RcStr {
    fn borrow(&self) -> &str { &self.0 }
}

impl std::borrow::Borrow<String> for RcStr {
    fn borrow(&self) -> &String { &self.0 }
}

impl std::ops::Deref for RcStr {
    type Target = String;

    fn deref(&self) -> &String { &self.0 }
}

impl std::fmt::Display for RcStr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { self.0.fmt(f) }
}

impl PartialEq<str> for RcStr {
    fn eq(&self, other: &str) -> bool { self.0.as_ref().eq(other) }
}

impl From<String> for RcStr {
    fn from(s: String) -> RcStr { RcStr::new(s) }
}

/// Booleans enhanced with a third `Maybe` value.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Serialize)]
pub enum Trivalent {
    False,
    Maybe,
    True,
}

impl Trivalent {
    /// Returns the boolean represented.
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Trivalent::False => Some(false),
            Trivalent::Maybe => None,
            Trivalent::True => Some(true),
        }
    }

    /// Returns `true` if the underlying boolean might be `true`.
    pub fn maybe_true(&self) -> bool { *self != Trivalent::False }

    /// Returns `true` if the underlying boolean might be `false`.
    pub fn maybe_false(&self) -> bool { *self != Trivalent::True }

    /// Returns `true` if the underlying boolean is `true`.
    pub fn is_true(&self) -> bool { *self == Trivalent::True }

    /// Returns `true` if the underlying boolean is `false`.
    pub fn is_false(&self) -> bool { *self == Trivalent::False }

    /// Returns `true` if the underlying boolean might be `true` and `false`.
    pub fn is_maybe(&self) -> bool { *self == Trivalent::Maybe }
}

impl std::ops::BitAnd for Trivalent {
    type Output = Trivalent;

    fn bitand(self, rhs: Trivalent) -> Trivalent {
        match (self, rhs) {
            (Trivalent::False, _) | (_, Trivalent::False) => Trivalent::False,
            (Trivalent::Maybe, _) | (_, Trivalent::Maybe) => Trivalent::Maybe,
            (Trivalent::True, Trivalent::True) => Trivalent::True,
        }
    }
}

impl std::ops::BitOr for Trivalent {
    type Output = Trivalent;

    fn bitor(self, rhs: Trivalent) -> Trivalent {
        match (self, rhs) {
            (Trivalent::True, _) | (_, Trivalent::True) => Trivalent::True,
            (Trivalent::Maybe, _) | (_, Trivalent::Maybe) => Trivalent::Maybe,
            (Trivalent::False, Trivalent::False) => Trivalent::False,
        }
    }
}

impl std::ops::Not for Trivalent {
    type Output = Trivalent;

    fn not(self) -> Trivalent {
        match self {
            Trivalent::False => Trivalent::True,
            Trivalent::Maybe => Trivalent::Maybe,
            Trivalent::True => Trivalent::False,
        }
    }
}

/// Performs an integer divison rounded to the upper number.
pub fn div_ceil<T: Integer + Copy>(lhs: T, rhs: T) -> T {
    let (quo, rem) = lhs.div_rem(&rhs);
    if rem == T::zero() {
        quo
    } else {
        quo + T::one()
    }
}

/// Returns the log2 of a power of 2.
pub fn log2_u32(x: u32) -> Option<u32> {
    if x.count_ones() == 1 {
        Some(x.trailing_zeros())
    } else {
        None
    }
}

/// Includes a generates file into the current file.
#[macro_export]
macro_rules! generated_file {
    ($name:ident) => {
        #[cfg_attr(feature = "cargo-clippy", allow(clippy))]
        mod $name {
            include!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".rs"));
        }
    };
    (pub $name:ident) => {
        #[cfg_attr(feature = "cargo-clippy", allow(clippy))]
        pub mod $name {
            include!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".rs"));
        }
    }
}

/// Clones a pair of reference.
pub fn clone_pair<T1: Clone, T2: Clone>(p: (&T1, &T2)) -> (T1, T2) {
    (p.0.clone(), p.1.clone())
}

/// Defines equality from a key.
#[macro_export]
macro_rules! eq_from_key {
    ($ty:ty, &$key:path $(, $args:tt)*) => {
        impl<$($args),*> ::std::cmp::PartialEq for $ty {
            fn eq(&self, other: &$ty) -> bool { $key(self).eq(&$key(other)) }
        }

        impl<$($args),*> ::std::cmp::Eq for $ty {}
    };
    ($ty:ty, $key:path $(, $args:tt)*) => {
        impl<$($args),*> ::std::cmp::PartialEq for $ty {
            fn eq(&self, other: &$ty) -> bool { $key(self).eq($key(other)) }
        }

        impl<$($args),*> ::std::cmp::Eq for $ty {}
    }
}

/// Defines equality and hash from a key.
#[macro_export]
macro_rules! hash_from_key {
    ($ty:ty, $key:path $(, $args:tt)*) => {
        eq_from_key!($ty, $key $(, $args)*);

        impl<$($args),*> ::std::hash::Hash for $ty {
            fn hash<H: ::std::hash::Hasher>(&self, state: &mut H) {
                $key(self).hash(state);
            }
        }
    };
    ($ty:ty, &$key:path $(, $args:tt)*) => {
        eq_from_key!($ty, &$key $(, $args)*);

        impl<$($args),*> ::std::hash::Hash for $ty {
            fn hash<H: ::std::hash::Hasher>(&self, state: &mut H) {
                $key(self).hash(state);
            }
        }
    };
}

/// Compare two f64, panic if one of them is nan
pub fn cmp_f64(a: f64, b: f64) -> std::cmp::Ordering {
    if a.is_nan() || b.is_nan() {
        panic!("Comparing a nan !");
    }

    if a < b {
        std::cmp::Ordering::Less
    } else if (a - b).abs() < std::f64::EPSILON {
        std::cmp::Ordering::Equal
    } else {
        std::cmp::Ordering::Greater
    }
}

/// A trait that implements useful methods on builders.
pub trait BuilderTrait: Sized {
    /// Runs the closure if the bool is true.
    fn doif<F>(&mut self, flag: bool, f: F) -> &mut Self
        where F: FnOnce(&mut Self) -> &mut Self
    {
        if flag { f(self) } else { self }
    }
}
