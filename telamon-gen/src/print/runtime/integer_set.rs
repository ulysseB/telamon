//! Define a domain to represent bounded sets of disjoint integers.
use proc_macro2::TokenStream;
use quote::quote;
// FIXME: move insert to the domain
// FIXME: move all to IntegerDomain

/// Defines the `NumericSet` type.
pub fn get() -> TokenStream {
    quote! {
        /// A small set (up to 16 values) represented using a bitset.  The actual values are stored
        /// separately in an `universe` (a vector of values), which needs to be provided separately
        /// for most meaningful uses of the `NumericSet`.
        ///
        /// This strategy avoids duplicating the set of actual values for each instance of the
        /// `NumericSet`; in particular, this allows us to have a single universe vector shared
        /// across all instances of a Telamon function, but each have a separate `NumericSet` which
        /// can be influenced by the choices and constraints.
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, Ord, PartialOrd)]
        #[repr(C)]
        pub struct NumericSet {
            enabled_values: u16
        }

        impl std::fmt::Debug for NumericSet {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "NS{{{}}}", self.list().map(|bit| bit.enabled_values).format(", "))
            }
        }

        /// Helper struct for printing numeric sets with `format!` and `{}`.
        ///
        /// Numeric sets represent values using a compact bitset representation which is ill-suited
        /// for display since it doesn't know the actual values represented by the set.  This
        /// struct embeds the universe, which allows it to implement the [`Display`] trait in a
        /// satisfactory way.
        ///
        /// [`Display`]: std::fmt::Display
        pub struct Display<'a> {
            numeric_set: &'a NumericSet,
            universe: &'a [u32],
        }

        impl<'a> std::fmt::Debug for Display<'a> {
            fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Debug::fmt(self.numeric_set, fmt)
            }
        }

        impl<'a> std::fmt::Display for Display<'a> {
            fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "{{{}}}", self.numeric_set.list_values(self.universe).format(", "))
            }
        }

        #[allow(dead_code)]
        impl NumericSet {
            pub const FAILED: Self = NumericSet { enabled_values: 0 };

            pub const MAX_LEN: usize = 16;

            /// Returns the set containing all the possibilities. Assumes the universe is
            /// sorted.
            pub fn all(univers: &[u32]) -> Self {
                NumericSet { enabled_values: (1 << univers.len()) - 1 }
            }

            /// Inserts alternatives into the domain. Both domains should be from the same
            /// universe.
            pub fn insert(&mut self, other: NumericSet) {
                self.enabled_values |= other.enabled_values;
            }

            /// Lists the constrained domains contained in `self`.
            pub fn list(&self) -> impl Iterator<Item=Self> {
                let enabled_values = self.enabled_values;
                (0..Self::MAX_LEN).map(|x| 1 << x)
                    .filter(move |x| (enabled_values & x) != 0)
                    .map(|bit| NumericSet { enabled_values: bit })
            }

            /// Lists the values this domain can take.
            pub fn list_values<'a>(&self, universe: &'a [u32])
                -> impl Iterator<Item=u32> + 'a
            {
                let enabled_values = self.enabled_values;
                universe.iter().enumerate()
                    .filter(move |(i, _)| ((1u16 << i) & enabled_values) != 0)
                    .map(|(_, &v)| v)
            }

            /// Returns the set containing the values of `universe` not in `self`.
            fn complement(&self, universe: &[u32]) -> Self {
                let mask = NumericSet::all(universe);
                NumericSet { enabled_values: !self.enabled_values & mask.enabled_values }
            }

            /// Returns an object that implements [`Display`] for printing numeric sets with the
            /// corresponding universe values.
            ///
            /// [`Display`]: std::fmt::Display
            pub fn display<'a>(&'a self, universe: &'a [u32]) -> Display<'a> {
                Display { numeric_set: self, universe }
            }

            /// Returns the greatest common divisor of possible values.
            pub fn gcd(&self, universe: &[u32]) -> u32 {
                self.list_values(universe).fold1(num::integer::gcd).unwrap_or(1)
            }

            /// Returns the least common multiple of possible values.
            pub fn lcm(&self, universe: &[u32]) -> u32 {
                self.list_values(universe).fold1(num::integer::lcm).unwrap_or(1)
            }
        }

        impl Domain for NumericSet {
            fn is_failed(&self) -> bool { self.enabled_values == 0 }

            fn is_constrained(&self) -> bool { self.enabled_values.count_ones() == 1 }

            fn contains(&self, other: NumericSet) -> bool {
                (self.enabled_values & other.enabled_values) == other.enabled_values
            }

            fn restrict(&mut self, other: NumericSet) {
                self.enabled_values &= other.enabled_values;
            }
        }

        impl NumSet for NumericSet {
            type Universe = [u32];

            fn min_value(&self, universe: &[u32]) -> u32 {
                if self.is_failed() {
                    0
                } else {
                    universe[self.enabled_values.trailing_zeros() as usize]
                }
            }

            fn max_value(&self, universe: &[u32]) -> u32 {
                if self.is_failed() {
                    std::u32::MAX
                } else {
                    let leading_zeros = self.enabled_values.leading_zeros() as usize;
                    universe[Self::MAX_LEN - leading_zeros - 1]
                }
            }

            fn into_num_set(
                &self,
                universe: &[u32],
                num_set_universe: &[u32]
            ) -> NumericSet {
                let mut self_idx = 0;
                let mut enabled_values = 0;
                'values: for (idx, &value) in num_set_universe.iter().enumerate() {
                    while {
                        if self_idx >= universe.len() { break 'values; }
                        (self.enabled_values & (1 << self_idx)) == 0
                            || universe[self_idx] < value
                    } { self_idx += 1; }
                    if universe[self_idx] == value { enabled_values |= 1 << idx; }
                }
                NumericSet { enabled_values }
            }

            fn is_neq<D: NumSet>(&self, universe: &[u32],
                              other: D, other_universe: &D::Universe) -> bool {
                self.min_value(universe) > other.max_value(other_universe)
                    || self.max_value(universe) < other.min_value(other_universe)
                    || other.into_num_set(other_universe, universe).is_failed()
            }
        }

        impl NumDomain for NumericSet {
            fn new_gt<D: NumSet>(universe: &[u32],
                                 min: D, min_universe: &D::Universe) -> Self {
                let start = universe.binary_search(&min.min_value(min_universe))
                    .map(|x| x+1).unwrap_or_else(|x| x);
                let len = universe.len() - start;
                let enabled_values = ((1 << len) - 1) << start;
                NumericSet { enabled_values }
            }

            fn new_lt<D: NumSet>(universe: &[u32],
                                 max: D, max_universe: &D::Universe) -> Self {
                let len = universe.binary_search(&max.max_value(max_universe))
                    .unwrap_or_else(|x| x);
                let enabled_values = (1 << len) - 1;
                NumericSet { enabled_values }
            }

            fn new_geq<D: NumSet>(universe: &[u32],
                                  min: D, min_universe: &D::Universe) -> Self {
                let start = universe.binary_search(&min.min_value(min_universe))
                    .unwrap_or_else(|x| x);
                let len = universe.len() - start;
                let enabled_values = ((1 << len) - 1) << start;
                NumericSet { enabled_values }
            }

            fn new_leq<D: NumSet>(universe: &[u32],
                                  max: D, max_universe: &D::Universe) -> Self {
                let len = universe.binary_search(&max.max_value(max_universe))
                    .map(|x| x+1).unwrap_or_else(|x| x);
                let enabled_values = (1 << len) - 1;
                NumericSet { enabled_values }
            }

            fn new_eq<D: NumSet>(universe: &[u32],
                                 eq: D, eq_universe: &D::Universe) -> Self {
                eq.into_num_set(eq_universe, universe)
            }
        }
    }
}
