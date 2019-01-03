//! Define a domain to represent bounded sets of disjoint integers.
use proc_macro2::TokenStream;
use quote::quote;
// FIXME: move insert to the domain
// FIXME: move all to IntegerDomain

/// Defines the `NumericSet` type.
pub fn get() -> TokenStream {
    quote! {
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
        #[repr(C)]
        pub struct NumericSet {
            enabled_values: u16
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

            fn min(&self, universe: &[u32]) -> u32 {
                if self.is_failed() {
                    0
                } else {
                    universe[self.enabled_values.trailing_zeros() as usize]
                }
            }

            fn max(&self, universe: &[u32]) -> u32 {
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

            fn neq<D: NumSet>(&self, universe: &[u32],
                              other: D, other_universe: &D::Universe) -> bool {
                self.min(universe) > other.max(other_universe)
                    || self.max(universe) < other.min(other_universe)
                    || other.into_num_set(other_universe, universe).is_failed()
            }
        }

        impl NumDomain for NumericSet {
            fn new_gt<D: NumSet>(universe: &[u32],
                                 min: D, min_universe: &D::Universe) -> Self {
                let start = universe.binary_search(&min.min(min_universe))
                    .map(|x| x+1).unwrap_or_else(|x| x);
                let len = universe.len() - start;
                let enabled_values = ((1 << len) - 1) << start;
                NumericSet { enabled_values }
            }

            fn new_lt<D: NumSet>(universe: &[u32],
                                 max: D, max_universe: &D::Universe) -> Self {
                let len = universe.binary_search(&max.max(max_universe))
                    .unwrap_or_else(|x| x);
                let enabled_values = (1 << len) - 1;
                NumericSet { enabled_values }
            }

            fn new_geq<D: NumSet>(universe: &[u32],
                                  min: D, min_universe: &D::Universe) -> Self {
                let start = universe.binary_search(&min.min(min_universe))
                    .unwrap_or_else(|x| x);
                let len = universe.len() - start;
                let enabled_values = ((1 << len) - 1) << start;
                NumericSet { enabled_values }
            }

            fn new_leq<D: NumSet>(universe: &[u32],
                                  max: D, max_universe: &D::Universe) -> Self {
                let len = universe.binary_search(&max.max(max_universe))
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
