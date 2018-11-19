//! Defines traits for domains of integers.
use proc_macro2::TokenStream;

/// Returns the definition of `NumDomain` and `NumSet` traits.
pub fn get() -> TokenStream {
    quote! {
        /// A domain containing integers.
        pub trait NumSet {
            type Universe: ?Sized;

            /// Returns the maximum value in the domain.
            fn min(&self, universe: &Self::Universe) -> u32;
            /// Returns the minimum value in the domain.
            fn max(&self, universe: &Self::Universe) -> u32;

            /// Converts the domain into a numeric set with the given domain. Values that
            /// are not in `new_universe` are skipped.
            fn into_num_set(
                &self,
                self_universe: &Self::Universe,
                new_universe: &[u32]
            ) -> NumericSet {
                let start = new_universe.binary_search(&self.min(self_universe))
                    .unwrap_or_else(|x| x);
                let len = new_universe.binary_search(&self.max(self_universe))
                    .map(|x| x + 1)
                    .unwrap_or_else(|x| x) - start;
                let enabled_values = (((1u32 << len) - 1) << start) as u16;
                NumericSet { enabled_values }
            }

            /// Returns the value of the domain, if it is constrained.
            fn as_constrained(&self, universe: &Self::Universe) -> Option<u32> {
                let value = self.min(universe);
                if value == self.max(universe) { Some(value) } else { None }
            }

            fn lt<D: NumSet>(&self, universe: &Self::Universe,
                             other: D, other_universe: &D::Universe) -> bool {
                self.max(universe) < other.min(other_universe)
            }

            fn gt<D: NumSet>(&self, universe: &Self::Universe,
                             other: D, other_universe: &D::Universe) -> bool {
                self.min(universe) > other.max(other_universe)
            }

            fn leq<D: NumSet>(&self, universe: &Self::Universe,
                              other: D, other_universe: &D::Universe) -> bool {
                self.max(universe) <= other.min(other_universe)
            }

            fn geq<D: NumSet>(&self, universe: &Self::Universe,
                              other: D, other_universe: &D::Universe) -> bool {
                self.min(universe) >= other.max(other_universe)
            }

            fn eq<D: NumSet>(&self, universe: &Self::Universe,
                             other: D, other_universe: &D::Universe) -> bool {
                self.min(universe) == other.max(other_universe) &&
                    self.max(universe) == other.min(other_universe)
            }

            fn neq<D: NumSet>(&self, universe: &Self::Universe,
                              other: D, other_universe: &D::Universe) -> bool {
                self.min(universe) > other.max(other_universe) ||
                    self.max(universe) < other.min(other_universe)
            }
        }

        /// A choice that contains integers.
        pub trait NumDomain: NumSet {
            /// Returns the domain containing the values of the universe greater than min.
            fn new_gt<D: NumSet>(universe: &Self::Universe,
                                 min: D, min_universe: &D::Universe) -> Self;
            /// Returns the domain containing the values of the universe smaller than max.
            fn new_lt<D: NumSet>(universe: &Self::Universe,
                                 max: D, max_universe: &D::Universe) -> Self;
            /// Retruns the domain containing the values of the universe greater or equal
            /// to min.
            fn new_geq<D: NumSet>(universe: &Self::Universe,
                                  min: D, min_universe: &D::Universe) -> Self;
            /// Returns the domain containing the values of the universe smaller or equal
            /// to min.
            fn new_leq<D: NumSet>(universe: &Self::Universe,
                                  max: D, max_universe: &D::Universe) -> Self;
            /// Returns the domain containing the values of `eq` that are also in the
            /// universe.
            fn new_eq<D: NumSet>(universe: &Self::Universe,
                                 eq: D, eq_universe: &D::Universe) -> Self;
        }

        impl NumSet for u32 {
            type Universe = ();

            fn min(&self, _: &()) -> u32 { *self }

            fn max(&self, _: &()) -> u32 { *self }
        }

        impl<'a> NumSet for &'a [u32] {
            type Universe = ();

            fn min(&self, _: &()) -> u32 {
                if self.is_empty() { 1 } else { self[0] }
            }

            fn max(&self, _: &()) -> u32 {
                if self.is_empty() { 0 } else { self[self.len()-1] }
            }

            fn into_num_set(&self, _: &(), universe: &[u32]) -> NumericSet {
                let mut idx = 0;
                let mut enabled_values = 0;
                'values: for &value in *self {
                    while {
                        if idx >= universe.len() { break 'values; }
                        universe[idx] < value
                    } { idx = idx+1; }
                    if universe[idx] == value { enabled_values |= 1 << idx; }
                }
                NumericSet { enabled_values }
            }
        }
    }
}
