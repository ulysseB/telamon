//! Define a domain to represent bounded sets of disjoint integers.
use proc_macro2::TokenStream;

// FIXME: replace num_set with `mask: u16, all: u16`.

// FIXME: functions that do not need the universe:
// - insert
// - list
// - is_failed
// - is_constrained
// - contains
// - restrict
// FIXME: functions to delete
// - new_fixed
// - restrict_to
// - as_num_set
// FIXME: functions that needs the universe (and that's ok)
// - gcd/lcm
// - min/max
// - as_constrained
// - complement
// FIXME: all NumSet methods need both universe
// > request domain in min, max
// > request domain in comparison
// > request domain in final_value, rename it as_constrained
// > fix templates that use each functions
// FIXME: all num domain methods require both universes
// > request domain in constructors
//   - in trait def
//   - in trait impl
//   - in value_set printing
//   - in other templates
// Problem: it is not known for code: use () instead
// FIXME: do range and half range need a universe ?

/// Defines the `NumericSet` type.
pub fn get() -> TokenStream {
    quote! {
        #[derive(Copy, Clone)]
        #[repr(C)]
        pub struct NumericSet {
            len: usize,
            values: [u32; NumericSet::MAX_LEN],
        }

        #[allow(dead_code)]
        impl NumericSet {
            pub const MAX_LEN: usize = 16;

            pub const FAILED: Self = NumericSet { len: 0, values: [0; NumericSet::MAX_LEN] };

            /// Creates a new `NumericSet` containing a single value.
            pub fn new_fixed(val: u32) -> Self {
                let mut values = [0; Self::MAX_LEN];
                values[0] = val;
                NumericSet { values, len: 1 }
            }

            /// Returns the set containing all the possibilities. Assumes the universe is
            /// sorted.
            pub fn all(univers: &[u32]) -> Self {
                assert!(univers.len() <= NumericSet::MAX_LEN);
                let mut values = [0; NumericSet::MAX_LEN];
                for (v, dst) in univers.iter().cloned().zip(&mut values) { *dst = v; }
                NumericSet { len: univers.len(), values }
            }

            /// Inserts alternatives into the domain. Both domains should be from the same
            /// universe.
            pub fn insert(&mut self, other: NumericSet) {
                let mut values = [0; NumericSet::MAX_LEN];
                let (mut idx, mut idx_self, mut idx_other) = (0, 0, 0);
                while idx_self < self.len && idx_other < other.len {
                    let mut v = self.values[idx_self];
                    if v <= other.values[idx_other] { idx_self += 1; }
                    if v >= other.values[idx_other] {
                        v = other.values[idx_other];
                        idx_other += 1;
                    }
                    values[idx] = v;
                    idx += 1;
                }
                for &item in &self.values[idx_self..self.len] {
                    values[idx] = item;
                    idx += 1;
                }
                for &item in &other.values[idx_other..other.len] {
                    values[idx] = item;
                    idx += 1;
                }
                self.values = values;
                self.len = idx;
            }

            /// Returns the greatest common divisor of possible values.
            pub fn gcd(&self) -> u32 {
                self.values[..self.len].iter().cloned()
                    .fold1(num::integer::gcd).unwrap_or(1)
            }

            /// Returns the least common multiple of possible values.
            pub fn lcm(&self) -> u32 {
                self.values[..self.len].iter().cloned()
                    .fold1(num::integer::lcm).unwrap_or(1)
            }

            /// Returns the only value in the set, if it is fully constrained.
            pub fn as_constrained(&self) -> Option<u32> {
                if self.len == 1 { Some(self.values[0]) } else { None }
            }

            /// Lists the possible values the domain can take.
            pub fn list<'a>(&'a self) -> Vec<Self> {
                self.values[0..self.len].iter().cloned().map(Self::new_fixed).collect()
            }

            fn restrict_to(&mut self, other: &[u32]) {
                let (mut new_lhs, mut old_lhs, mut rhs) = (0, 0, 0);
                while old_lhs < self.len && rhs < other.len() {
                    if self.values[old_lhs] > other[rhs] {
                        rhs += 1;
                    } else if self.values[old_lhs] < other[rhs] {
                        old_lhs += 1;
                    } else {
                        self.values[new_lhs] = self.values[old_lhs];
                        old_lhs += 1;
                        new_lhs += 1;
                        rhs += 1;
                    }
                }
                self.len = new_lhs;
            }

            fn complement(&self, universe: &[u32]) -> Self {
                let mut values = [0; Self::MAX_LEN];
                let mut self_idx = 0;
                let mut new_idx = 0;
                for &item in universe {
                    if self_idx >= self.len || item < self.values[self_idx] {
                        values[new_idx] = item;
                        new_idx += 1;
                    } else if item > self.values[self_idx] {
                        panic!("self should be contained in universe")
                    } else {
                        self_idx += 1;
                    }
                }
                NumericSet { values, len: new_idx }
            }
        }

        impl Domain for NumericSet {
            fn is_failed(&self) -> bool { self.len == 0 }

            fn is_constrained(&self) -> bool { self.len == 1 }

            fn contains(&self, other: NumericSet) -> bool {
                let (mut lhs, mut rhs) = (0, 0);
                while lhs < self.len && rhs < other.len {
                    if self.values[lhs] > other.values[rhs] { return false; }
                    if self.values[lhs] == other.values[rhs] { rhs += 1; }
                    lhs += 1;
                }
                rhs == other.len
            }

            fn restrict(&mut self, other: NumericSet) {
                self.restrict_to(&other.values[..other.len])
            }
        }

        impl PartialEq for NumericSet {
            fn eq(&self, other: &NumericSet) -> bool {
                if self.len != other.len { return false; }
                for i in 0..self.len {
                    if self.values[i] != other.values[i] { return false; }
                }
                true
            }
        }

        impl std::fmt::Debug for NumericSet {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "{:?}", &self.values[..self.len])
            }
        }

        impl Eq for NumericSet { }

        impl NumSet for NumericSet {
            type Universe = [u32];

            fn min(&self) -> u32 {
                if self.len == 0 { 1 } else { self.values[0] }
            }

            fn max(&self) -> u32 {
                if self.len == 0 { 0 } else { self.values[self.len-1] }
            }

            fn as_num_set(&self) -> Option<NumericSet> { Some(*self) }

            fn neq<D: NumSet>(&self, _: &[u32], other: D, _: &D::Universe) -> bool {
                if let Some(other) = other.as_num_set() {
                    let (mut self_idx, mut other_idx) = (0, 0);
                    while self_idx < self.len && other_idx < other.len {
                        if self.values[self_idx] < other.values[other_idx] {
                            self_idx += 1;
                        } else if self.values[self_idx] > other.values[other_idx] {
                            other_idx += 1
                        } else {
                            return false;
                        }
                    }
                    true
                } else {
                    self.min() > other.max() || self.max() < other.min()
                }
            }
        }

        impl NumDomain for NumericSet {
            fn new_gt<D: NumSet>(universe: &[u32], min: D, _: &D::Universe) -> Self {
                let mut values = [0; NumericSet::MAX_LEN];
                let min = std::cmp::min(std::u32::MAX, min.min());
                let start = universe.binary_search(&min).map(|x| x+1).unwrap_or_else(|x| x);
                let len = universe.len() - start;
                for i in 0..len { values[i] = universe[start+i]; }
                NumericSet { values, len }
            }

            fn new_lt<D: NumSet>(universe: &[u32], max: D, _: &D::Universe) -> Self {
                let mut values = [0; NumericSet::MAX_LEN];
                let max = std::cmp::min(std::u32::MAX, max.max());
                let len = universe.binary_search(&max).unwrap_or_else(|x| x);
                for i in 0..len { values[i] = universe[i]; }
                NumericSet { values, len }
            }

            fn new_geq<D: NumSet>(universe: &[u32], min: D, _: &D::Universe) -> Self {
                let mut values = [0; NumericSet::MAX_LEN];
                let min = std::cmp::min(std::u32::MAX, min.min());
                let start = universe.binary_search(&min).unwrap_or_else(|x| x);
                let len = universe.len() - start;
                for i in 0..len { values[i] = universe[start+i]; }
                NumericSet { values, len }
            }

            fn new_leq<D: NumSet>(universe: &[u32], max: D, _: &D::Universe) -> Self {
                let mut values = [0; NumericSet::MAX_LEN];
                let max = std::cmp::min(std::u32::MAX, max.max());
                let len = universe.binary_search(&max).map(|x| x+1).unwrap_or_else(|x| x);
                for i in 0..len { values[i] = universe[i]; }
                NumericSet { values, len }
            }

            fn new_eq<D: NumSet>(universe: &[u32], eq: D, _: &D::Universe) -> Self {
                if let Some(mut eq) = eq.as_num_set() {
                    eq.restrict_to(universe);
                    eq
                } else {
                    let mut values = [0; NumericSet::MAX_LEN];
                    let min = std::cmp::min(std::u32::MAX, eq.min());
                    let max = std::cmp::min(std::u32::MAX, eq.max());
                    let start = universe.binary_search(&min).unwrap_or_else(|x| x);
                    let len = universe.binary_search(&max).unwrap_or_else(|x| x) - start;
                    for i in 0..len { values[i] = universe[start+i]; }
                    NumericSet { values, len }
                }
            }
        }
    }
}
