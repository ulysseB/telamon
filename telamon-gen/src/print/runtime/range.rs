//! Defines `Range` and `HalfRange` types.
use proc_macro2::TokenStream;

/// Returns the tokens defining `Range` and `HalfRange`.
pub fn get() -> TokenStream {
    quote! {
        /// Abstracts integer choices by a range.
        #[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
        #[repr(C)]
        pub struct Range {
            pub min: u32,
            pub max: u32,
        }

        #[allow(dead_code)]
        impl Range {
            pub const ALL: Range = Range { min: 0, max: std::u32::MAX };

            pub const FAILED: Range = Range { min: 1, max: 0 };

            /// Returns the full range.
            pub fn all() -> Self { Self::ALL }

            /// Inserts alternatives into the domain.
            pub fn insert(&mut self, other: Range) {
                self.min = std::cmp::min(self.min, other.min);
                self.max = std::cmp::max(self.max, other.max);
            }

            /// Returns the difference between the min and max of two ranges.
            fn get_diff_add(&self, other: Range) -> Range {
                Range { min: other.min - self.min, max: self.max - other.max }
            }

            /// Restricts the `Range` by applying the result of `get_diff`.
            fn apply_diff_add(&mut self, diff: Range) {
                self.min += diff.min;
                self.max -= diff.max;
            }

            /// Returns the difference between the min and max of two ranges.
            fn get_diff_mul(&self, other: Range) -> Range {
                Range { min: other.min / self.min, max: self.max / other.max }
            }

            /// Restricts the `Range` by applying the result of `get_diff`.
            fn apply_diff_mul(&mut self, diff: Range) {
                self.min *= diff.min;
                self.max /= diff.max;
            }

            fn add_add(&mut self, diff: Range) {
                self.min += diff.min;
                self.max += diff.max;
            }

            fn add_mul(&mut self, diff: Range) {
                self.min *= diff.min;
                self.max *= diff.max;
            }

            fn sub_add(&mut self, diff: Range) {
                self.min -= diff.min;
                self.max -= diff.max;
            }

            fn sub_mul(&mut self, diff: Range) {
                self.min /= diff.min;
                self.max /= diff.max;
            }

            pub fn as_fixed(&self) -> Option<u32> {
                if self.min == self.max { Some(self.min) } else { None }
            }
        }

        impl Domain for Range {
            fn is_failed(&self) -> bool { self.min > self.max }

            fn is_constrained(&self) -> bool { self.min == self.max }

            fn contains(&self, other: Range) -> bool {
                self.min <= other.min && self.max >= other.max
            }

            fn restrict(&mut self, other: Range) {
                self.min = std::cmp::max(self.min, other.min);
                self.max = std::cmp::min(self.max, other.max);
            }
        }

        impl NumSet for Range {
            type Universe = ();

            fn min(&self, _: &()) -> u32 { self.min }

            fn max(&self, _: &()) -> u32 { self.max }
        }

        impl NumDomain for Range {
            fn new_gt<D: NumSet>(_: &(), min: D, min_universe: &D::Universe) -> Self {
                let min = min.min(min_universe).saturating_add(1);
                Range { min, .. Range::ALL }
            }

            fn new_lt<D: NumSet>(_: &(), max: D, max_universe: &D::Universe) -> Self {
                let max = max.max(max_universe).saturating_sub(1);
                Range { max, .. Range::ALL }
            }

            fn new_geq<D: NumSet>(_: &(), min: D, min_universe: &D::Universe) -> Self {
                Range { min: min.min(min_universe), .. Range::ALL }
            }

            fn new_leq<D: NumSet>(_: &(), max: D, max_universe: &D::Universe) -> Self {
                Range { max: max.max(max_universe), .. Range::ALL }
            }

            fn new_eq<D: NumSet>(_: &(), eq: D, eq_universe: &D::Universe) -> Self {
                Range {
                    max: eq.max(eq_universe),
                    min: eq.min(eq_universe),
                }
            }
        }

        /// Abstracts integer choices by a range, but only store `min`.
        #[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
        #[repr(C)]
        pub struct HalfRange { pub min: u32 }

        #[allow(dead_code)]
        impl HalfRange {
            pub const ALL: HalfRange = HalfRange { min: 0 };

            /// Returns the full `HalfRange`.
            pub fn all() -> Self { Self::ALL }

            /// Inserts alternatives into the domain.
            pub fn insert(&mut self, other: HalfRange) {
                self.min = std::cmp::min(self.min, other.min);
            }

            /// Returns the difference between the min and max of two ranges.
            fn get_diff_add(&self, other: HalfRange) -> HalfRange {
                HalfRange { min: other.min - self.min }
            }

            /// Restricts the `Range` by applying the result of `get_diff`.
            fn apply_diff_add(&mut self, diff: HalfRange) { self.min += diff.min; }

            /// Returns the difference between the min and max of two ranges.
            fn get_diff_mul(&self, other: HalfRange) -> HalfRange {
                HalfRange { min: other.min / self.min }
            }

            /// Restricts the `Range` by applying the result of `get_diff`.
            fn apply_diff_mul(&mut self, diff: HalfRange) { self.min *= diff.min; }

            fn add_add(&mut self, diff: HalfRange) {
                self.min += diff.min;
            }

            fn add_mul(&mut self, diff: HalfRange) {
                self.min *= diff.min;
            }

            fn sub_add(&mut self, diff: HalfRange) {
                self.min -= diff.min;
            }

            fn sub_mul(&mut self, diff: HalfRange) {
                self.min /= diff.min;
            }
        }

        impl Domain for HalfRange {
            fn is_failed(&self) -> bool { false }

            fn is_constrained(&self) -> bool { false }

            fn contains(&self, other: HalfRange) -> bool {
                self.min <= other.min
            }

            fn restrict(&mut self, other: HalfRange) {
                self.min = std::cmp::max(self.min, other.min);
            }
        }

        impl NumSet for HalfRange {
            type Universe = ();

            fn min(&self, _: &()) -> u32 { self.min }

            fn max(&self, _: &()) -> u32 { std::u32::MAX }
        }

        impl NumDomain for HalfRange {
            fn new_gt<D: NumSet>(_: &(), min: D, min_universe: &D::Universe) -> Self {
                let min = min.min(min_universe).saturating_add(1);
                HalfRange { min }
            }

            fn new_lt<D: NumSet>(_: &(), _: D, _: &D::Universe) -> Self {
                HalfRange::ALL
            }

            fn new_geq<D: NumSet>(_: &(), min: D, min_universe: &D::Universe) -> Self {
                HalfRange { min: min.min(min_universe) }
            }

            fn new_leq<D: NumSet>(_: &(), _: D, _: &D::Universe) -> Self {
                HalfRange::ALL
            }

            fn new_eq<D: NumSet>(_: &(), eq: D, eq_universe: &D::Universe) -> Self {
                HalfRange { min: eq.min(eq_universe) }
            }
        }
    }
}
