use super::ir;
#[allow(unused_imports)]
use super::ir::prelude::*;
#[allow(unused_imports)]
use std;
use std::sync::Arc;
#[allow(unused_imports)]
use utils::*;

{{>store}}

{{#each choices}}
    {{>choice_def this}}
{{/each}}

{{enums}}

{{>actions this}}

/// Propagate the changes stored in `diff`.
pub fn propagate_changes(diff: &mut DomainDiff, ir_instance: &mut Arc<ir::Function>,
                     store: &mut DomainStore) -> Result<(), ()> {
    {{~#each choices}}
        while let Some((({{#each arguments}}{{this.[0]}}, {{/each}}), old, new)) =
                diff.pop_{{name}}_diff() {
            debug!("propagating {{name}}{:?} {:?} -> {:?}",
                   ({{>choice.arg_names this}}), old, new);
            {{name}}::on_change(old, new,
                {{~>choice.arg_names this}}ir_instance, store, diff)?;
        }
    {{~/each}}
    Ok(())
}

/// Applies a set of decisions to the domain and propagate the changes.
pub fn apply_decisions(actions: Vec<Action>, ir_instance: &mut Arc<ir::Function>,
                   domain: &mut DomainStore) -> Result<(), ()> {
    let mut diff = DomainDiff::default();
    for action in actions { apply_action(action, domain, &mut diff)?; }
    while !diff.is_empty() { propagate_changes(&mut diff, ir_instance, domain)?; }
    Ok(())
}

/// Update the domain after a lowering.
#[cfg(feature="gen_applicators")]
fn process_lowering(ir_instance: &mut ir::Function,
                    domain: &mut DomainStore,
                    new_objs: &ir::NewObjs,
                    diff: &mut DomainDiff) -> Result<Vec<Action>, ()> {
    let mut actions = Vec::new();
    domain.alloc(ir_instance, &new_objs);
    actions.extend(init_domain_partial(domain, ir_instance, &new_objs, diff)?);
    Ok(actions)
}
#[cfg(not(feature="gen_applicators"))]
use super::process_lowering;



/// Initializes the `DomainStore` with available choices for each decision.
#[allow(unused_variables, unused_mut)]
pub fn init_domain(store: &mut DomainStore,
                   ir_instance: &mut ir::Function) -> Result<Vec<Action>, ()> {
    trace!("called init_domain from file {}", file!());
    // Run all the filters once.
    let ref mut diff = DomainDiff::default(); // Pass an empty diff to propagate and triggers.
    let mut unused_diff = DomainDiff::default();
    {{#each choices~}}
        {{#>loop_nest iteration_space~}}
            {{>run_filters this}}
        {{/loop_nest~}}
    {{/each~}}
    {{store.filter_all}}
    // Propagate the filters where necessary.
    let mut actions: Vec<Action> = Vec::new();
    {{#each triggers~}}
        let mut trigger_{{id}} = Vec::new();
        {{#>loop_nest loop_nest}}
            if check_trigger_{{id}}({{>choice.arg_names}}ir_instance, store, diff) {
                trigger_{{id}}.push(({{>choice.arg_ids}}));
            }
        {{/loop_nest}}
    {{/each}}
    {{#each choices~}}
        {{#>loop_nest iteration_space}}{{>propagate this}}{{/loop_nest~}}
    {{/each~}}
    // Propagate triggers.
    {{#each triggers}}{{>trigger_call call_id=id delayed=true}}{{/each~}}
    Ok(actions)
}

/// Initializes the part of the `DomainStore` allocated for the given objects with available
/// choices for each decision.
#[allow(unused_variables, unused_mut)]
pub fn init_domain_partial(store: &mut DomainStore,
                       ir_instance: &mut ir::Function,
                       new_objs: &ir::NewObjs,
                       diff: &mut DomainDiff) -> Result<Vec<Action>, ()> {
    let mut unused_diff = DomainDiff::default();
    // Disable new increments of existing counters.
    {{#each incr_iterators~}}
        {{#>iter_new_objects iter~}}
            {{>disable_increment}}
        {{/iter_new_objects~}}
    {{/each~}}
    // Call filters.
    {{#each partial_iterators~}}
        {{#>iter_new_objects this.[0]~}}
            {{>run_filters this.[1].choice arg_names=this.[1].arg_names}}
        {{/iter_new_objects~}}
    {{/each~}}
    // Propagate decisions that are not already propagted.
    let mut actions: Vec<Action> = Vec::new();
    {{#each partial_iterators~}}
        {{#>iter_new_objects this.[0]~}}
                {{>propagate this.[1].choice arg_names=this.[1].arg_names}}
        {{/iter_new_objects~}}
    {{/each~}}
    // Take new increment on existing counters into account.
    {{#each incr_iterators~}}
        {{#>iter_new_objects iter~}}
            {{>account_new_incrs}}
        {{/iter_new_objects~}}
    {{/each~}}
    // Check if we should call the new triggers.
    {{#each triggers~}}
        let mut trigger_{{id}} = Vec::new();
        {{#each partial_iterators}}
            {{#>iter_new_objects this.[0]}}
    if check_trigger_{{../id}}(
        {{~>choice.arg_names ../../this arg_names=this.[1]~}}ir_instance, store, diff)
    {
        trigger_{{../id}}.push(({{>choice.arg_ids ../../this arg_names=this.[1]}}));
    }
            {{/iter_new_objects~}}
        {{/each~}}
    {{/each~}}
    // Propagate triggers.
    {{#each triggers}}{{>trigger_call call_id=id delayed=true}}{{/each~}}
    Ok(actions)
}

{{#each triggers~}}
    {{>trigger_check}}
{{/each~}}

// TODO(cleanup): generate (IrInstance, Domain) pair here.

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

/// Abstracts integer choices by a range.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
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

/// Abstracts integer choices by a range, but only store `min`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
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

#[derive(Copy, Clone)]
pub struct NumericSet {
    len: usize,
    values: [u32; NumericSet::MAX_LEN],
}

#[allow(dead_code)]
impl NumericSet {
    const MAX_LEN: usize = 16;

    const FAILED: Self = NumericSet { len: 0, values: [0; NumericSet::MAX_LEN] };

    /// Returns the set containing all the possibilities. Assumes the universe is sorted.
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

    fn complement(&self, universe: &[u16]) -> Self {
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

/// A domain containing integers.
pub trait NumDomain {
    type Universe: ?Sized;

    /// Returns the maximum value in the domain.
    fn min(&self) -> u32;
    /// Returns the minimum value in the domain.
    fn max(&self) -> u32;
    /// Returns the domain as a `NumericSet`, if applicable.
    fn as_num_set(&self) -> Option<NumericSet> { None }

    /// Returns the domain containing the values of the universe greater than min.
    fn new_gt<D: NumDomain>(universe: &Self::Universe, min: D) -> Self;
    /// Returns the domain containing the values of the universe smaller than max.
    fn new_lt<D: NumDomain>(universe: &Self::Universe, max: D) -> Self;
    /// Retruns the domain containing the values of the universe greater or equal to min.
    fn new_geq<D: NumDomain>(universe: &Self::Universe, min: D) -> Self;
    /// Returns the domain containing the values of the universe smaller or equal to min.
    fn new_leq<D: NumDomain>(universe: &Self::Universe, min: D) -> Self;
    /// Returns the domain containing the values of `eq` that are also in the universe.
    fn new_eq<D: NumDomain>(universe: &Self::Universe, eq: D) -> Self;

    /// Returns the value of the domain, if it is constrained.
    fn final_value(&self) -> u32 {
        assert_eq!(self.min(), self.max());
        self.min()
    }


    fn lt<D: NumDomain>(&self, other: D) -> bool { self.max() < other.min() }

    fn gt<D: NumDomain>(&self, other: D) -> bool { self.min() > other.max() }

    fn leq<D: NumDomain>(&self, other: D) -> bool { self.max() <= other.min() }

    fn geq<D: NumDomain>(&self, other: D) -> bool { self.min() >= other.max() }

    fn eq<D: NumDomain>(&self, other: D) -> bool {
        self.min() == other.max() && self.max() == other.min()
    }

    fn neq<D: NumDomain>(&self, other: D) -> bool {
        self.min() > other.max() || self.max() < other.min()
    }
}

impl NumDomain for Range {
    type Universe = Range;

    fn min(&self) -> u32 { self.min }

    fn max(&self) -> u32 { self.max }

    fn new_gt<D: NumDomain>(universe: &Range, min: D) -> Self {
        let min = min.min().saturating_add(1);
        Range { min: std::cmp::max(min, universe.min), .. *universe }
    }

    fn new_lt<D: NumDomain>(universe: &Range, max: D) -> Self {
        let max = max.max().saturating_sub(1);
        Range { max: std::cmp::min(max, universe.max), .. *universe }
    }

    fn new_geq<D: NumDomain>(universe: &Range, min: D) -> Self {
        Range { min: std::cmp::max(min.min(), universe.min), .. *universe }
    }

    fn new_leq<D: NumDomain>(universe: &Range, max: D) -> Self {
        Range { max: std::cmp::min(max.max(), universe.max), .. *universe }
    }

    fn new_eq<D: NumDomain>(universe: &Range, eq: D) -> Self {
        Range {
            max: std::cmp::min(eq.max(), universe.max),
            min: std::cmp::max(eq.min(), universe.min),
        }
    }
}

impl NumDomain for HalfRange {
    type Universe = HalfRange;

    fn min(&self) -> u32 { self.min }

    fn max(&self) -> u32 { std::u32::MAX }

    fn new_gt<D: NumDomain>(universe: &HalfRange, min: D) -> Self {
        let min = min.min().saturating_add(1);
        HalfRange { min: std::cmp::max(min, universe.min) }
    }

    fn new_lt<D: NumDomain>(universe: &HalfRange, _: D) -> Self { *universe }

    fn new_geq<D: NumDomain>(universe: &HalfRange, min: D) -> Self {
        HalfRange { min: std::cmp::max(min.min(), universe.min) }
    }

    fn new_leq<D: NumDomain>(universe: &HalfRange, _: D) -> Self { *universe }

    fn new_eq<D: NumDomain>(universe: &HalfRange, eq: D) -> Self {
        HalfRange { min: std::cmp::max(eq.min(), universe.min) }
    }
}

impl NumDomain for NumericSet {
    type Universe = [u32];

    fn min(&self) -> u32 {
        if self.len == 0 { 1 } else { self.values[0] }
    }

    fn max(&self) -> u32 {
        if self.len == 0 { 0 } else { self.values[self.len-1] }
    }

    fn as_num_set(&self) -> Option<NumericSet> { Some(*self) }

    fn new_gt<D: NumDomain>(universe: &[u32], min: D) -> Self {
        let mut values = [0; NumericSet::MAX_LEN];
        let min = std::cmp::min(std::u32::MAX, min.min());
        let start = universe.binary_search(&min).map(|x| x+1).unwrap_or_else(|x| x);
        let len = universe.len() - start;
        for i in 0..len { values[i] = universe[start+i]; }
        NumericSet { values, len }
    }

    fn new_lt<D: NumDomain>(universe: &[u32], max: D) -> Self {
        let mut values = [0; NumericSet::MAX_LEN];
        let max = std::cmp::min(std::u32::MAX, max.max());
        let len = universe.binary_search(&max).unwrap_or_else(|x| x);
        for i in 0..len { values[i] = universe[i]; }
        NumericSet { values, len }
    }

    fn new_geq<D: NumDomain>(universe: &[u32], min: D) -> Self {
        let mut values = [0; NumericSet::MAX_LEN];
        let min = std::cmp::min(std::u32::MAX, min.min());
        let start = universe.binary_search(&min).unwrap_or_else(|x| x);
        let len = universe.len() - start;
        for i in 0..len { values[i] = universe[start+i]; }
        NumericSet { values, len }
    }

    fn new_leq<D: NumDomain>(universe: &[u32], max: D) -> Self {
        let mut values = [0; NumericSet::MAX_LEN];
        let max = std::cmp::min(std::u32::MAX, max.max());
        let len = universe.binary_search(&max).map(|x| x+1).unwrap_or_else(|x| x);
        for i in 0..len { values[i] = universe[i]; }
        NumericSet { values, len }
    }

    fn new_eq<D: NumDomain>(universe: &[u32], eq: D) -> Self {
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

    fn neq<D: NumDomain>(&self, other: D) -> bool {
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

impl NumDomain for u32 {
    type Universe = u32;

    fn min(&self) -> u32 { *self }

    fn max(&self) -> u32 { *self }

    fn new_gt<D: NumDomain>(universe: &u32, min: D) -> Self {
        std::cmp::max(*universe, min.min().saturating_add(1))
    }

    fn new_lt<D: NumDomain>(universe: &u32, max: D) -> Self {
        std::cmp::min(*universe, max.max().saturating_sub(1))
    }

    fn new_geq<D: NumDomain>(universe: &u32, min: D) -> Self {
        std::cmp::max(*universe, min.min())
    }

    fn new_leq<D: NumDomain>(universe: &u32, max: D) -> Self {
        std::cmp::min(*universe, max.max())
    }

    fn new_eq<D: NumDomain>(universe: &u32, _: D) -> Self { *universe }
}
