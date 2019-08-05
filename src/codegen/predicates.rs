use std::ops;

use fxhash::FxHashMap;

use crate::ir;

use super::iteration::IterationVarId;
use super::Size;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct PredicateId(usize);

pub struct Predicate {
    iteration_var: IterationVarId,
    instantiation_dims: Vec<(ir::DimId, u32)>,
    bound: Size,
}

impl Predicate {
    // This is sorted in decreasing stride order for compatibility with
    // `.multi_cartesian_product()`
    pub fn instantiation_dims(&self) -> std::slice::Iter<'_, (ir::DimId, u32)> {
        self.instantiation_dims.iter()
    }

    pub fn iteration_var(&self) -> IterationVarId {
        self.iteration_var
    }

    pub fn bound(&self) -> &Size {
        &self.bound
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PredicateKey {
    pub iteration_var: IterationVarId,
    pub instantiation_dims: Vec<(ir::DimId, u32)>,
    pub bound: Size,
}

#[derive(Default)]
pub struct Predicates {
    predicates: Vec<Predicate>,
    cache: FxHashMap<PredicateKey, PredicateId>,
}

impl Predicates {
    pub fn add(&mut self, mut key: PredicateKey) -> PredicateId {
        use std::collections::hash_map::Entry;

        // Sort in descending stride order.  See `Predicate::instantiation_dims`.
        key.instantiation_dims
            .sort_unstable_by(|(_, lhs), (_, rhs)| rhs.cmp(lhs));

        match self.cache.entry(key) {
            Entry::Occupied(occupied) => *occupied.get(),
            Entry::Vacant(vacant) => {
                let id = PredicateId(self.predicates.len());
                let key = vacant.key();

                self.predicates.push(Predicate {
                    iteration_var: key.iteration_var,
                    instantiation_dims: key.instantiation_dims.clone(),
                    bound: key.bound.clone(),
                });

                *vacant.insert(id)
            }
        }
    }
}

impl ops::Index<PredicateId> for Predicates {
    type Output = Predicate;

    fn index(&self, idx: PredicateId) -> &Predicate {
        &self.predicates[idx.0]
    }
}
