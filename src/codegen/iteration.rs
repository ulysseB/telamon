use std::ops;

use fxhash::FxHashMap;

use crate::ir;

use super::Size;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct IterationVarId(usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IterationVarKey {
    pub global_dims: Vec<(ir::DimId, Size)>,
    pub loop_dims: Vec<(ir::DimId, Size)>,
}

pub struct IterationVar {
    outer_dims: Vec<(ir::DimId, Size)>,
}

impl IterationVar {
    pub fn outer_dims(&self) -> impl Iterator<Item = (ir::DimId, &Size)> + '_ {
        self.outer_dims.iter().map(|&(id, ref size)| (id, size))
    }
}

#[derive(Default)]
pub struct IterationVars {
    iteration_vars: Vec<IterationVar>,
    cache: FxHashMap<IterationVarKey, IterationVarId>,

    // Variable that the loop updates.  Also needs to reset at the end.
    loop_updates: FxHashMap<ir::DimId, Vec<(IterationVarId, Size)>>,
    // Variable that the loop defines.
    loop_defs: FxHashMap<ir::DimId, Vec<IterationVarId>>,
    // Variable defined at the start of the code.
    global_defs: Vec<IterationVarId>,
}

impl IterationVars {
    pub fn global_defs(&self) -> impl Iterator<Item = IterationVarId> + '_ {
        self.global_defs.iter().cloned()
    }

    pub fn loop_defs(&self, dim: ir::DimId) -> impl Iterator<Item = IterationVarId> + '_ {
        self.loop_defs.get(&dim).cloned().into_iter().flatten()
    }

    pub fn loop_updates(
        &self,
        dim: ir::DimId,
    ) -> impl Iterator<Item = (IterationVarId, &Size)> {
        self.loop_updates
            .get(&dim)
            .into_iter()
            .flat_map(|updates| updates.iter().map(|&(id, ref size)| (id, size)))
    }

    pub fn iter(&self) -> impl Iterator<Item = (IterationVarId, &IterationVar)> + '_ {
        self.iteration_vars
            .iter()
            .enumerate()
            .map(|(id, var)| (IterationVarId(id), var))
    }

    pub fn add(
        &mut self,
        mut global_dims: Vec<(ir::DimId, Size)>,
        loop_dims: Vec<(ir::DimId, Size)>,
    ) -> IterationVarId {
        use std::collections::hash_map::Entry;

        global_dims.sort_unstable_by_key(|&(dim, _)| dim);
        match self.cache.entry(IterationVarKey {
            global_dims,
            loop_dims,
        }) {
            Entry::Occupied(occupied) => *occupied.get(),
            Entry::Vacant(vacant) => {
                let id = IterationVarId(self.iteration_vars.len());
                let key = vacant.key();

                for &(loop_dim, ref stride) in &key.loop_dims {
                    self.loop_updates
                        .entry(loop_dim)
                        .or_insert(Vec::new())
                        .push((id, stride.clone()));
                }

                if let Some(&(outer_dim, _)) = key.loop_dims.first() {
                    self.loop_defs
                        .entry(outer_dim)
                        .or_insert(Vec::new())
                        .push(id);
                } else {
                    self.global_defs.push(id);
                }

                self.iteration_vars.push(IterationVar {
                    outer_dims: key.global_dims.clone(),
                });

                *vacant.insert(id)
            }
        }
    }
}

impl ops::Index<IterationVarId> for IterationVars {
    type Output = IterationVar;

    fn index(&self, idx: IterationVarId) -> &IterationVar {
        &self.iteration_vars[idx.0]
    }
}
