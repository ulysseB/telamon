//! Search space datastructures and constraint propagation.
use ir;

mod dim_map;
mod operand;
generated_file!(choices);

pub use self::choices::{
    Action, Bool, Choice, DimKind, Domain, DomainStore, InstFlag, MemorySpace, MemSpace, NumSet,
    Order, ThreadMapping,
};

use self::choices::{apply_action, init_domain, DomainDiff};
use std::sync::Arc;

/// A partially specified implementation.
#[derive(Clone)]
pub struct SearchSpace<'a> {
    ir_instance: Arc<ir::Function<'a>>,
    domain: DomainStore,
}

impl<'a> SearchSpace<'a> {
    /// Creates a new `SearchSpace` for the given `ir_instance`.
    pub fn new(
        ir_instance: ir::Function<'a, ()>,
        mut actions: Vec<Action>,
    ) -> Result<Self, ()> {
        // Pre-allocate IDs for future lowerings.
        let mut ir_instance = ir_instance.freeze();

        let mut domain = DomainStore::new(&ir_instance);
        // Enforce invariants.
        for inst in ir_instance.insts() {
            actions.extend(operand::inst_invariants(&ir_instance, inst));
        }
        let mut unused_diff = DomainDiff::default();
        for action in actions {
            apply_action(action, &mut domain, &mut unused_diff)?;
        }
        let actions = init_domain(&mut domain, &mut ir_instance)?;
        let mut space = SearchSpace {
            ir_instance: Arc::new(ir_instance),
            domain,
        };
        space.apply_decisions(actions)?;
        Ok(space)
    }

    /// Returns the underlying ir instance.
    pub fn ir_instance(&self) -> &ir::Function<'a> {
        &self.ir_instance
    }

    /// Returns the domain of choices.
    pub fn domain(&self) -> &DomainStore {
        &self.domain
    }

    /// Allows rewritting the domain.
    pub fn domain_mut(&mut self) -> &mut DomainStore {
        &mut self.domain
    }

    /// Applies a list of decisions to the domain and propagate constraints.
    pub fn apply_decisions(&mut self, actions: Vec<Action>) -> Result<(), ()> {
        choices::apply_decisions(actions, &mut self.ir_instance, &mut self.domain)
    }

    /// Triggers a layout lowering.
    pub fn lower_layout(
        &mut self,
        mem: ir::MemId,
        st_dims: Vec<ir::DimId>,
        ld_dims: Vec<ir::DimId>,
    ) -> Result<(), ()> {
        let mut diff = DomainDiff::default();
        let actions = {
            let ir_instance = Arc::make_mut(&mut self.ir_instance);
            let (new_objs, mut actions) =
                dim_map::lower_layout(ir_instance, mem, st_dims, ld_dims)?;
            actions.extend(process_lowering(
                ir_instance,
                &mut self.domain,
                &new_objs,
                &mut diff,
            )?);
            actions
        };
        // Manually apply actions since telamon-gen does not expose an `apply_actions`
        // function that takes a `diff` as argument. This code will be removed when we
        // support dynamic layout with variables anyway.
        for action in actions {
            apply_action(action, &mut self.domain, &mut diff)?;
        }
        while !diff.is_empty() {
            choices::propagate_changes(
                &mut diff,
                &mut self.ir_instance,
                &mut self.domain,
            )?;
        }
        Ok(())
    }
}

/// Update the domain after a lowering.
fn process_lowering(
    ir_instance: &mut ir::Function,
    domain: &mut DomainStore,
    new_objs: &ir::NewObjs,
    diff: &mut DomainDiff,
) -> Result<Vec<Action>, ()> {
    let mut actions = Vec::new();
    debug!("adding objects {:?}", new_objs);
    domain.alloc(ir_instance, new_objs);
    actions.extend(choices::init_domain_partial(
        domain,
        ir_instance,
        new_objs,
        diff,
    )?);
    // Enforce invariants and call manual triggers.
    for &inst in &new_objs.instructions {
        actions.extend(operand::inst_invariants(
            ir_instance,
            ir_instance.inst(inst),
        ));
    }
    Ok(actions)
}

/// Trigger to call when two dimensions are merged.
fn merge_dims(
    lhs: ir::DimId,
    rhs: ir::DimId,
    ir_instance: &mut ir::Function,
) -> Result<(ir::NewObjs, Vec<Action>), ()> {
    debug!("merge {:?} and {:?}", lhs, rhs);
    ir_instance.merge(lhs, rhs);
    Ok(Default::default())
}

/// Adds a iteration dimension to a basic block.
fn add_iteration_dim(
    ir_instance: &mut ir::Function,
    inst: ir::InstId,
    dim: ir::DimId,
) -> ir::NewObjs {
    debug!("set {:?} as iteration dim of inst {:?}", dim, inst);
    let mut new_objs = ir::NewObjs::default();
    if ir_instance.set_iteration_dim(inst, dim) {
        new_objs.add_iteration_dim(inst, dim);
    }
    new_objs
}

/// Adds a dimension to the list of thread dimensions.
fn add_thread_dim(ir_instance: &mut ir::Function, dim: ir::DimId) -> ir::NewObjs {
    debug!("set {:?} as a thread dimension", dim);
    let mut new_objs = ir::NewObjs::default();
    if ir_instance.add_thread_dim(dim) {
        new_objs.add_thread_dim(dim);
    }
    new_objs
}

/// Returns the memory space accessed by an access pattern.
pub fn access_pattern_space(
    pattern: &ir::AccessPattern,
    space: &SearchSpace,
) -> MemSpace {
    // We either have a `MemId` or the array is an external array in global memory.
    pattern
        .mem_block()
        .map(|id| space.domain().get_mem_space(id))
        .unwrap_or(MemSpace::GLOBAL)
}
