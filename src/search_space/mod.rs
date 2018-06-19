//! Search space datastructures and constraint propagation.
use ir;

mod dim_map;
mod operand;
generated_file!(choices);

pub use self::choices::{Action, Bool, DimKind, Domain, DomainStore, InstFlag, Order,
                        MemSpace, ThreadMapping, NumDomain};

use self::choices::{apply_action, DomainDiff, init_domain};
use std::sync::Arc;

/// A partially specified implementation.
#[derive(Clone)]
pub struct SearchSpace<'a> {
    ir_instance: Arc<ir::Function<'a>>,
    domain: DomainStore,
}

impl<'a> SearchSpace<'a> {
    /// Creates a new `SearchSpace` for the given `ir_instance`.
    pub fn new(mut ir_instance: ir::Function<'a>,
               mut actions: Vec<Action>) -> Result<Self, ()> {
        let mut domain = DomainStore::new(&ir_instance);
        // Enforce invariants.
        for inst in ir_instance.insts() {
            actions.extend(operand::inst_invariants(&ir_instance, inst));
        }
        let mut unused_diff = DomainDiff::default();
        for action in actions { apply_action(action, &mut domain, &mut unused_diff)?; }
        let actions = init_domain(&mut domain, &mut ir_instance)?;
        let mut space = SearchSpace { ir_instance: Arc::new(ir_instance), domain };
        space.apply_decisions(actions)?;
        Ok(space)
    }

    /// Returns the underlying ir instance.
    pub fn ir_instance(&self) -> &ir::Function<'a> { &self.ir_instance }

    /// Returns the domain of choices.
    pub fn domain(&self) -> &DomainStore { &self.domain }

    /// Allows rewritting the domain.
    pub fn domain_mut(&mut self) -> &mut DomainStore { &mut self.domain }

    /// Applies a list of decisions to the domain and propagate constraints.
    pub fn apply_decisions(&mut self, actions: Vec<Action>) -> Result<(), ()> {
        choices::apply_decisions(actions, &mut self.ir_instance, &mut self.domain)
    }

    /// Triggers a layout lowering.
    pub fn lower_layout(&mut self, mem: ir::mem::InternalId, st_dims: Vec<ir::dim::Id>,
                        ld_dims: Vec<ir::dim::Id>) -> Result<(), ()> {
        let actions = {
            let ir_instance = Arc::make_mut(&mut self.ir_instance);
            dim_map::lower_layout(ir_instance, mem, st_dims, ld_dims, &self.domain)?
        };
        self.apply_decisions(actions)
    }
}

/// Update the domain after a lowering.
fn process_lowering(ir_instance: &mut ir::Function,
                    domain: &mut DomainStore,
                    new_objs: &ir::NewObjs,
                    diff: &mut DomainDiff) -> Result<Vec<Action>, ()> {
    let mut actions = Vec::new();
    debug!("adding objects {:?}", new_objs);
    domain.alloc(ir_instance, new_objs);
    actions.extend(choices::init_domain_partial(domain, ir_instance, new_objs, diff)?);
    // Enforce invariants and call manual triggers.
    for &inst in &new_objs.instructions {
        actions.extend(operand::inst_invariants(ir_instance, ir_instance.inst(inst)));
    }
    Ok(actions)
}

/// Trigger to call when two dimensions are merged.
fn merge_dims(lhs: ir::dim::Id, rhs: ir::dim::Id, ir_instance: &mut ir::Function)
    -> Result<(ir::NewObjs, Vec<Action>), ()>
{
    debug!("merge {:?} and {:?}", lhs, rhs);
    ir_instance.merge(lhs, rhs);
    Ok(Default::default())
}

/// Adds a iteration dimension to a basic block.
fn add_iteration_dim(ir_instance: &mut ir::Function,
                     inst: ir::InstId, dim: ir::dim::Id) -> ir::NewObjs {
    debug!("set {:?} as iteration dim of inst {:?}", dim, inst);
    let mut new_objs = ir::NewObjs::default();
    if ir_instance.set_iteration_dim(inst, dim) {
        new_objs.add_iteration_dim(inst, dim);
    }
    new_objs
}

/// Adds a dimension to the list of thread dimensions.
fn add_thread_dim(ir_instance: &mut ir::Function, dim: ir::dim::Id) -> ir::NewObjs {
    debug!("set {:?} as a thread dimension", dim);
    let mut new_objs = ir::NewObjs::default();
    if ir_instance.add_thread_dim(dim) { new_objs.add_thread_dim(dim); }
    new_objs
}
