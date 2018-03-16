//! Search space datastructures and constraint propagation.
use ir;

mod dim_map;
mod operand;
generated_file!(choices);

// FIXME: unrolled loops of size 1 should not be allowed

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
        let ref mut diff = DomainDiff::default();
        for action in actions { apply_action(action, &mut self.domain, diff)?; }
        while !diff.is_empty() {
            propagate_changes(diff, &mut self.ir_instance, &mut self.domain)?;
        }
        Ok(())
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
                    new_objs: &NewObjs,
                    diff: &mut DomainDiff) -> Result<Vec<Action>, ()> {
    let mut actions = Vec::new();
    debug!("adding objects {:?}", new_objs);
    domain.alloc(ir_instance, new_objs);
    actions.extend(init_domain_partial(domain, ir_instance, new_objs, diff)?);
    // Enforce invariants and call manual triggers.
    for &inst in &new_objs.instructions {
        actions.extend(operand::inst_invariants(ir_instance, ir_instance.inst(inst)));
    }
    Ok(actions)
}

/// Trigger to call when two dimensions are merged.
fn merge_dims(lhs: ir::dim::Id, rhs: ir::dim::Id, ir_instance: &mut ir::Function)
    -> Result<(NewObjs, Vec<Action>), ()>
{
    debug!("merge {:?} and {:?}", lhs, rhs);
    ir_instance.merge(lhs, rhs);
    Ok(Default::default())
}

/// Adds a iteration dimension to a basic block.
fn add_iteration_dim(ir_instance: &mut ir::Function,
                     bb: ir::BBId, dim: ir::dim::Id) -> NewObjs {
    debug!("set {:?} as iteration dim of {:?}", dim, bb);
    let mut new_objs = NewObjs::default();
    if ir_instance.set_iteration_dim(bb, dim) {
        new_objs.add_iteration_dim(bb, dim);
    }
    new_objs
}

/// Stores the objects created by a lowering.
#[derive(Default, Debug)]
pub struct NewObjs {
    instructions: Vec<ir::InstId>,
    dimensions: Vec<ir::dim::Id>,
    basic_blocks: Vec<ir::BBId>,
    mem_blocks: Vec<ir::mem::Id>,
    internal_mem_blocks: Vec<ir::mem::InternalId>,
    mem_insts: Vec<ir::InstId>,
    iteration_dims: Vec<(ir::BBId, ir::dim::Id)>,
}

impl NewObjs {
    /// Registers a new instruction.
    fn add_instruction(&mut self, inst: &ir::Instruction) {
        self.add_bb(inst);
        self.instructions.push(inst.id());
    }

    /// Registers a new memory instruction.
    fn add_mem_instruction(&mut self, inst: &ir::Instruction) {
        self.add_instruction(inst);
        self.mem_insts.push(inst.id());
    }

    /// Registers a new dimension.
    fn add_dimension(&mut self, dim: &ir::Dimension) {
        self.add_bb(dim);
        self.dimensions.push(dim.id());
    }

    /// Registers a new basic block.
    fn add_bb(&mut self, bb: &ir::BasicBlock) {
        self.basic_blocks.push(bb.bb_id());
        for &dim in bb.iteration_dims() { self.iteration_dims.push((bb.bb_id(), dim)); }
    }

    /// Sets a dimension as a new iteration dimension.
    fn add_iteration_dim(&mut self, bb: ir::BBId, dim: ir::dim::Id) {
        self.iteration_dims.push((bb, dim));
    }

    /// Registers a new memory block.
    fn add_mem_block(&mut self, id: ir::mem::InternalId) {
        self.mem_blocks.push(id.into());
        self.internal_mem_blocks.push(id);
    }
}
