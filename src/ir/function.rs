//! Provides a representation of functions.
use device::Device;
use ir::{self, Dimension, InstId, Instruction, Operator, Statement, StmtId};
use ir::{mem, Operand, SparseVec};
use itertools::Itertools;
use std;
use utils::*;

/// Represents an argument of a function.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Parameter {
    /// The name of the `Parameter`
    pub name: String,
    /// The type of the `Parameter`.
    pub t: ir::Type,
    /// If the parameter point to an array, indicates the element type.
    pub elem_t: Option<ir::Type>,
}

/// Holds the signature of a function.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Signature {
    /// Mame of the function.
    pub name: String,
    /// Arguments of the function.
    pub params: Vec<Parameter>,
}

impl Signature {
    /// Creates a new signature without any parameter.
    pub fn new(name: String) -> Self {
        Signature {
            name,
            params: vec![],
        }
    }

    /// Adds a scalar parameter.
    pub fn add_scalar(&mut self, name: String, t: ir::Type) {
        self.params.push(Parameter {
            name,
            t,
            elem_t: None,
        });
    }

    /// Adds a parameter with the given name and type to the signature.
    pub fn add_array(&mut self, device: &Device, name: String, elem_t: ir::Type) {
        self.params.push(Parameter {
            name,
            t: device.pointer_type(ir::MemorySpace::Global),
            elem_t: Some(elem_t),
        });
    }
}

/// Describes a function and the set of its possible implementation.
///
/// The type parameter `L` indicates if the function is fozen (`L = ir::FutureMemAccess`) or
/// not (`L = ())`. A frozen function cannot have any more IDs allocated. We use this to
/// know the exact list of potential decisions before we run the exploration.
#[derive(Clone)]
pub struct Function<'a, L = ir::FutureMemAccess> {
    signature: &'a Signature,
    device: &'a Device,
    insts: SparseVec<ir::InstId, Instruction<'a>>,
    dims: SparseVec<ir::DimId, Dimension<'a>>,
    static_dims: Vec<ir::DimId>,
    thread_dims: VecSet<ir::DimId>,
    mem_insts: Vec<ir::InstId>,
    mem_blocks: Vec<ir::mem::Block>,
    induction_vars: Vec<ir::InductionVar<'a>>,
    logical_dims: Vec<ir::LogicalDim<'a>>,
    variables: SparseVec<ir::VarId, ir::Variable<L>>,
    layout_dims: SparseVec<ir::LayoutDimId, ir::LayoutDimension>,
    // We list layouts of in-memory variables in a separate vector so we can iterate on
    // them faster.
    mem_layout_dims: Vec<ir::LayoutDimId>,
    memory_vars: Vec<ir::VarId>,
    available_shared_mem: u32,
}

impl<'a, L> Function<'a, L> {
    /// Creates a new function.
    pub fn new(signature: &'a Signature, device: &'a Device) -> Self {
        Function {
            signature,
            device,
            insts: SparseVec::new(),
            mem_insts: vec![],
            dims: SparseVec::new(),
            static_dims: vec![],
            thread_dims: VecSet::default(),
            mem_blocks: Vec::new(),
            induction_vars: Vec::new(),
            logical_dims: Vec::new(),
            variables: SparseVec::new(),
            layout_dims: SparseVec::new(),
            mem_layout_dims: Vec::new(),
            memory_vars: Vec::new(),
            available_shared_mem: device.shared_mem(),
        }
    }

    /// Returns the function signature.
    pub fn signature(&self) -> &'a Signature {
        self.signature
    }

    /// Returns the device the function is compiled for.
    pub fn device(&self) -> &'a Device {
        self.device
    }

    /// Creates a new instruction (with given ID) without adding it to
    /// the `insts` vector. Used as an internal helper for when either
    /// adding a new instruction (`add_inst`) or filling an existing
    /// hole in the instructions vector.
    fn create_inst(
        &mut self,
        id: InstId,
        op: Operator<'a>,
        iter_dims: HashSet<ir::DimId>,
        mem_access_layout: VecSet<ir::LayoutDimId>,
    ) -> Result<ir::Instruction<'a>, ir::Error> {
        // Create and check the instruction.
        let mut inst = ir::Instruction::new(op, id, iter_dims, mem_access_layout, self)?;
        // Register the instruction in iteration dimensions.
        for &dim in inst.iteration_dims() {
            self.dim_mut(dim).add_iterated(id);
        }
        // Register the memory blocks used.
        if inst.operator().is_mem_access() {
            self.mem_insts.push(id);
        }
        if let Some(var_id) = inst.operator().loaded_mem_var() {
            let dir = ir::variable::MemAccessDirection::Load;
            self.variable_mut(var_id).register_mem_access(id, dir);
        }
        if let Some(var_id) = inst.operator().stored_mem_var() {
            let dir = ir::variable::MemAccessDirection::Store;
            self.variable_mut(var_id).register_mem_access(id, dir);
        }
        // Update the usepoint of all variables
        for var_id in inst.used_vars().clone() {
            let stmt: &mut ir::Statement = &mut inst;
            self.register_var_use(var_id, stmt.into());
        }
        Ok(inst)
    }

    /// Returns a variable without adding it to self.variables.
    fn create_variable(
        &self,
        id: ir::VarId,
        def: ir::VarDef,
        layout: VecSet<ir::LayoutDimId>,
        def_mode: ir::VarDefMode<L>,
        use_mode: ir::VarUseMode<L>,
    ) -> Result<ir::Variable<L>, ir::Error> {
        def.check(self)?;
        Ok(ir::Variable::new(id, def, layout, def_mode, use_mode, self))
    }

    /// Registers a layout dimension in the function.
    fn register_layout_dim(&mut self, dim: ir::LayoutDimension) {
        dim.register(self);
        if dim.is_memory_layout() {
            self.mem_layout_dims.push(dim.id());
        }
        // Create holes in the sparse vector so we can ues this function both with fresh
        // and pre-allocated IDs.
        // available_shared_mem: device.shared_mem(),
        let id_idx: usize = dim.id().into();
        if self.layout_dims.len() <= id_idx {
            self.layout_dims.expand_to(id_idx + 1);
        }
        self.layout_dims.set_lazy(dim.id(), dim);
    }

    /// Adds an induction variable.
    pub fn add_ind_var(&mut self, ind_var: ir::InductionVar<'a>) -> ir::IndVarId {
        let id = ir::IndVarId(self.induction_vars.len() as u32);
        self.induction_vars.push(ind_var);
        id
    }

    /// Returns the list of instructions of the function.
    pub fn insts<'b>(&'b self) -> impl Iterator<Item = &'b Instruction<'a>> {
        self.insts.iter()
    }

    /// Returns the list of dimensions of the function.
    pub fn dims<'b>(&'b self) -> impl Iterator<Item = &'b Dimension<'a>> + Clone {
        self.dims.iter()
    }

    /// Returns the list of logical dimensions.
    pub fn logical_dims(&self) -> impl Iterator<Item = &ir::LogicalDim<'a>> {
        self.logical_dims.iter()
    }

    /// Returns the list of stastic dimensions in the function.
    pub fn static_dims<'b>(&'b self) -> impl Iterator<Item = &'b Dimension<'a>> {
        self.static_dims.iter().map(move |&id| self.dim(id))
    }

    pub fn variables(&self) -> impl Iterator<Item = &ir::Variable<L>> {
        self.variables.iter()
    }

    /// Returns the list of thread dimensions.
    pub fn thread_dims(&self) -> impl Iterator<Item = &Dimension<'a>> {
        self.thread_dims.iter().map(move |&d| self.dim(d))
    }

    /// Returns an instruction given its id.
    pub fn inst(&self, id: InstId) -> &Instruction<'a> {
        &self.insts[id]
    }

    /// Returns a mutable reference to an instruction given its id.
    pub(super) fn inst_mut(&mut self, id: InstId) -> &mut Instruction<'a> {
        &mut self.insts[id]
    }

    /// Retuns a dimension given its id.
    pub fn dim(&self, id: ir::DimId) -> &Dimension<'a> {
        &self.dims[id]
    }

    /// Returns a mutable reference to a dimension given its ID.
    pub(super) fn dim_mut(&mut self, id: ir::DimId) -> &mut Dimension<'a> {
        &mut self.dims[id]
    }

    /// Returns a mutable reference to a statement given its id.
    pub(super) fn statement_mut(&mut self, id: ir::StmtId) -> &mut Statement<'a> {
        match id {
            StmtId::Inst(id) => self.inst_mut(id),
            StmtId::Dim(id) => self.dim_mut(id),
        }
    }

    /// Returns a `Statement` given its id.
    pub fn statement(&self, id: StmtId) -> &Statement<'a> {
        match id {
            StmtId::Inst(id) => &self.insts[id],
            StmtId::Dim(id) => self.dim(id),
        }
    }

    /// Lists all `Statement`s.
    pub fn statements<'b>(&'b self) -> impl Iterator<Item = &'b Statement<'a>> {
        self.insts()
            .map(|x| x as _)
            .chain(self.dims().map(|x| x as _))
    }

    /// Retrives a logical dimension given its ID.
    pub fn logical_dim(&self, id: ir::LogicalDimId) -> &ir::LogicalDim<'a> {
        &self.logical_dims[id.0 as usize]
    }

    /// Returns a `Variable` given its id.
    pub fn variable(&self, id: ir::VarId) -> &ir::Variable<L> {
        &self.variables[id]
    }

    /// Returns a mutable reference to a `Variable` given its id.
    pub(super) fn variable_mut(&mut self, id: ir::VarId) -> &mut ir::Variable<L> {
        &mut self.variables[id]
    }

    /// Returns the list of memory blocks. The block with id `i` is in i-th position.
    pub fn mem_blocks<'b>(&'b self) -> impl Iterator<Item = &'b mem::Block> {
        self.mem_blocks.iter()
    }

    /// Iterates over memory instructions.
    pub fn mem_insts<'b>(&'b self) -> impl Iterator<Item = &'b Instruction<'a>> + 'b {
        self.mem_insts.iter().map(move |&id| self.inst(id))
    }

    /// Returns a memory block given its id.
    pub fn mem_block(&self, id: ir::MemId) -> &mem::Block {
        &self.mem_blocks[id.0 as usize]
    }

    /// Retrieves an induction variable given its Id.
    pub fn induction_var(&self, id: ir::IndVarId) -> &ir::InductionVar<'_> {
        &self.induction_vars[id.0 as usize]
    }

    /// Iterates over induction variables.
    pub fn induction_vars<'b>(
        &'b self,
    ) -> impl Iterator<Item = (ir::IndVarId, &'b ir::InductionVar<'a>)> {
        self.induction_vars
            .iter()
            .enumerate()
            .map(|(id, v)| (ir::IndVarId(id as u32), v))
    }

    /// Sets a dimension as an iteration dimension for an instruction. Indicates if the
    /// iteration dimension was not aleady present in the set.
    pub fn set_iteration_dim(&mut self, inst: ir::InstId, dim: ir::DimId) -> bool {
        if self.inst_mut(inst).add_iteration_dimension(dim) {
            self.dim_mut(dim).add_iterated(inst);
            true
        } else {
            false
        }
    }

    /// Adds a thread dimension. Indicates if the the dimension was not already present
    /// in the set.
    pub fn add_thread_dim(&mut self, dim: ir::DimId) -> bool {
        self.dim_mut(dim).set_thread_dim();
        self.thread_dims.insert(dim)
    }

    /// Registers that `var` is used by `stmt`.
    pub(super) fn register_var_use(
        &mut self,
        var: ir::VarId,
        mut stmt: ir::statement::IdOrMut<'a, '_>,
    ) {
        stmt.get_statement(self).register_used_var(var);
        let pred = {
            let var = self.variable_mut(var);
            var.add_use(stmt.id());
            var.def().origin()
        };
        if let Some(var) = pred {
            self.register_var_use(var, stmt);
        }
    }

    /// Sets the loop-carried dependency of a `fby` variable.
    pub fn set_loop_carried_variable(
        &mut self,
        fby: ir::VarId,
        loop_carried: ir::VarId,
    ) -> Result<(), ir::Error> {
        let mut fby = unwrap!(self.variables.remove(fby));
        fby.set_loop_carried_variable(loop_carried);
        fby.def().check(self)?;
        fby.register(self);
        self.variables.set_lazy(fby.id(), fby);
        Ok(())
    }

    /// Returns a layout dimension given its ID.
    pub fn layout_dimension(&self, id: ir::LayoutDimId) -> &ir::LayoutDimension {
        &self.layout_dims[id]
    }

    /// Returns a layout dimension given its ID.
    pub(super) fn layout_dimension_mut(
        &mut self,
        id: ir::LayoutDimId,
    ) -> &mut ir::LayoutDimension {
        &mut self.layout_dims[id]
    }

    /// Iterates over all layout dimensions.
    pub fn layout_dimensions(&self) -> impl Iterator<Item = &ir::LayoutDimension> + '_ {
        self.layout_dims.iter()
    }

    /// Iterate over memory layout dimensions.
    pub fn mem_layout_dimensions(
        &self,
    ) -> impl Iterator<Item = &ir::LayoutDimension> + '_ {
        self.mem_layout_dims
            .iter()
            .map(move |&id| self.layout_dimension(id))
    }

    /// Indicates how much shared memory is available for variables.
    pub fn available_shared_mem(&self) -> u32 {
        self.available_shared_mem
    }

    /// Lists variables corresponding to memory blocks to allocate.
    pub fn memory_vars(&self) -> impl Iterator<Item = &ir::Variable<L>> {
        self.memory_vars.iter().map(move |&id| self.variable(id))
    }
}

impl<'a> Function<'a, ()> {
    /// Adds an instruction to the function.
    pub fn add_inst(
        &mut self,
        op: Operator<'a>,
        iter_dims: HashSet<ir::DimId>,
    ) -> Result<InstId, ir::Error> {
        trace!(
            "adding instruction with operator {:?} and dims {:?}",
            op,
            iter_dims
        );
        // Create dimension mappings for the operands.
        // TODO(cleanup): the operands should list `DimMapping` rather that pairs of
        // dimensions so `DimMapping` should be allocated before.
        let id = ir::InstId(self.insts.len() as u32);
        let mem_access_layout = op
            .mem_access_pattern()
            .map(|pattern| {
                let layout_ids: VecSet<_> = (self.layout_dims.len()..)
                    .take(pattern.num_layout_dimensions())
                    .map(ir::LayoutDimId)
                    .collect();
                let dims = ir::LayoutDimension::from_access_pattern(
                    &layout_ids,
                    id,
                    &pattern,
                    self,
                );
                for dim in dims {
                    self.register_layout_dim(dim);
                }
                layout_ids
            }).unwrap_or_default();
        let inst = self.create_inst(id, op, iter_dims, mem_access_layout)?;
        self.insts.push(inst);
        Ok(id)
    }

    /// Allocates a new memory block.
    pub fn add_mem_block(
        &mut self,
        elements_type: ir::Type,
        len: u32,
        space: ir::MemorySpace,
    ) -> ir::MemId {
        let id = ir::mem::MemId(self.mem_blocks.len() as u32);
        let block = ir::mem::Block {
            id,
            elements_type,
            len,
            space,
        };
        if space == ir::MemorySpace::Shared {
            assert!(block.byte_size() <= self.available_shared_mem);
            self.available_shared_mem -= block.byte_size();
        }
        self.mem_blocks.push(block);
        id
    }

    /// Create a new logical dimension composed of multiple dimensions to implement
    /// strip-mining.
    pub fn add_logical_dim(
        &mut self,
        size: ir::Size<'a>,
        tiling_factors: VecSet<u32>,
        possible_tile_sizes: Vec<VecSet<u32>>,
    ) -> Result<(ir::LogicalDimId, Vec<ir::DimId>), ir::Error> {
        // TODO(strip-mining): allow all tiling factors at all levels
        let logical_id = ir::LogicalDimId(self.logical_dims.len() as u32);
        let dim_ids = (0..possible_tile_sizes.len() + 1)
            .map(|id| ir::DimId((id + self.dims.len()) as u32))
            .collect_vec();
        // Create the objects, but don't add anythin yet so we can rollback if an error
        // occurs.
        let mut dims = Vec::new();
        let mut possible_tiled_sizes = VecSet::default();
        let logical_dim = if let Some(size) = size.as_constant() {
            possible_tiled_sizes =
                tiling_factors.iter().map(|factor| size / factor).collect();
            ir::LogicalDim::new_static(logical_id, dim_ids.clone(), size)
        } else {
            ir::LogicalDim::new_dynamic(
                logical_id,
                dim_ids[0],
                dim_ids[1..].to_vec(),
                tiling_factors,
                size.clone(),
            )
        };
        let mut tiled_size: ir::PartialSize = size.into();
        tiled_size.add_divisors(&VecSet::new(dim_ids[1..].to_vec()));
        dims.push(Dimension::new(
            dim_ids[0],
            tiled_size,
            possible_tiled_sizes,
            Some(logical_id),
        )?);
        for (&id, sizes) in dim_ids[1..].iter().zip_eq(possible_tile_sizes) {
            dims.push(Dimension::new_tile(id, sizes, logical_id)?);
        }
        // Register the new objects.
        for dim in &dims {
            if dim.possible_sizes().is_some() {
                self.static_dims.push(dim.id());
            }
        }
        self.dims.extend(dims);
        self.logical_dims.push(logical_dim);
        Ok((logical_id, dim_ids))
    }

    /// Adds a variable to the function. Also register its definition into the relevant instruction
    pub fn add_variable(
        &mut self,
        def: ir::VarDef,
        def_mode: ir::VarDefMode<()>,
        use_mode: ir::VarUseMode<()>,
    ) -> Result<ir::VarId, ir::Error> {
        let id = ir::VarId(self.variables.len() as u16);
        let layout = def
            .dimensions(self)
            .into_iter()
            .map(|dim| {
                let layout_id = ir::LayoutDimId(self.layout_dims.len());
                let layout_dim = ir::LayoutDimension::new_dynamic(layout_id, dim, id);
                self.register_layout_dim(layout_dim);
                layout_id
            }).collect();
        let var = self.create_variable(id, def, layout, def_mode, use_mode)?;
        trace!("adding variable {:?}", var);
        var.register(self);
        if var.is_memory() {
            self.memory_vars.push(id);
        }
        self.variables.push(var);
        Ok(id)
    }

    pub(crate) fn freeze(self) -> Function<'a> {
        let mut counter = ir::Counter {
            next_inst: self.insts.len(),
            next_dim: self.dims.len(),
            next_layout_dim: self.layout_dims.len(),
            next_variable: self.variables.len(),
        };
        let Function {
            signature,
            device,
            mut insts,
            mut dims,
            static_dims,
            thread_dims,
            mem_insts,
            mem_blocks,
            induction_vars,
            logical_dims,
            variables,
            mut layout_dims,
            mem_layout_dims,
            memory_vars,
            available_shared_mem,
        } = self;

        let mut variables = SparseVec::from_vec(
            variables
                .into_iter()
                .map(|var| var.map(|var| var.freeze(&mut counter)))
                .collect(),
        );

        let ir::Counter {
            next_inst,
            next_dim,
            next_layout_dim,
            next_variable,
        } = counter;
        insts.expand_to(next_inst);
        dims.expand_to(next_dim);
        variables.expand_to(next_variable);
        layout_dims.expand_to(next_layout_dim);

        Function {
            signature,
            device,
            insts,
            dims,
            static_dims,
            thread_dims,
            mem_insts,
            mem_blocks,
            induction_vars,
            logical_dims,
            variables,
            layout_dims,
            mem_layout_dims,
            memory_vars,
            available_shared_mem,
        }
    }
}

impl<'a> Function<'a> {
    /// Lowers a variable definition when a copy is needed.
    pub fn lower_var_def(&mut self, var_id: ir::VarId, new_objs: &mut ir::NewObjs) {
        // The variable corresponds to a memory block that will be reused by all dependent
        // variables. If the variable is use in a Fby instruction, it aliases with the
        // Fby variable instead. This works because We allow a single lowering to memory
        // in a chain of dependencies. Otherwise, `var_id` may be loaded back into
        // registers before it is used by the fby variable. In that case it would not
        // alias and we would need to allocate it into memory. We have currently no way of
        // doing that.
        // TODO(ulysse): allow non-aliasing `FbyPrev` variables.
        if self.variable(var_id).consumer() != ir::variable::Consumer::FbyPrev {
            let var = &mut self.variables[var_id];
            if !var.is_memory() {
                var.set_is_memory();
                self.memory_vars.push(var_id);
                new_objs.memory_vars.push(var_id);
            }
        }
        if let Some(new_ids) = self.variable_mut(var_id).lower_def() {
            assert!(
                new_ids.output_var.is_none(),
                "stores have no output variables"
            );
            trace!("lowering {:?} definition with ids {:?}", var_id, new_ids);
            // Activate the store instruction.
            let t = self.variable(var_id).t();
            let value = ir::Operand::Variable(new_ids.dim_map_var.id, t);
            self.activate_mapped_dims(&new_ids.dim_mapping, new_objs);
            self.activate_variable(new_ids.dim_map_var, new_objs);
            let store_id = new_ids.inst.id;
            let pattern = new_ids.access_pattern;
            self.activate_layout(&new_ids.inst.layout, store_id, &pattern, new_objs);
            let addr = self.index_variable(var_id, &new_ids.inst.layout);
            let store = ir::Operator::St(addr, value, false, pattern);
            self.activate_instruction(store, new_ids.inst, new_objs);
            // Register that `store_id` defines `var_id` as this is not done automatically by
            // `activate_instruction`.
            new_objs.def_statements.push((var_id, store_id.into()));
        }
    }

    /// Registers a variable's layout as a memory layout if not already done.
    pub fn lower_layout(&mut self, var_id: ir::VarId, new_objs: &mut ir::NewObjs) {
        for id in self.variable(var_id).layout().clone() {
            self.mem_layout_dims.push(id);
            let layout_dim = self.layout_dimension_mut(id);
            if layout_dim.is_memory_layout() {
                continue;
            }
            layout_dim.set_possible_ranks((0..16).collect());
            new_objs.mem_layout_dims.push(id);
            new_objs.mem_var_layout.push((var_id, id));
            for &successor in layout_dim.successors() {
                new_objs.predecessor_mem_layout_dims.push((successor, id));
            }
        }
    }

    /// Lowers accesses to a variable when not placed in registers.
    pub fn lower_uses(&mut self, var_id: ir::VarId, new_objs: &mut ir::NewObjs) {
        if let Some(new_ids) = self.variable_mut(var_id).lower_uses() {
            // Update the operands using the variable.
            for var_use in self.variables[var_id].use_points() {
                if let ir::StmtId::Inst(var_use) = var_use {
                    let new_var = new_ids.dim_map_var.id;
                    self.insts[var_use].rename_var(var_id, new_var);
                    new_objs.use_statements.push((var_id, var_use.into()));
                }
            }
            // Add the load instruction.
            self.activate_mapped_dims(&new_ids.dim_mapping, new_objs);
            let load_id = new_ids.inst.id;
            let pattern = new_ids.access_pattern;
            self.activate_layout(&new_ids.inst.layout, load_id, &pattern, new_objs);
            let addr = self.index_variable(var_id, &new_ids.inst.layout);
            let load = ir::Operator::Ld(self.variable(var_id).t(), addr, pattern);
            self.activate_instruction(load, new_ids.inst, new_objs);
            self.activate_variable(unwrap!(new_ids.output_var), new_objs);
            self.activate_variable(new_ids.dim_map_var, new_objs);
        };
    }

    /// Creates an instruction with pre-allocated IDs.
    fn activate_instruction(
        &mut self,
        operator: ir::Operator<'a>,
        ids: ir::FutureInstruction,
        new_objs: &mut ir::NewObjs,
    ) {
        let ir::FutureInstruction {
            id,
            dimensions,
            layout,
        } = ids;
        let inst = unwrap!(self.create_inst(id, operator, dimensions, layout));
        new_objs.add_instruction(&inst);
        self.insts.set_lazy(id, inst);
    }

    /// Creates a variable with preallocated IDs.
    fn activate_variable(&mut self, var: ir::FutureVariable, new_objs: &mut ir::NewObjs) {
        let ir::FutureVariable { id, def, layout } = var;
        let def_mode = ir::VarDefMode::InPlace { allow_sync: false };
        let use_mode = ir::VarUseMode::FromRegisters;
        // Activate the variable layout.
        for (&layout_id, dim) in layout.iter().zip_eq(def.dimensions(self)) {
            let layout_dim = ir::LayoutDimension::new_dynamic(layout_id, dim, id);
            self.register_layout_dim(layout_dim);
        }
        // Activate the variable.
        let var = unwrap!(self.create_variable(id, def, layout, def_mode, use_mode));
        var.register(self);
        if var.is_memory() {
            self.memory_vars.push(id);
        }
        new_objs.add_variable(&var, self);
        self.variables.set_lazy(id, var);
    }

    /// Creates new layout dimensions from an access pattern with preallocated IDs.
    fn activate_layout(
        &mut self,
        ids: &[ir::LayoutDimId],
        inst: ir::InstId,
        pattern: &ir::AccessPattern,
        new_objs: &mut ir::NewObjs,
    ) {
        let layout = ir::LayoutDimension::from_access_pattern(ids, inst, pattern, self);
        for dim in layout {
            new_objs.add_layout_dim(&dim, self);
            self.register_layout_dim(dim);
        }
    }

    /// Create dimensions with preallocated IDs and maps then to existing dimensions. `mapping`
    /// provides the pairs of old and new IDs along with the mapping ID.
    fn activate_mapped_dims(
        &mut self,
        mappings: &[(ir::DimId, ir::DimId)],
        new_objs: &mut ir::NewObjs,
    ) {
        for &(old_dim, new_dim) in mappings {
            let dimension = Dimension::with_same_size(new_dim, &self.dims[old_dim]);
            new_objs.add_dimension(&dimension);
            if dimension.possible_sizes().is_some() {
                self.static_dims.push(new_dim);
            }
            self.dims.set_lazy(new_dim, dimension);
        }
    }

    /// Generates an access pattern and the corresponding induction variable to access a
    /// memory block.
    fn index_variable(
        &mut self,
        var: ir::VarId,
        layout: &[ir::LayoutDimId],
    ) -> Operand<'a> {
        let array_id = ir::ArrayId::Variable(var);
        let increments = layout
            .iter()
            .map(|&id| (self.layout_dimension(id).dim(), id.into()))
            .collect();
        let ind_var = ir::InductionVar::new(increments, Operand::Addr(array_id), self);
        let ind_var = self.add_ind_var(unwrap!(ind_var));
        ir::Operand::InductionVar(ind_var, ir::Type::PtrTo(array_id))
    }
}

impl<'a> std::ops::Deref for Function<'a> {
    type Target = Signature;

    fn deref(&self) -> &Self::Target {
        self.signature
    }
}
