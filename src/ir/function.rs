//! Provides a representation of functions.
use std::sync::Arc;
use std::{self, fmt};

use crate::device::Device;
use crate::ir::{
    self, Dimension, InstId, Instruction, IrDisplay, Operator, Statement, StmtId,
};
use crate::ir::{mem, AccessPattern, Operand, SparseVec};
use crate::search_space::MemSpace;
use itertools::Itertools;
use log::debug;
use serde::{Deserialize, Serialize};
use utils::*;

/// Represents an argument of a function.
#[derive(PartialEq, Eq, Hash, Clone, Debug, Serialize, Deserialize)]
pub struct Parameter {
    /// The name of the `Parameter`
    pub name: String,
    /// The type of the `Parameter`.
    pub t: ir::Type,
    /// If the parameter point to an array, indicates the element type.
    pub elem_t: Option<ir::Type>,
}

impl fmt::Display for Parameter {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.name)
    }
}

/// Holds the signature of a function.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Signature {
    /// Mame of the function.
    pub name: String,
    /// Arguments of the function.
    pub params: Vec<Arc<Parameter>>,
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
        self.params.push(Arc::new(Parameter {
            name,
            t,
            elem_t: None,
        }));
    }

    /// Adds a parameter with the given name and type to the signature.
    pub fn add_array(&mut self, device: &dyn Device, name: String, elem_t: ir::Type) {
        self.params.push(Arc::new(Parameter {
            name,
            t: device.pointer_type(MemSpace::GLOBAL),
            elem_t: Some(elem_t),
        }));
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Body<L = ir::LoweringMap> {
    insts: SparseVec<ir::InstId, Instruction<L>>,
    dims: SparseVec<ir::DimId, Dimension<L>>,
    static_dims: Vec<ir::DimId>,
    thread_dims: VecSet<ir::DimId>,
    mem_insts: Vec<ir::InstId>,
    mem_blocks: mem::BlockMap,
    layouts_to_lower: Vec<ir::MemId>,
    induction_vars: Vec<ir::InductionVar<L>>,
    logical_dims: Vec<ir::LogicalDim>,
    dim_mappings: SparseVec<ir::DimMappingId, ir::DimMapping>,
    variables: SparseVec<ir::VarId, ir::Variable>,
}

impl<L> Body<L> {
    fn new() -> Self {
        Body {
            insts: SparseVec::new(),
            mem_insts: vec![],
            dims: SparseVec::new(),
            static_dims: vec![],
            thread_dims: VecSet::default(),
            mem_blocks: mem::BlockMap::default(),
            layouts_to_lower: Vec::new(),
            induction_vars: Vec::new(),
            logical_dims: Vec::new(),
            dim_mappings: SparseVec::new(),
            variables: SparseVec::new(),
        }
    }
}

impl<L> Default for Body<L> {
    fn default() -> Self {
        Body::new()
    }
}

/// Describes a function and the set of its possible implementation.
///
/// The type parameter `L` indicates if the function is fozen (`L = ir::LoweringMap`) or
/// not (`L = ())`. A frozen function cannot have any more IDs allocated. We use this to
/// know the exact list of potential decisions before we run the exploration.
#[derive(Clone)]
pub struct Function<L = ir::LoweringMap> {
    signature: Arc<Signature>,
    device: Arc<dyn Device>,
    body: Body<L>,
}

impl<L> Function<L> {
    /// Creates a new function.
    pub fn new(signature: Arc<Signature>, device: Arc<dyn Device>) -> Self {
        Self::from_body(signature, device, Body::new())
    }

    pub fn from_body(
        signature: Arc<Signature>,
        device: Arc<dyn Device>,
        body: Body<L>,
    ) -> Self {
        Function {
            signature,
            device,
            body,
        }
    }

    /// Returns the function's body
    pub fn body(&self) -> &Body<L> {
        &self.body
    }

    /// Returns the function signature.
    pub fn signature(&self) -> &Signature {
        &*self.signature
    }

    /// Returns the function's name.
    pub fn name(&self) -> &str {
        &self.signature.name
    }

    /// Returns the device the function is compiled for.
    pub fn device(&self) -> &dyn Device {
        &*self.device
    }

    /// Creates a new instruction (with given ID) without adding it to
    /// the `insts` vector. Used as an internal helper for when either
    /// adding a new instruction (`add_inst`) or filling an existing
    /// hole in the instructions vector.
    fn create_inst(
        &mut self,
        id: InstId,
        op: Operator<L>,
        iter_dims: FnvHashSet<ir::DimId>,
    ) -> Result<ir::Instruction<L>, ir::Error> {
        // Create and check the instruction.
        let inst = ir::Instruction::new(op, id, iter_dims, self)?;
        // Register the instruction in iteration dimensions.
        for &dim in inst.iteration_dims() {
            self.dim_mut(dim).add_iterated(id);
        }
        // Register the memory blocks used.
        if inst.operator().is_mem_access() {
            self.body.mem_insts.push(id);
        }
        if let Some(mem_id) = inst.operator().mem_used() {
            self.body.mem_blocks.register_use(mem_id, id);
        }
        // Update the usepoint of all variables
        for &var_id in inst.used_vars() {
            self.body.variables[var_id].add_use(id.into());
        }
        Ok(inst)
    }

    /// Returns a variable without adding it to self.variables.
    fn create_variable(
        &self,
        id: ir::VarId,
        def: ir::VarDef,
    ) -> Result<ir::Variable, ir::Error> {
        def.check(self)?;
        Ok(ir::Variable::new(id, def, self))
    }

    /// Adds an induction variable.
    pub fn add_ind_var(&mut self, ind_var: ir::InductionVar<L>) -> ir::IndVarId {
        let id = ir::IndVarId(self.body.induction_vars.len() as u32);
        self.body.induction_vars.push(ind_var);
        id
    }

    /// Returns the list of instructions of the function.
    pub fn insts<'b>(&'b self) -> impl Iterator<Item = &'b Instruction<L>> {
        self.body.insts.iter()
    }

    /// Returns the list of dimensions of the function.
    pub fn dims<'b>(&'b self) -> impl Iterator<Item = &'b Dimension<L>> + Clone {
        self.body.dims.iter()
    }

    /// Returns the list of logical dimensions.
    pub fn logical_dims(&self) -> impl Iterator<Item = &ir::LogicalDim> {
        self.body.logical_dims.iter()
    }

    /// Returns the list of stastic dimensions in the function.
    pub fn static_dims<'b>(&'b self) -> impl Iterator<Item = &'b Dimension<L>> {
        self.body.static_dims.iter().map(move |&id| self.dim(id))
    }

    pub fn variables(&self) -> impl Iterator<Item = &ir::Variable> {
        self.body.variables.iter()
    }

    /// Returns the list of thread dimensions.
    pub fn thread_dims(&self) -> impl Iterator<Item = &Dimension<L>> {
        self.body.thread_dims.iter().map(move |&d| self.dim(d))
    }

    /// Returns an instruction given its id.
    pub fn inst(&self, id: InstId) -> &Instruction<L> {
        &self.body.insts[id]
    }

    /// Returns a mutable reference to an instruction given its id.
    pub(super) fn inst_mut(&mut self, id: InstId) -> &mut Instruction<L> {
        &mut self.body.insts[id]
    }

    /// Retuns a dimension given its id.
    pub fn dim(&self, id: ir::DimId) -> &Dimension<L> {
        &self.body.dims[id]
    }

    /// Returns a mutable reference to a dimension given its ID.
    pub(super) fn dim_mut(&mut self, id: ir::DimId) -> &mut Dimension<L> {
        &mut self.body.dims[id]
    }

    /// Returns a mutable reference to a statement given its id.
    pub(super) fn statement_mut(&mut self, id: ir::StmtId) -> &mut dyn Statement<L> {
        match id {
            StmtId::Inst(id) => self.inst_mut(id),
            StmtId::Dim(id) => self.dim_mut(id),
        }
    }

    /// Returns a `Statement` given its id.
    pub fn statement(&self, id: StmtId) -> &dyn Statement<L> {
        match id {
            StmtId::Inst(id) => &self.body.insts[id],
            StmtId::Dim(id) => self.dim(id),
        }
    }

    /// Lists all `Statement`s.
    pub fn statements<'b>(&'b self) -> impl Iterator<Item = &'b dyn Statement<L>> {
        self.insts()
            .map(|x| x as _)
            .chain(self.dims().map(|x| x as _))
    }

    /// Retrives a logical dimension given its ID.
    pub fn logical_dim(&self, id: ir::LogicalDimId) -> &ir::LogicalDim {
        &self.body.logical_dims[id.0 as usize]
    }

    /// Returns a `Variable` given its id.
    pub fn variable(&self, id: ir::VarId) -> &ir::Variable {
        &self.body.variables[id]
    }

    /// Adds a variable to the function. Also register its definition into the relevant instruction
    pub fn add_variable(&mut self, def: ir::VarDef) -> Result<ir::VarId, ir::Error> {
        let id = ir::VarId(self.body.variables.len() as u16);
        let var = self.create_variable(id, def)?;
        var.register(self);
        self.body.variables.push(var);
        Ok(id)
    }

    /// Returns the list of memory blocks. The block with id `i` is in i-th position.
    pub fn mem_blocks(&self) -> impl Iterator<Item = &mem::Block> {
        self.body.mem_blocks.blocks()
    }

    /// Iterates over memory instructions.
    pub fn mem_insts<'b>(&'b self) -> impl Iterator<Item = &'b Instruction<L>> + 'b {
        self.body.mem_insts.iter().map(move |&id| self.inst(id))
    }

    /// Returns a memory block given its id.
    pub fn mem_block(&self, id: ir::MemId) -> &mem::Block {
        self.body.mem_blocks.block(id)
    }

    /// Retrieves an induction variable given its Id.
    pub fn induction_var(&self, id: ir::IndVarId) -> &ir::InductionVar<L> {
        &self.body.induction_vars[id.0 as usize]
    }

    /// Iterates over induction variables.
    pub fn induction_vars<'b>(
        &'b self,
    ) -> impl Iterator<Item = (ir::IndVarId, &'b ir::InductionVar<L>)> {
        self.body
            .induction_vars
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
        self.body.thread_dims.insert(dim)
    }

    /// Trigger to call when two dimensions are merged.
    // TODO(cleanup): externalize in the search space the merging of dimensions in dim
    // maps.
    pub(crate) fn merge(&mut self, src: ir::DimId, dst: ir::DimId) {
        for inst in self.body.insts.iter_mut() {
            inst.merge_dims(src, dst);
        }
        for var in &mut self.body.induction_vars {
            var.merge_dims(src, dst);
        }
        self.body
            .layouts_to_lower
            .extend(self.body.mem_blocks.merge_dims(src, dst));
    }

    /// Lowers a layout into conventional memory accesses.
    pub(crate) fn lower_layout(
        &mut self,
        id: ir::MemId,
        st_dims: &[ir::DimId],
        ld_dims: &[ir::DimId],
    ) where
        L: Clone,
    {
        let pos = unwrap!(self.body.layouts_to_lower.iter().position(|&x| x == id));
        self.body.layouts_to_lower.swap_remove(pos);
        self.body.mem_blocks.lower_layout(id);
        let (st_index, st_pattern) = self.gen_internal_index(id, &st_dims);
        let (ld_index, ld_pattern) = self.gen_internal_index(id, &ld_dims);
        for &mem_use in self.body.mem_blocks.block(id).uses() {
            self.body.insts[mem_use].lower_layout(
                ld_index.clone(),
                ld_pattern.clone(),
                st_index.clone(),
                st_pattern.clone(),
            );
        }
    }

    /// Generates an operand repesenting a pointer to a cell of a memory block.
    fn gen_internal_index(
        &mut self,
        id: ir::MemId,
        dims: &[ir::DimId],
    ) -> (Operand<L>, AccessPattern) {
        let ty_len = self.body.mem_blocks.block(id).base_size();
        self.gen_index(id, ty_len, Operand::Addr(id), &dims)
    }

    /// Generates an access pattern and the corresponding induction variable to access a
    /// memory block.
    fn gen_index(
        &mut self,
        mem: ir::MemId,
        base_incr: u32,
        base_addr: Operand<L>,
        dims: &[ir::DimId],
    ) -> (Operand<L>, AccessPattern) {
        let var_type = base_addr.t();
        let base_size = ir::PartialSize::new(base_incr, vec![]);
        let increments = dims
            .iter()
            .rev()
            .scan(base_size, |size, &dim| {
                let old_size = size.clone();
                *size *= self.dim(dim).size();
                Some((dim, old_size))
            })
            .collect_vec();
        let pattern = ir::AccessPattern::Tensor {
            mem_id: Some(mem),
            dims: increments.iter().cloned().collect(),
        };
        let ind_var = unwrap!(ir::InductionVar::new(increments, base_addr));
        let ind_var = self.add_ind_var(ind_var);
        let addr = ir::Operand::InductionVar(ind_var, var_type);
        (addr, pattern)
    }

    /// Returns the list of layouts to lower.
    pub(crate) fn layouts_to_lower(&self) -> &[ir::MemId] {
        &self.body.layouts_to_lower
    }

    /// Returns the list of dimensions mapping.
    pub fn dim_mappings(&self) -> impl Iterator<Item = &ir::DimMapping> + '_ {
        self.body.dim_mappings.iter()
    }

    /// Retrives a dimension mapping given its ID.
    pub fn dim_mapping(&self, id: ir::DimMappingId) -> &ir::DimMapping {
        &self.body.dim_mappings[id]
    }

    /// Retrives a mutable reference to a dimension mapping given its ID.
    pub(super) fn dim_mapping_mut(
        &mut self,
        id: ir::DimMappingId,
    ) -> &mut ir::DimMapping {
        &mut self.body.dim_mappings[id]
    }

    /// Tries to find a mapping between two dimensions.
    pub fn find_mapping(
        &self,
        lhs: ir::DimId,
        rhs: ir::DimId,
    ) -> Option<ir::DimMappingId> {
        self.dim(lhs)
            .dim_mappings()
            .iter()
            .cloned()
            .find(|&id| self.dim_mapping(id).dims().contains(&rhs))
    }

    /// Creates a mapping between two dimensions.
    fn create_mapping(
        &mut self,
        id: ir::DimMappingId,
        dims: [ir::DimId; 2],
    ) -> ir::DimMapping {
        let mapping = ir::DimMapping::new(id, dims);
        for &dim in &dims {
            self.dim_mut(dim).register_dim_mapping(&mapping);
        }
        mapping
    }
}

impl Function<()> {
    /// Adds an instruction to the function.
    pub fn add_inst(
        &mut self,
        op: Operator<()>,
        iter_dims: FnvHashSet<ir::DimId>,
    ) -> Result<InstId, ir::Error> {
        // Create dimension mappings for the operands.
        // TODO(cleanup): the operands should list `DimMapping` rather that pairs of
        // dimensions so `DimMapping` should be allocated before.
        for operand in op.operands() {
            if let Some(dim_map) = operand.mapped_dims() {
                for &(lhs, rhs) in dim_map {
                    self.map_dimensions([lhs, rhs]);
                }
            }
        }
        let id = ir::InstId(self.body.insts.len() as u32);
        let inst = self.create_inst(id, op, iter_dims)?;
        self.body.insts.push(inst);
        Ok(id)
    }

    /// Allocates a new memory block.
    pub fn add_mem_block(&mut self, size: u32) -> ir::MemId {
        self.body.mem_blocks.alloc_block(size, None)
    }

    /// Create a new logical dimension composed of multiple dimensions to implement
    /// strip-mining.
    pub fn add_logical_dim(
        &mut self,
        size: ir::Size,
        tiling_factors: VecSet<u32>,
        possible_tile_sizes: Vec<VecSet<u32>>,
    ) -> Result<(ir::LogicalDimId, Vec<ir::DimId>), ir::Error> {
        // TODO(strip-mining): allow all tiling factors at all levels
        let logical_id = ir::LogicalDimId(self.body.logical_dims.len() as u32);
        let dim_ids = (0..=possible_tile_sizes.len())
            .map(|id| ir::DimId((id + self.body.dims.len()) as u32))
            .collect_vec();
        // Create the objects, but don't add anythin yet so we can rollback if an error
        // occurs.
        let mut dims = Vec::new();
        let logical_dim = if let Some(size) = size.as_constant() {
            let possible_sizes =
                tiling_factors.iter().map(|factor| size / factor).collect();
            let dim =
                Dimension::new_static(dim_ids[0], possible_sizes, Some(logical_id))?;
            dims.push(dim);
            ir::LogicalDim::new_static(logical_id, dim_ids.clone(), size)
        } else {
            let static_dims = dim_ids[1..].to_vec();
            let mut tiled_size: ir::PartialSize = size.clone().into();
            tiled_size.add_divisors(&VecSet::new(static_dims.clone()));
            dims.push(Dimension::new(dim_ids[0], tiled_size, Some(logical_id))?);
            ir::LogicalDim::new_dynamic(
                logical_id,
                dim_ids[0],
                static_dims,
                tiling_factors,
                size,
            )
        };
        for (&id, sizes) in dim_ids[1..].iter().zip_eq(possible_tile_sizes) {
            dims.push(Dimension::new_static(id, sizes, Some(logical_id))?);
        }
        // Register the new objects.
        for dim in &dims {
            if dim.possible_sizes().is_some() {
                self.body.static_dims.push(dim.id());
            }
        }
        self.body.dims.extend(dims);
        self.body.logical_dims.push(logical_dim);
        Ok((logical_id, dim_ids))
    }

    /// Specifies two dimensions must have the same size have can be used for point-to-point
    /// communication.
    pub fn map_dimensions(&mut self, dims: [ir::DimId; 2]) -> ir::DimMappingId {
        self.find_mapping(dims[0], dims[1]).unwrap_or_else(|| {
            let id = ir::DimMappingId(self.body.dim_mappings.len() as u16);
            let mapping = self.create_mapping(id, dims);
            self.body.dim_mappings.push(mapping);
            id
        })
    }

    pub(crate) fn freeze(self) -> Function {
        let mut counter = ir::Counter {
            next_mem: self.body.mem_blocks.num_blocks(),
            next_inst: self.body.insts.len(),
            next_dim: self.body.dims.len(),
            next_dim_mapping: self.body.dim_mappings.len() as u16,
        };
        let Function {
            signature,
            device,
            body:
                Body {
                    insts,
                    dims,
                    static_dims,
                    thread_dims,
                    mem_insts,
                    mut mem_blocks,
                    layouts_to_lower,
                    induction_vars,
                    logical_dims,
                    mut dim_mappings,
                    variables,
                },
        } = self;

        let mut insts = SparseVec::from_vec(
            insts
                .into_iter()
                .map(|inst| inst.map(|inst| inst.freeze(&mut counter)))
                .collect(),
        );
        let induction_vars: Vec<_> = induction_vars
            .into_iter()
            .map(|induction_var| induction_var.freeze(&mut counter))
            .collect();
        let mut dims = SparseVec::from_vec(
            dims.into_iter()
                .map(|dim| dim.map(|dim| dim.freeze()))
                .collect(),
        );

        let ir::Counter {
            next_mem,
            next_inst,
            next_dim,
            next_dim_mapping,
        } = counter;
        insts.expand_to(next_inst);
        dims.expand_to(next_dim);
        mem_blocks.expand_blocks_to(next_mem);
        dim_mappings.expand_to(next_dim_mapping as usize);

        Function {
            signature,
            device,
            body: Body {
                insts,
                dims,
                static_dims,
                thread_dims,
                mem_insts,
                mem_blocks,
                layouts_to_lower,
                induction_vars,
                logical_dims,
                dim_mappings,
                variables,
            },
        }
    }
}

impl Function {
    /// Lowers a dim map into a partially defined layout.
    pub(crate) fn lower_dim_map(
        &mut self,
        dst_inst: InstId,
        dst_operand_pos: usize,
    ) -> Result<ir::LoweredDimMap, ()> {
        // TODO(search_space): allow temporary memory generation for reduce operators.
        let (src_inst, data_type, lowered) = {
            match self.body.insts[dst_inst].operands()[dst_operand_pos] {
                Operand::Inst(src_id, t, dim_map, ir::DimMapScope::Global(lowering)) => {
                    (*src_id, *t, lowering.lower(dim_map))
                }
                Operand::Inst(_, _, _, _) => {
                    debug!(
                        "The dimension mapping {:?}.{} cannot be lowered",
                        dst_inst, dst_operand_pos
                    );
                    return Err(());
                }
                Operand::Reduce(..) => return Err(()),
                _ => panic!(),
            }
        };
        // Activate the new dimensions
        let st_dim_map = self.activate_mapped_dims(&lowered.st_dims_mapping, true);
        let ld_dim_map = self.activate_mapped_dims(&lowered.ld_dims_mapping, false);

        // Activate the temporary memory block
        self.body.mem_blocks.set_lazy_tmp(
            lowered.mem,
            data_type,
            lowered.mem_dimensions(),
        );

        // Build and activate the store instruction
        let st_operand =
            Operand::new_inst(self.inst(src_inst), st_dim_map, ir::DimMapScope::Local);
        let st = unwrap!(self.create_inst(
            lowered.store,
            Operator::TmpSt(st_operand, lowered.mem),
            lowered.store_dims().collect(),
        ));
        self.body.insts.set_lazy(lowered.store, st);

        // Build and activate the load instruction
        let ld = unwrap!(self.create_inst(
            lowered.load,
            Operator::TmpLd(data_type, lowered.mem),
            lowered.load_dims().collect(),
        ));
        self.body.insts.set_lazy(lowered.load, ld);
        self.body.insts[dst_inst].lower_dim_map(
            dst_operand_pos,
            lowered.load,
            ld_dim_map,
        );

        Ok(lowered)
    }

    /// Create dimensions with preallocated IDs and maps then to existing dimensions. `mapping`
    /// provides the pairs of old and new IDs along with the mapping ID. Returns a `DimMap`
    /// containing the mapped dimensions. The boolean controls in wich order mappings should be
    /// returned: from the old dimension to the new or the converse.
    fn activate_mapped_dims(
        &mut self,
        mappings: &[(ir::DimMappingId, [ir::DimId; 2])],
        old_to_new: bool,
    ) -> ir::dim::Map {
        let dims = mappings.iter().map(|&(mapping_id, dims)| {
            let [old_dim, new_dim] = dims;
            let dimension = Dimension::with_same_size(new_dim, &self.body.dims[old_dim]);
            if dimension.possible_sizes().is_some() {
                self.body.static_dims.push(new_dim);
            }
            self.body.dims.set_lazy(new_dim, dimension);
            // We can only create the mapping after we activate dimensions.
            let mapping = self.create_mapping(mapping_id, dims);
            self.body.dim_mappings.set_lazy(mapping_id, mapping);
            if old_to_new {
                (old_dim, new_dim)
            } else {
                (new_dim, old_dim)
            }
        });
        ir::dim::Map::new(dims)
    }

    /// Trigger to call when two dimensions are not merged.
    pub(crate) fn dim_not_merged(&mut self, lhs: ir::DimId, rhs: ir::DimId) {
        let to_lower = self.body.mem_blocks.not_merged(&self.body.dims[lhs], rhs);
        self.body.layouts_to_lower.extend(to_lower);
    }
}

impl fmt::Display for Function {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        writeln!(fmt, "Instructions:")?;
        for inst in self.body.insts.iter() {
            writeln!(fmt, "  {}", inst.display(self))?
        }

        writeln!(fmt, "Logical dimensions:")?;
        for logical_dim in &self.body.logical_dims {
            writeln!(fmt, "  {}", logical_dim)?;
        }

        writeln!(fmt, "Extra dimensions:")?;
        for dim in self.body.dims.iter().filter(|d| d.logical_dim().is_none()) {
            writeln!(fmt, "  {}", dim)?;
        }

        Ok(())
    }
}
