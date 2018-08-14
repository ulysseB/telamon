//! Provides a representation of functions.
use device::Device;
use ir::{self, BasicBlock, BBId, Dimension, InstId, Instruction, Operator};
use ir::{AccessPattern, Operand, Size, Type, dim, mem, SparseVec};
use ir::mem::Block;
use itertools::Itertools;
use std;
use utils::*;

/// Represents an argument of a function.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Parameter {
    /// The name of the `Parameter`
    pub name: String,
    /// The type of the `Parameter`.
    pub t: Type,
}

/// Holds the signature of a function.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Signature {
    /// Mame of the function.
    pub name: String,
    /// Arguments of the function.
    pub params: Vec<Parameter>,
    /// The number of external memory blocks.
    pub mem_blocks: u32,
}

impl Signature {
    /// Creates a new signature without any parameter.
    pub fn new(name: String) -> Self {
        Signature { name, params: vec![], mem_blocks: 0 }
    }

    /// Adds a scalar parameter.
    pub fn add_scalar(&mut self, name: String, t: ir::Type) {
        self.params.push(Parameter { name, t });
    }

    /// Adds a parameter with the given name and type to the signature.
    pub fn add_array(&mut self, name: String) -> ir::MemId {
        let id = ir::MemId::External(self.mem_blocks);
        self.mem_blocks += 1;
        self.params.push(Parameter { name, t: ir::Type::PtrTo(id) });
        id
    }
}

/// Describes a function and the set of its possible implementation.
#[derive(Clone)]
pub struct Function<'a, L = ir::LoweringMap> {
    signature: &'a Signature,
    device: &'a Device,
    insts: SparseVec<ir::InstId, Instruction<'a, L>>,
    dims: SparseVec<ir::DimId, Dimension<'a>>,
    static_dims: Vec<ir::DimId>,
    thread_dims: VecSet<ir::DimId>,
    mem_insts: Vec<ir::InstId>,
    mem_blocks: mem::BlockMap,
    layouts_to_lower: Vec<ir::mem::InternalId>,
    induction_vars: Vec<ir::InductionVar<'a, L>>,
}

impl<'a, L> Function<'a, L> {
    /// Creates a new function.
    pub fn new(signature: &'a Signature, device: &'a Device) -> Self {
        let mem_blocks = mem::BlockMap::new(signature.mem_blocks);
        Function {
            signature,
            device,
            insts: SparseVec::new(),
            mem_insts: vec![],
            dims: SparseVec::new(),
            static_dims: vec![],
            thread_dims: VecSet::default(),
            mem_blocks,
            layouts_to_lower: Vec::new(),
            induction_vars: Vec::new(),
        }
    }

    /// Returns the function signature.
    pub fn signature(&self) -> &'a Signature { self.signature }

    /// Returns the device the function is compiled for.
    pub fn device(&self) -> &'a Device { self.device }

    /// Creates a new instruction (with given ID) without adding it to
    /// the `insts` vector. Used as an internal helper for when either
    /// adding a new instruction (`add_inst`) or filling an existing
    /// hole in the instructions vector.
    fn create_inst(
        &mut self,
        id: InstId,
        op: Operator<'a, L>,
        iter_dims: HashSet<ir::DimId>,
    ) -> Result<ir::Instruction<'a, L>, ir::Error> {
        let inst = ir::Instruction::new(op, id, iter_dims, self.device)?;
        // Register the instruction in iteration dimensions.
        for &dim in inst.iteration_dims() { self.dim_mut(dim).add_iterated(id.into()); }
        // Register the memory blocks used.
        if let Some(mem_id) = inst.operator().mem_used() {
            self.mem_insts.push(id);
            self.mem_blocks.register_use(mem_id, id);
        }
        Ok(inst)
    }

    /// Adds an induction variable.
    pub fn add_ind_var(&mut self, ind_var: ir::InductionVar<'a, L>) -> ir::IndVarId {
        let id = ir::IndVarId(self.induction_vars.len() as u32);
        self.induction_vars.push(ind_var);
        id
    }

    /// Returns the list of instructions of the function.
    pub fn insts<'b>(&'b self) -> impl Iterator<Item=&'b Instruction<'a, L>> {
        self.insts.iter()
    }

    /// Returns the list of dimensions of the function.
    pub fn dims<'b>(&'b self) -> impl Iterator<Item=&'b Dimension<'a>> + Clone {
        self.dims.iter()
    }

    /// Returns the list of stastic dimensions in the function.
    pub fn static_dims<'b>(&'b self) -> impl Iterator<Item=&'b Dimension<'a>> {
        self.static_dims.iter().map(move |&id| self.dim(id))
    }

    /// Returns the list of thread dimensions.
    pub fn thread_dims(&self) -> impl Iterator<Item=&Dimension<'a>> {
        self.thread_dims.iter().map(move |&d| self.dim(d))
    }

    /// Returns an instruction given its id.
    pub fn inst(&self, id: InstId) -> &Instruction<'a, L> { &self.insts[id] }

    /// Returns a mutable reference to an instruction given its id.
    fn inst_mut(&mut self, id: InstId) -> &mut Instruction<'a, L> {
        &mut self.insts[id]
    }

    /// Retuns a dimension given its id.
    pub fn dim(&self, id: ir::DimId) -> &Dimension<'a> {
        &self.dims[id]
    }

    /// Returns a mutable reference to a dimension given its ID.
    fn dim_mut(&mut self, id: ir::DimId) -> &mut Dimension<'a> {
        &mut self.dims[id]
    }

    /// Returns the list of memory blocks. The block with id `i` is in i-th position.
    pub fn mem_blocks<'b>(&'b self) -> impl Iterator<Item=&'b mem::Block> {
        self.mem_blocks.blocks()
    }

    /// Iterates over memory instructions.
    pub fn mem_insts<'b>(&'b self) -> impl Iterator<Item=&'b Instruction<'a, L>> + 'b {
        self.mem_insts.iter().map(move |&id| self.inst(id))
    }

    /// Returns the internal memory blocks.
    pub fn internal_mem_blocks<'b>(&'b self)
            -> impl Iterator<Item=&'b mem::InternalBlock> {
        self.mem_blocks.internal_blocks()
    }

    /// Returns a memory block given its id.
    pub fn mem_block(&self, id: ir::MemId) -> &mem::Block { self.mem_blocks.block(id) }

    /// Returns an internal memory block given its id.
    pub fn internal_mem_block(&self, id: mem::InternalId) -> &mem::InternalBlock {
        self.mem_blocks.internal_block(id)
    }

    /// Retrieves an induction variable given its Id.
    pub fn induction_var(&self, id: ir::IndVarId) -> &ir::InductionVar<'_, L> {
        &self.induction_vars[id.0 as usize]
    }

    /// Iterates over induction variables.
    pub fn induction_vars<'b>(&'b self)
            -> impl Iterator<Item=(ir::IndVarId, &'b ir::InductionVar<'a, L>)> {
        self.induction_vars.iter().enumerate().map(|(id, v)| (ir::IndVarId(id as u32), v))
    }

    /// Sets a dimension as an iteration dimension for an instruction. Indicates if the
    /// iteration dimension was not aleady present in the set.
    pub fn set_iteration_dim(&mut self, inst: ir::InstId, dim: ir::DimId) -> bool {
        if self.inst_mut(inst).add_iteration_dimension(dim) {
            self.dim_mut(dim).add_iterated(inst);
            true
        } else { false }
    }

    /// Adds a thread dimension. Indicates if the the dimension was not already present
    /// in the set.
    pub fn add_thread_dim(&mut self, dim: ir::DimId) -> bool {
        self.dim_mut(dim).set_thread_dim();
        self.thread_dims.insert(dim)
    }

    /// Trigger to call when two dimensions are merged.
    // TODO(cleanup): externalize in the search space the merging of dimensions in dim
    // maps.
    pub(crate) fn merge(&mut self, src: ir::DimId, dst: ir::DimId) {
        for inst in self.insts.iter_mut() { inst.merge_dims(src, dst); }
        for var in &mut self.induction_vars { var.merge_dims(src, dst); }
        self.layouts_to_lower.extend(self.mem_blocks.merge_dims(src, dst));
    }

    /// Lowers a layout into conventional memory accesses.
    pub(crate) fn lower_layout(&mut self, id: mem::InternalId, st_dims: Vec<ir::DimId>,
                               ld_dims: Vec<ir::DimId>)
    where L: Clone {
        let pos = unwrap!(self.layouts_to_lower.iter().position(|&x| x == id));
        self.layouts_to_lower.swap_remove(pos);
        self.mem_blocks.lower_layout(id);
        let (st_index, st_pattern) = self.gen_internal_index(id, st_dims);
        let (ld_index, ld_pattern) = self.gen_internal_index(id, ld_dims);
        for &mem_use in self.mem_blocks.internal_block(id).uses() {
            self.insts[mem_use].lower_layout(ld_index.clone(),
                ld_pattern.clone(), st_index.clone(), st_pattern.clone());
        }
    }

    /// Generates an operand repesenting a pointer to a cell of a memory block.
    fn gen_internal_index(&mut self, id: mem::InternalId, dims: Vec<ir::DimId>)
            -> (Operand<'a, L>, AccessPattern<'a>) {
        let ty_len = self.mem_blocks.internal_block(id).base_size();
        self.gen_index(id.into(), ty_len, Operand::Addr(id), dims)
    }

    /// Generates an access pattern and the corresponding induction variable to access a
    /// memory block.
    fn gen_index(&mut self, mem: ir::MemId, base_incr: u32, base_addr: Operand<'a, L>,
                     dims: Vec<ir::DimId>) -> (Operand<'a, L>, AccessPattern<'a>) {
        let var_type = base_addr.t();
        let base_size = ir::Size::new(base_incr, vec![], 1);
        let increments = dims.iter().rev().scan(base_size, |size, &dim| {
            let old_size = size.clone();
            *size *= self.dim(dim).size();
            Some((dim, old_size))
        }).collect_vec();
        let pattern = ir::AccessPattern::Tensor {
            mem_id: mem,
            dims: increments.iter().cloned().collect(),
        };
        let ind_var = unwrap!(ir::InductionVar::new(increments, base_addr));
        let ind_var = self.add_ind_var(ind_var);
        let addr = ir::Operand::InductionVar(ind_var, var_type);
        (addr, pattern)
    }

    /// Trigger to call when two dimensions are not merged.
    pub(crate) fn dim_not_merged(&mut self, lhs: ir::DimId, rhs: ir::DimId) {
        let to_lower = self.mem_blocks.not_merged(&self.dims[lhs], rhs);
        self.layouts_to_lower.extend(to_lower);
    }

    /// Returns the list of layouts to lower.
    pub(crate) fn layouts_to_lower(&self) -> &[ir::mem::InternalId] { &self.layouts_to_lower }
}

impl<'a> Function<'a, ()> {
    /// Adds an instruction to the function.
    pub fn add_inst(&mut self, op: Operator<'a, ()>, iter_dims: HashSet<ir::DimId>)
        -> Result<InstId, ir::Error>
    {
        let id = ir::InstId(self.insts.len() as u32);
        let inst = self.create_inst(id, op, iter_dims)?;
        self.insts.push(inst);
        Ok(id)
    }

    /// Creates a new dimension.
    pub fn add_dim(&mut self, size: Size<'a>) -> Result<ir::DimId, ir::Error> {
        let id = ir::DimId(self.dims.len() as u32);
        let dim = Dimension::new(size, id)?;
        if dim.possible_sizes().is_some() { self.static_dims.push(id); }
        self.dims.push(dim);
        Ok(id)
    }

    /// Allocates a new memory block.
    pub fn add_mem_block(&mut self, size: u32) -> mem::InternalId {
        self.mem_blocks.alloc_block(size, None)
    }

    pub(crate) fn freeze(self) -> Function<'a> {
        let mut counter = ir::Counter {
            next_mem: self.mem_blocks.num_internal_blocks(),
            next_inst: self.insts.len(),
            next_dim: self.dims.len(),
        };
        let Function {
            signature,
            device,
            insts,
            mut dims,
            static_dims,
            thread_dims,
            mem_insts,
            mut mem_blocks,
            layouts_to_lower,
            induction_vars
        } = self;

        let mut insts =
            SparseVec::from_vec(
                insts.into_iter().map(|inst| {
                    inst.map(|inst| inst.freeze(&mut counter))
                }).collect());
        let induction_vars: Vec<_> =
            induction_vars.into_iter().map(|induction_var| {
                induction_var.freeze(&mut counter)
            }).collect();

        let ir::Counter { next_mem, next_inst, next_dim } = counter;
        insts.expand_to(next_inst);
        dims.expand_to(next_dim);
        mem_blocks.expand_internal_blocks_to(next_mem);

        Function {
            signature,
            device,
            insts,
            dims,
            static_dims,
            thread_dims,
            mem_insts,
            mem_blocks,
            layouts_to_lower,
            induction_vars,
        }
    }
}

impl<'a> Function<'a> {
    /// Returns a `BasicBlock` given its id.
    pub fn block(&self, id: BBId) -> &BasicBlock<'a> {
        match id {
            BBId::Inst(id) => &self.insts[id],
            BBId::Dim(id) => self.dim(id),
        }
    }

    /// Lists all `BasicBlock`s.
    pub fn blocks<'b>(&'b self) -> impl Iterator<Item=&'b BasicBlock<'a>> {
        self.insts.iter().map(|x| x as _).chain(self.dims().map(|x| x as _))
    }

    /// Lowers a dim map into a partially defined layout.
    pub(crate) fn lower_dim_map(&mut self, dst_inst: InstId, dst_operand_pos: usize)
        -> Result<ir::LoweredDimMap, ()> {
        // TODO(search_space): allow temporary memory generation for reduce operators.
        let (src_inst, data_type, dims, lowered) = {
            match *self.insts[dst_inst].operands()[dst_operand_pos] {
                Operand::Inst(
                    ref src_id, t, ref dim_map, ir::DimMapScope::Global(ref lowering)) => {
                    let lowered = lowering.lower(dim_map);
                    (*src_id, t, dim_map.iter().cloned().unzip(), lowered)
                },
                Operand::Inst(_, _, _, _) => {
                    debug!("The dimension mapping {:?}.{} cannot be lowered",
                           dst_inst, dst_operand_pos);
                    return Err(())
                },
                Operand::Reduce(..) => return Err(()),
                _ => panic!(),
            }
        };
        // Flattens the dimensions
        let (src_dims, dst_dims): (Vec<_>, Vec<_>) = dims;
        let (st_dims, ld_dims): (Vec<_>, Vec<_>) =
            lowered.dimensions.iter().cloned().unzip();

        // Activate the new dimensions
        for (&src_dim, &st_dim) in src_dims.iter().zip_eq(&st_dims) {
            let dimension = Dimension::with_same_size(
                st_dim, &self.dims[src_dim]);
            if dimension.possible_sizes().is_some() {
                self.static_dims.push(st_dim);
            }
            self.dims.set_lazy(st_dim, dimension);
        }

        for (&dst_dim, &ld_dim) in dst_dims.iter().zip_eq(&ld_dims) {
            let dimension = Dimension::with_same_size(
                ld_dim, &self.dims[dst_dim]);
            if dimension.possible_sizes().is_some() {
                self.static_dims.push(ld_dim);
            }
            self.dims.set_lazy(ld_dim, dimension);
        }

        // Activate the temporary memory block
        self.mem_blocks.set_lazy_tmp(
            lowered.mem, data_type, lowered.dimensions.iter().cloned());

        // Build and activate the store instruction
        let st_dim_map = dim::Map::new(src_dims.iter().zip_eq(&st_dims).map(clone_pair));
        let st_operand =  Operand::new_inst(
            self.inst(src_inst), st_dim_map, ir::DimMapScope::Local);
        let st_dim_set = st_dims.iter().cloned().collect();
        let st = unwrap!(self.create_inst(
            lowered.store, Operator::TmpSt(st_operand, lowered.mem.into()), st_dim_set));
        self.insts.set_lazy(lowered.store, st);

        // Build and activate the load instruction
        let ld_dim_map = dim::Map::new(ld_dims.iter().zip_eq(&dst_dims).map(clone_pair));
        let ld_dim_set = ld_dims.iter().cloned().collect();
        let ld = unwrap!(self.create_inst(
            lowered.load, Operator::TmpLd(data_type, lowered.mem.into()), ld_dim_set));
        self.insts.set_lazy(lowered.load, ld);
        self.insts[dst_inst].lower_dim_map(
            dst_operand_pos, lowered.load, ld_dim_map);

        Ok(lowered)
    }
}

impl<'a> std::ops::Deref for Function<'a> {
    type Target = Signature;

    fn deref(&self) -> &Self::Target { self.signature }
}
