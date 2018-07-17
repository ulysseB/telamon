//! Provides a representation of functions.
use device::Device;
use ir::{self, BasicBlock, BBId, Dimension, InstId, Instruction, Operator};
use ir::{AccessPattern, Operand, Size, Type, dim, mem};
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
    pub fn add_array(&mut self, name: String) -> ir::mem::Id {
        let id = mem::Id::External(self.mem_blocks);
        self.mem_blocks += 1;
        self.params.push(Parameter { name, t: ir::Type::PtrTo(id) });
        id
    }
}

/// Describes a function and the set of its possible implementation.
#[derive(Clone)]
pub struct Function<'a> {
    signature: &'a Signature,
    device: &'a Device,
    insts: Vec<Instruction<'a>>,
    dims: Vec<Dimension<'a>>,
    thread_dims: VecSet<ir::dim::Id>,
    mem_insts: Vec<ir::InstId>,
    mem_blocks: mem::BlockMap<'a>,
    layouts_to_lower: Vec<ir::mem::InternalId>,
    induction_vars: Vec<ir::InductionVar<'a>>,
}

impl<'a> Function<'a> {
    /// Creates a new function.
    pub fn new(signature: &'a Signature, device: &'a Device) -> Self {
        let mem_blocks = mem::BlockMap::new(signature.mem_blocks);
        Function {
            signature,
            device,
            insts: vec![],
            mem_insts: vec![],
            dims: vec![],
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

    /// Adds an instruction to the function.
    pub fn add_inst(&mut self, op: Operator<'a>, iter_dims: HashSet<dim::Id>)
        -> Result<InstId, ir::Error>
    {
        let id = ir::InstId(self.insts.len() as u32);
        let inst = ir::Instruction::new(op, id, iter_dims, self.device)?;
        // Register the instruction in iteration dimensions.
        for &dim in inst.iteration_dims() { self.dim_mut(dim).add_iterated(id.into()); }
        // Register the memory blocks used.
        if let Some(mem_id) = inst.operator().mem_used() {
            self.mem_insts.push(id);
            self.mem_blocks.register_use(mem_id, id);
        }
        self.insts.push(inst);
        Ok(id)
    }

    /// Creates a new dimension.
    pub fn add_dim(&mut self, size: Size<'a>) -> Result<dim::Id, ir::Error> {
        let id = dim::Id(self.dims.len() as u32);
        self.dims.push(Dimension::new(size, id)?);
        Ok(id)
    }

    /// Allocates a new memory block.
    pub fn add_mem_block(&mut self, size: Size<'a>, private: bool) -> mem::InternalId {
        self.mem_blocks.alloc_block(size, private, None)
    }

    /// Adds an induction variable.
    pub fn add_ind_var(&mut self, ind_var: ir::InductionVar<'a>) -> ir::IndVarId {
        let id = ir::IndVarId(self.induction_vars.len() as u32);
        self.induction_vars.push(ind_var);
        id
    }

    /// Returns the list of instructions of the function.
    pub fn insts<'b>(&'b self) -> impl Iterator<Item=&'b Instruction<'a>> {
        self.insts.iter()
    }

    /// Returns the list of dimensions of the function.
    pub fn dims<'b>(&'b self) -> impl Iterator<Item=&'b Dimension<'a>> + Clone {
        self.dims.iter()
    }

    /// Lists all `BasicBlock`s.
    pub fn blocks<'b>(&'b self) -> impl Iterator<Item=&'b BasicBlock<'a>> {
        self.insts.iter().map(|x| x as _).chain(self.dims.iter().map(|x| x as _))
    }

    /// Returns the list of thread dimensions.
    pub fn thread_dims(&self) -> impl Iterator<Item=&Dimension<'a>> {
        self.thread_dims.iter().map(move |&d| self.dim(d))
    }

    /// Returns an instruction given its id.
    pub fn inst(&self, id: InstId) -> &Instruction<'a> { &self.insts[id.0 as usize] }

    /// Returns a mutable reference to an instruction given its id.
    fn inst_mut(&mut self, id: InstId) -> &mut Instruction<'a> {
        &mut self.insts[id.0 as usize]
    }

    /// Retuns a dimension given its id.
    pub fn dim(&self, id: dim::Id) -> &Dimension<'a> { &self.dims[id.0 as usize] }

    /// Returns a mutable reference to a dimension given its ID.
    fn dim_mut(&mut self, id: dim::Id) -> &mut Dimension<'a> {
        &mut self.dims[id.0 as usize]
    }

    /// Returns a `BasicBlock` given its id.
    pub fn block(&self, id: BBId) -> &BasicBlock<'a> {
        match id {
            BBId::Inst(id) => &self.insts[id.0 as usize],
            BBId::Dim(id) => self.dim(id),
        }
    }

    /// Returns the list of memory blocks. The block with id `i` is in i-th position.
    pub fn mem_blocks<'b>(&'b self) -> impl Iterator<Item=&'b mem::Block> {
        self.mem_blocks.blocks()
    }

    /// Iterates over memory instructions.
    pub fn mem_insts<'b>(&'b self) -> impl Iterator<Item=&'b Instruction<'a>> + 'b {
        self.mem_insts.iter().map(move |&id| self.inst(id))
    }

    /// Returns the internal memory blocks.
    pub fn internal_mem_blocks<'b>(&'b self)
            -> impl Iterator<Item=&'b mem::InternalBlock<'a>> {
        self.mem_blocks.internal_blocks()
    }

    /// Returns a memory block given its id.
    pub fn mem_block(&self, id: mem::Id) -> &mem::Block { self.mem_blocks.block(id) }

    /// Returns an internal memory block given its id.
    pub fn internal_mem_block(&self, id: mem::InternalId) -> &mem::InternalBlock {
        self.mem_blocks.internal_block(id)
    }

    /// Retrieves an induction variable given its Id.
    pub fn induction_var(&self, id: ir::IndVarId) -> &ir::InductionVar {
        &self.induction_vars[id.0 as usize]
    }

    /// Iterates over induction variables.
    pub fn induction_vars<'b>(&'b self)
            -> impl Iterator<Item=(ir::IndVarId, &'b ir::InductionVar<'a>)> {
        self.induction_vars.iter().enumerate().map(|(id, v)| (ir::IndVarId(id as u32), v))
    }

    /// Sets a dimension as an iteration dimension for an instruction. Indicates if the
    /// iteration dimension was not aleady present in the set.
    pub fn set_iteration_dim(&mut self, inst: ir::InstId, dim: ir::dim::Id) -> bool {
        if self.inst_mut(inst).add_iteration_dimension(dim) {
            self.dim_mut(dim).add_iterated(inst);
            true
        } else { false }
    }

    /// Adds a thread dimension. Indicates if the the dimension was not already present
    /// in the set.
    pub fn add_thread_dim(&mut self, dim: ir::dim::Id) -> bool {
        self.dim_mut(dim).set_thread_dim();
        self.thread_dims.insert(dim)
    }

    /// Trigger to call when two dimensions are merged.
    // TODO(cleanup): externalize in the search space the merging of dimensions in dim
    // maps.
    pub(crate) fn merge(&mut self, src: ir::dim::Id, dst: ir::dim::Id) {
        for inst in &mut self.insts { inst.merge_dims(src, dst); }
        for var in &mut self.induction_vars { var.merge_dims(src, dst); }
        self.layouts_to_lower.extend(self.mem_blocks.merge_dims(src, dst));
    }

    /// Lowers a layout into conventional memory accesses.
    pub(crate) fn lower_layout(&mut self, id: mem::InternalId, st_dims: Vec<ir::dim::Id>,
                        ld_dims: Vec<ir::dim::Id>) {
        let pos = unwrap!(self.layouts_to_lower.iter().position(|&x| x == id));
        self.layouts_to_lower.swap_remove(pos);
        self.mem_blocks.lower_layout(id);
        let (st_index, st_pattern) = self.gen_internal_index(id, st_dims);
        let (ld_index, ld_pattern) = self.gen_internal_index(id, ld_dims);
        for &mem_use in self.mem_blocks.internal_block(id).uses() {
            self.insts[mem_use.0 as usize].lower_layout(ld_index.clone(),
                ld_pattern.clone(), st_index.clone(), st_pattern.clone());
        }
    }

    /// Generates an operand repesenting a pointer to a cell of a memory block.
    fn gen_internal_index(&mut self, id: mem::InternalId, dims: Vec<dim::Id>)
            -> (Operand<'a>, AccessPattern<'a>) {
        let ty_len = unwrap!(self.mem_blocks.internal_block(id).base_size());
        self.gen_index(id.into(), ty_len, Operand::Addr(id), dims)
    }

    /// Generates an access pattern and the corresponding induction variable to access a
    /// memory block.
    fn gen_index(&mut self, mem: mem::Id, base_incr: u32, base_addr: Operand<'a>,
                     dims: Vec<dim::Id>) -> (Operand<'a>, AccessPattern<'a>) {
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

    /// Lowers a dim map into a partially defined layout.
    pub(crate) fn lower_dim_map(&mut self, dst_inst: InstId, dst_operand_pos: usize)
        -> Result<ir::LoweredDimMap, ()> {
        // TODO(search_space): allow temporary memory generation for reduce operators.
        let (src_inst, data_type, src_dims, dst_dims): (_, _, Vec<_>, Vec<_>) = {
            match *self.inst(dst_inst).operands()[dst_operand_pos] {
                Operand::Inst(ref src_id, t, ref dim_map, ir::DimMapScope::Global) => {
                    let (src_dims, dst_dims) = dim_map.iter().cloned().unzip();
                    (*src_id, t, src_dims, dst_dims)
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
        let st_dims = self.spawn_mapped_dims(&src_dims);
        let ld_dims = self.spawn_mapped_dims(&dst_dims);
        // Build the temporary memory block.
        let dims = st_dims.iter().cloned().zip_eq(ld_dims.iter().cloned()).collect_vec();
        let tmp_mem = self.mem_blocks.new_tmp(data_type, dims.iter().cloned());
        // Build the store.
        let st_dim_map = dim::Map::new(src_dims.iter().zip_eq(&st_dims).map(clone_pair));
        let st_operand =  Operand::new_inst(
            self.inst(src_inst), st_dim_map, ir::DimMapScope::Local);
        let st_dim_set = st_dims.iter().cloned().collect();
        let st = self.add_inst(Operator::TmpSt(st_operand, tmp_mem.into()), st_dim_set);
        let st = unwrap!(st);
        // Build the load
        let ld_dim_map = dim::Map::new(ld_dims.iter().zip_eq(&dst_dims).map(clone_pair));
        let ld_dim_set = ld_dims.iter().cloned().collect();
        let ld = self.add_inst(Operator::TmpLd(data_type, tmp_mem.into()), ld_dim_set);
        let ld = unwrap!(ld);
        self.insts[dst_inst.0 as usize].lower_dim_map(dst_operand_pos, ld, ld_dim_map);
        Ok(ir::LoweredDimMap { mem: tmp_mem, store: st, load: ld, dimensions: dims })
    }

    /// Adds multiple dimensions at once, using the sizes of the given dimensions.
    fn spawn_mapped_dims(&mut self, old_dims: &[dim::Id]) -> Vec<dim::Id> {
        old_dims.iter().map(|&old_dim| {
            let size = self.dim(old_dim).size().clone();
            unwrap!(self.add_dim(size))
        }).collect()
    }

    /// Trigger to call when two dimensions are not merged.
    pub(crate) fn dim_not_merged(&mut self, lhs: dim::Id, rhs: dim::Id) {
        let to_lower = self.mem_blocks.not_merged(&self.dims[lhs.0 as usize], rhs);
        self.layouts_to_lower.extend(to_lower);
    }

    /// Returns the list of layouts to lower.
    pub(crate) fn layouts_to_lower(&self) -> &[ir::mem::InternalId] { &self.layouts_to_lower }
}

impl<'a> std::ops::Deref for Function<'a> {
    type Target = Signature;

    fn deref(&self) -> &Self::Target { self.signature }
}
