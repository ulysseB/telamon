//! A module for handling accesses to the device memory.
use ir::{self, InstId, Type, dim};
use utils::*;

// TODO(cleanup): move layouts into internal blocks.

/// Uniquely identifies a block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[repr(C)]
pub enum MemId {
    /// cbindgen:field-names=[id]
    Internal(u32),
    /// cbindgen:field-names=[id]
    External(u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
/// cbindgen:field-names=[id]
pub struct InternalId(pub u32);

impl Into<usize> for InternalId {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl From<InternalId> for MemId {
    fn from(id: InternalId) -> Self { MemId::Internal(id.0) }
}

/// Represents a memory block.
pub trait Block {
    /// The ID of the block.
    fn mem_id(&self) -> MemId;
    /// Returns self if it is an internal memory block.
    fn as_internal(&self) -> Option<&InternalBlock> { None }
    /// The list of instructions referencing the memory block.
    fn uses(&self) -> &[InstId];
    /// Add a use to the block.
    fn add_use(&mut self, inst: InstId);
}

/// A block of memory allocated on the device by the kernel.
#[derive(Clone)]
pub struct InternalBlock {
    id: InternalId,
    uses: Vec<InstId>,
    base_size: u32,
    mapped_dims: Vec<(ir::DimId, ir::DimId)>,
    // TODO(search_space): enable layout transformations.
    maybe_mapped: dim::Map,
}

/// A memory block allocated by the user.
#[derive(Clone)]
pub struct ExternalBlock {
    id: MemId,
    uses: Vec<InstId>,
}

impl InternalBlock {
    /// Returns the unique identifer of the memory block.
    pub fn id(&self) -> InternalId { self.id }

    /// Returns true if the layout is ready to be lowered.
    fn is_ready(&self) -> bool { self.maybe_mapped.is_empty() }

    /// Indicates if two dimensions are mapped by a temporary memory block.
    pub fn maps_dims(&self, lhs: ir::DimId, rhs: ir::DimId) -> bool {
        self.mapped_dims.contains(&(lhs, rhs)) ||
            self.maybe_mapped.iter().any(|&x| x == (lhs, rhs))
    }

    /// Returns the list of mapped dimensions.
    pub fn mapped_dims(&self) -> &[(ir::DimId, ir::DimId)] { &self.mapped_dims }

    /// Indicates if the block is privatised per block of thread.
    pub fn is_private(&self) -> bool { true }

    /// Return the base size of the block, if it is statically known.
    pub fn base_size(&self) -> u32 { self.base_size }
}

impl Block for InternalBlock {
    fn mem_id(&self) -> MemId { MemId::Internal(self.id.0) }

    fn as_internal(&self) -> Option<&InternalBlock> { Some(self) }

    fn uses(&self) -> &[InstId] { &self.uses }

    fn add_use(&mut self, inst: InstId) { self.uses.push(inst); }
}

impl Block for ExternalBlock {
    fn mem_id(&self) -> MemId { self.id }

    fn uses(&self) -> &[InstId] { &self.uses }

    fn add_use(&mut self, inst: InstId) { self.uses.push(inst); }
}

/// Holds the blocks of memory to allocate on the device.
#[derive(Clone)]
pub struct BlockMap {
    internal_blocks: ir::SparseVec<InternalId, InternalBlock>,
    external_blocks: Vec<ExternalBlock>,
    layouts: HashSet<InternalId>,
}

impl BlockMap {
    /// Creates a new `BlocksMap`.
    pub fn new(num_external: u32) -> Self {
        let external_blocks = (0..num_external).map(|id| {
            ExternalBlock { id: MemId::External(id), uses: vec![] }
        }).collect();
        BlockMap {
            internal_blocks: ir::SparseVec::new(),
            external_blocks,
            layouts: HashSet::default(),
        }
    }

    pub fn num_internal_blocks(&self) -> usize {
        self.internal_blocks.len()
    }

    /// Allocates a new `Block` with the given type and sizes. Must call not merged on
    /// the dimensions that cannot be merged upon creation.
    pub fn alloc_block(&mut self, base_size: u32, maybe_mapped: Option<ir::DimMap>)
        -> InternalId
    {
        let id = InternalId(self.internal_blocks.len() as u32);
        let block = self.create_block(id, base_size, maybe_mapped);
        self.internal_blocks.push(block);
        id
    }

    fn create_block(
        &mut self,
        id: InternalId,
        base_size: u32,
        maybe_mapped: Option<ir::DimMap>,
    ) -> InternalBlock {
        if let Some(ref dim_map) = maybe_mapped {
            assert!(!dim_map.is_empty());
            self.layouts.insert(id);
        }
        InternalBlock {
            id,
            base_size: base_size,
            uses: vec![],
            mapped_dims: vec![],
            maybe_mapped: maybe_mapped.unwrap_or_else(ir::DimMap::empty),
        }
    }

    pub fn expand_internal_blocks_to(&mut self, capacity: usize) {
        self.internal_blocks.expand_to(capacity);
    }

    /// Inserts a new temporary memory. Must be inserted before not_merged is called
    /// on dimensions.
    pub fn set_lazy_tmp<IT>(&mut self, id: InternalId, t: Type, dims: IT)
            where IT: Iterator<Item=(ir::DimId, ir::DimId)> {
        let block = self.create_block(
            id, unwrap!(t.len_byte()), Some(ir::DimMap::new(dims)));
        self.internal_blocks.set_lazy(id, block);
    }

    /// Registers a use of a memory block by an instruction.
    pub fn register_use(&mut self, mem: MemId, inst: InstId) {
        self.block_mut(mem).add_use(inst)
    }

    /// Returns a block given its Id.
    pub fn block(&self, id: MemId) -> &Block {
        match id {
            MemId::Internal(num) => &self.internal_blocks[InternalId(num)],
            MemId::External(num) => &self.external_blocks[num as usize],
        }
    }

    /// Returns a block given its Id.
    pub fn block_mut(&mut self, id: MemId) -> &mut Block {
        match id {
            MemId::Internal(num) => &mut self.internal_blocks[InternalId(num)],
            MemId::External(num) => &mut self.external_blocks[num as usize],
        }
    }

    /// Returns the internal block given its ID.
    pub fn internal_block(&self, id: InternalId) -> &InternalBlock {
        &self.internal_blocks[id]
    }

    /// Retuns the list of internal blocks.
    pub fn internal_blocks<'b>(&'b self) -> impl Iterator<Item=&'b InternalBlock> {
        self.internal_blocks.iter()
    }

    /// Returns the list of memory blocks.
    pub fn blocks<'b>(&'b self) -> impl Iterator<Item=&'b Block> {
        self.internal_blocks.iter().map(|b| b as &Block)
            .chain(self.external_blocks.iter().map(|b| b as &Block))
    }

    /// Rename a basic block. Returns the lyaouts to lower.
    pub fn merge_dims(&mut self, lhs: ir::DimId, rhs: ir::DimId) -> Vec<InternalId> {
        let mut to_lower = Vec::new();
        for block in self.internal_blocks.iter_mut() {
            if block.maybe_mapped.merge_dims(lhs, rhs) && block.is_ready() {
                to_lower.push(block.id());
            }
        }
        to_lower
    }

    /// Registers that two dimensions may not be merged. Returns a list of dimensions
    /// removed from the memory blocks and a list of layouts to lower.
    pub fn not_merged(&mut self, lhs_dim: &ir::Dimension, rhs: ir::DimId)
            -> Vec<InternalId> {
        let lhs = lhs_dim.id();
        let mut to_lower = Vec::new();
        for &id in &self.layouts {
            let mut changed = false; // Ensure we only lower once.
            let block = &mut self.internal_blocks[id];
            for pair in block.maybe_mapped.filter(|&mut (lhs2, rhs2)| {
                (lhs2 == lhs && rhs2 == rhs) || (lhs2 == rhs && rhs2 == lhs)
            }) {
                block.mapped_dims.push(pair);
                changed = true;
            }
            if changed && block.is_ready() { to_lower.push(id); }
        }
        to_lower
    }

    /// Lowers a fully defined layout. Returns the mapping of dimensions.
    pub fn lower_layout(&mut self, id: InternalId) -> Vec<(ir::DimId, ir::DimId)> {
        assert!(self.layouts.remove(&id));
        let block = &self.internal_blocks[id];
        assert!(block.is_ready());
        block.mapped_dims.clone()
    }
}
