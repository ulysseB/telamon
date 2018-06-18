//! A module for handling accesses to the device memory.
use ir::{self, InstId, Size, Type, dim};
use std::hash::{Hash, Hasher};
use utils::*;

// TODO(cleanup): move layouts into internal blocks.

/// Uniquely identifies a block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Id { Internal(u32), External(u32) }

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InternalId(pub u32);

impl From<InternalId> for Id {
    fn from(id: InternalId) -> Self { Id::Internal(id.0) }
}

/// Represents a memory block.
pub trait Block {
    /// The ID of the block.
    fn mem_id(&self) -> Id;
    /// Returns self if it is an internal memory block.
    fn as_internal(&self) -> Option<&InternalBlock> { None }
    /// The list of instructions referencing the memory block.
    fn uses(&self) -> &[InstId];
    /// Add a use to the block.
    fn add_use(&mut self, inst: InstId);
}

/// A block of memory allocated on the device by the kernel.
#[derive(Clone)]
pub struct InternalBlock<'a> {
    id: InternalId,
    uses: Vec<InstId>,
    base_size: Size<'a>,
    is_private: bool,
    mapped_dims: Vec<(ir::dim::Id, ir::dim::Id)>,
    // TODO(search_space): enable layout transformations.
    maybe_mapped: dim::Map,
}

impl<'a> PartialEq for InternalBlock<'a> {
    fn eq(&self, other: &InternalBlock) -> bool { self.id() == other.id() }
}

impl<'a> Eq for InternalBlock<'a> {}

impl<'a> Hash for InternalBlock<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) { self.id().hash(state) }
}

/// A memory block allocated by the user.
#[derive(Clone)]
pub struct ExternalBlock {
    id: Id,
    uses: Vec<InstId>,
}

impl<'a> InternalBlock<'a> {
    /// Returns the unique identifer of the memory block.
    pub fn id(&self) -> InternalId { self.id }

    /// Returns true if the layout is ready to be lowered.
    fn is_ready(&self) -> bool { self.maybe_mapped.is_empty() }

    /// Indicates if two dimensions are mapped by a temporary memory block.
    pub fn maps_dims(&self, lhs: ir::dim::Id, rhs: ir::dim::Id) -> bool {
        self.mapped_dims.contains(&(lhs, rhs)) ||
            self.maybe_mapped.iter().any(|&x| x == (lhs, rhs))
    }

    /// Returns the list of mapped dimensions.
    pub fn mapped_dims(&self) -> &[(ir::dim::Id, ir::dim::Id)] { &self.mapped_dims }

    /// Indicates if the block is privatised per block of thread.
    pub fn is_private(&self) -> bool { self.is_private }

    /// Return the base size of the block, if it is statically known.
    pub fn base_size(&self) -> Option<u32> { self.base_size.as_int() }
}

impl<'a> Block for InternalBlock<'a> {
    fn mem_id(&self) -> Id { Id::Internal(self.id.0) }

    fn as_internal(&self) -> Option<&InternalBlock> { Some(self) }

    fn uses(&self) -> &[InstId] { &self.uses }

    fn add_use(&mut self, inst: InstId) { self.uses.push(inst); }
}

impl Block for ExternalBlock {
    fn mem_id(&self) -> Id { self.id }

    fn uses(&self) -> &[InstId] { &self.uses }

    fn add_use(&mut self, inst: InstId) { self.uses.push(inst); }
}

/// Holds the blocks of memory to allocate on the device.
#[derive(Clone)]
pub struct BlockMap<'a> {
    internal_blocks: Vec<InternalBlock<'a>>,
    external_blocks: Vec<ExternalBlock>,
    layouts: HashSet<InternalId>,
}

impl<'a> BlockMap<'a> {
    /// Creates a new `BlocksMap`.
    pub fn new(num_external: u32) -> BlockMap<'a> {
        let external_blocks = (0..num_external).map(|id| {
            ExternalBlock { id: Id::External(id), uses: vec![] }
        }).collect();
        BlockMap {
            internal_blocks: vec![],
            external_blocks,
            layouts: HashSet::default(),
        }
    }

    /// Allocates a new `Block` with the given type and sizes. Must call not merged on
    /// the dimensions that cannot be merged upon creation.
    pub fn alloc_block(&mut self, base_size: Size<'a>, private: bool,
                       maybe_mapped: Option<ir::DimMap>) -> InternalId {
        let id = InternalId(self.internal_blocks.len() as u32);
        if let Some(ref dim_map) = maybe_mapped {
            assert!(!dim_map.is_empty());
            self.layouts.insert(id);
        }
        let block = InternalBlock {
            id,
            base_size: base_size.clone(),
            is_private: private,
            uses: vec![],
            mapped_dims: vec![],
            maybe_mapped: maybe_mapped.unwrap_or_else(ir::DimMap::empty),
        };
        self.internal_blocks.push(block);
        id
    }

    /// Inserts a new temporary memory. Must be inserted before not_merged is called
    /// on dimensions.
    pub fn new_tmp<IT>(&mut self, t: Type, dims: IT) -> InternalId
            where IT: Iterator<Item=(dim::Id, dim::Id)> {
        let base_size = ir::Size::new(unwrap!(t.len_byte()), vec![], 1);
        self.alloc_block(base_size, true, Some(ir::DimMap::new(dims)))
    }

    /// Registers a use of a memory block by an instruction.
    pub fn register_use(&mut self, mem: Id, inst: InstId) {
        self.block_mut(mem).add_use(inst)
    }

    /// Returns a block given its Id.
    pub fn block(&self, id: Id) -> &Block {
        match id {
            Id::Internal(num) => &self.internal_blocks[num as usize],
            Id::External(num) => &self.external_blocks[num as usize],
        }
    }

    /// Returns a block given its Id.
    pub fn block_mut(&mut self, id: Id) -> &mut Block {
        match id {
            Id::Internal(num) => &mut self.internal_blocks[num as usize],
            Id::External(num) => &mut self.external_blocks[num as usize],
        }
    }

    /// Returns the internal block given its ID.
    pub fn internal_block(&self, id: InternalId) -> &InternalBlock {
        &self.internal_blocks[id.0 as usize]
    }

    /// Retuns the list of internal blocks.
    pub fn internal_blocks<'b>(&'b self) -> impl Iterator<Item=&'b InternalBlock<'a>> {
        self.internal_blocks.iter()
    }

    /// Returns the list of memory blocks.
    pub fn blocks<'b>(&'b self) -> impl Iterator<Item=&'b Block> {
        self.internal_blocks.iter().map(|b| b as &Block)
            .chain(self.external_blocks.iter().map(|b| b as &Block))
    }

    /// Rename a basic block. Returns the lyaouts to lower.
    pub fn merge_dims(&mut self, lhs: dim::Id, rhs: dim::Id) -> Vec<InternalId> {
        let mut to_lower = Vec::new();
        for block in &mut self.internal_blocks {
            if block.maybe_mapped.merge_dims(lhs, rhs) && block.is_ready() {
                to_lower.push(block.id());
            }
        }
        to_lower
    }

    /// Registers that two dimensions may not be merged. Returns a list of dimensions
    /// removed from the memory blocks and a list of layouts to lower.
    pub fn not_merged(&mut self, lhs_dim: &ir::Dimension<'a>, rhs: dim::Id)
            -> Vec<InternalId> {
        let lhs = lhs_dim.id();
        let mut to_lower = Vec::new();
        for &id in &self.layouts {
            let mut changed = false; // Ensure we only lower once.
            let block = &mut self.internal_blocks[id.0 as usize];
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
    pub fn lower_layout(&mut self, id: InternalId) -> Vec<(dim::Id, dim::Id)> {
        assert!(self.layouts.remove(&id));
        let block = &self.internal_blocks[id.0 as usize];
        assert!(block.is_ready());
        block.mapped_dims.clone()
    }
}
