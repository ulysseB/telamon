//! A module for handling accesses to the device memory.
use ir::{self, dim, InstId, Type};
use utils::*;

// TODO(cleanup): move layouts into internal blocks.

/// Uniquely identifies a block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[repr(C)]
/// cbindgen:field-names=[id]
pub struct MemId(pub u32);

impl From<MemId> for usize {
    fn from(id: MemId) -> usize {
        id.0 as usize
    }
}

/// A block of memory allocated on the device by the kernel.
#[derive(Clone, Debug)]
pub struct Block {
    id: MemId,
    uses: Vec<InstId>,
    elements_type: ir::Type,
    base_size: u32,
    // TODO(variables): use variables instead of passing layout dimensions here.
    mapped_dims: Vec<(ir::DimId, ir::DimId)>,
    maybe_mapped: dim::Map,
    layout_dims: HashMap<ir::DimId, ir::LayoutDimId>,
    // This store the temporary store and load instructions before the layout is fixed.
    // TODO(variables): removed once we can use variables for partially specified layouts.
    tmp_st_ld: Option<(ir::InstId, ir::InstId)>,
}

impl Block {
    /// The ID of the block.
    pub fn mem_id(&self) -> MemId {
        self.id
    }

    /// Returns true if the layout is ready to be lowered.
    fn is_ready(&self) -> bool {
        self.maybe_mapped.is_empty()
    }

    /// Indicates if two dimensions are mapped by a temporary memory block.
    pub fn maps_dims(&self, lhs: ir::DimId, rhs: ir::DimId) -> bool {
        self.mapped_dims.contains(&(lhs, rhs))
            || self.maybe_mapped.iter().any(|&x| x == (lhs, rhs))
    }

    /// Returns the list of mapped dimensions.
    pub fn mapped_dims(&self) -> &[(ir::DimId, ir::DimId)] {
        &self.mapped_dims
    }

    /// Indicates if the block is privatised per block of thread.
    pub fn is_private(&self) -> bool {
        true
    }

    /// Return the base size of the block, if it is statically known.
    pub fn base_size(&self) -> u32 {
        self.base_size * unwrap!(self.elements_type.len_byte())
    }

    /// Indicates the type of the memory block elements.
    pub fn elements_type(&self) -> ir::Type {
        self.elements_type
    }

    /// The list of instructions referencing the memory block.
    pub fn uses(&self) -> &[InstId] {
        &self.uses
    }

    /// Add a use to the block.
    pub fn add_use(&mut self, inst: InstId) {
        self.uses.push(inst);
    }
}

/// Holds the blocks of memory to allocate on the device.
#[derive(Clone, Default)]
pub struct BlockMap {
    blocks: ir::SparseVec<MemId, Block>,
    layouts: HashSet<MemId>,
}

impl BlockMap {
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Allocates a new `Block` with the given type and sizes.
    pub fn alloc_block(&mut self, t: ir::Type, len: u32) -> MemId {
        let id = MemId(self.blocks.len() as u32);
        let block = self.create_block(id, t, len, None, Default::default(), None);
        self.blocks.push(block);
        id
    }

    fn create_block(
        &mut self,
        id: MemId,
        elements_type: ir::Type,
        len: u32,
        maybe_mapped: Option<ir::DimMap>,
        layout_dims: HashMap<ir::DimId, ir::LayoutDimId>,
        tmp_st_ld: Option<(ir::InstId, ir::InstId)>,
    ) -> Block {
        if let Some(ref dim_map) = maybe_mapped {
            assert!(!dim_map.is_empty());
            self.layouts.insert(id);
        }
        Block {
            id,
            elements_type,
            base_size: len,
            uses: vec![],
            mapped_dims: vec![],
            maybe_mapped: maybe_mapped.unwrap_or_else(ir::DimMap::empty),
            layout_dims,
            tmp_st_ld,
        }
    }

    pub fn expand_blocks_to(&mut self, capacity: usize) {
        self.blocks.expand_to(capacity);
    }

    /// Inserts a new temporary memory. Must be inserted before not_merged is called
    /// on dimensions.
    pub fn set_lazy_tmp<IT>(
        &mut self,
        id: MemId,
        t: Type,
        dims: IT,
        layout_dims: HashMap<ir::DimId, ir::LayoutDimId>,
        tmp_st: ir::InstId,
        tmp_ld: ir::InstId,
    ) where
        IT: Iterator<Item = (ir::DimId, ir::DimId)>,
    {
        let block = self.create_block(
            id,
            t,
            1,
            Some(ir::DimMap::new(dims)),
            layout_dims,
            Some((tmp_st, tmp_ld)),
        );
        self.blocks.set_lazy(id, block);
    }

    /// Registers a use of a memory block by an instruction.
    pub fn register_use(&mut self, mem: MemId, inst: InstId) {
        self.blocks[mem].add_use(inst)
    }

    /// Returns a block given its Id.
    pub fn block(&self, id: MemId) -> &Block {
        &self.blocks[id]
    }

    /// Returns the list of memory blocks.
    pub fn blocks<'b>(&'b self) -> impl Iterator<Item = &'b Block> {
        self.blocks.iter()
    }

    /// Rename a basic block. Returns the lyaouts to lower.
    pub fn merge_dims(&mut self, lhs: ir::DimId, rhs: ir::DimId) -> Vec<MemId> {
        let mut to_lower = Vec::new();
        for block in self.blocks.iter_mut() {
            if block.maybe_mapped.merge_dims(lhs, rhs) && block.is_ready() {
                to_lower.push(block.mem_id());
            }
        }
        to_lower
    }

    /// Registers that two dimensions may not be merged. Returns a list of dimensions
    /// removed from the memory blocks and a list of layouts to lower.
    pub fn not_merged(&mut self, lhs_dim: &ir::Dimension, rhs: ir::DimId) -> Vec<MemId> {
        let lhs = lhs_dim.id();
        let mut to_lower = Vec::new();
        for &id in &self.layouts {
            let mut changed = false; // Ensure we only lower once.
            let block = &mut self.blocks[id];
            trace!("{:?}: {:?} and {:?} not merged", block, lhs_dim.id(), rhs);
            for pair in block.maybe_mapped.filter(|&mut (lhs2, rhs2)| {
                (lhs2 == lhs && rhs2 == rhs) || (lhs2 == rhs && rhs2 == lhs)
            }) {
                block.mapped_dims.push(pair);
                changed = true;
            }
            if changed && block.is_ready() {
                to_lower.push(id);
            }
        }
        to_lower
    }

    /// Lowers a fully defined layout. Returns the load and store instructions with the
    /// corresponding layout dimensions.
    pub fn lower_layout(
        &mut self,
        id: MemId,
    ) -> (
        ir::InstId,
        Vec<ir::LayoutDimId>,
        ir::InstId,
        Vec<ir::LayoutDimId>,
    ) {
        assert!(self.layouts.remove(&id));
        let block = &mut self.blocks[id];
        assert!(block.is_ready());
        let (st_layouts, ld_layouts) = block
            .mapped_dims
            .iter()
            .map(|(lhs, rhs)| (block.layout_dims[lhs], block.layout_dims[rhs]))
            .unzip();
        let (st_inst, ld_inst) = unwrap!(block.tmp_st_ld.take());
        (st_inst, st_layouts, ld_inst, ld_layouts)
    }
}
