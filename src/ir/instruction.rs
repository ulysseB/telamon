//! Describes the instructions.
use device::Device;
use ir::{self, BasicBlock, BBId, Operand, Operator, Type, LoweringMap, DimMapScope};
use std;
use utils::*;

/// Uniquely identifies an instruction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct InstId(pub u32);

impl Into<usize> for InstId {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for InstId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "instruction {}", self.0)
    }
}

/// Represents an instruction.
#[derive(Clone, Debug)]
pub struct Instruction<'a, L = LoweringMap> {
    operator: Operator<'a, L>,
    id: InstId,
    iter_dims: HashSet<ir::DimId>,
}

impl<'a, L> Instruction<'a, L> {
    /// Creates a new instruction and type-check the operands.
    pub fn new(operator: Operator<'a, L>, id: InstId, iter_dims: HashSet<ir::DimId>,
               device: &Device) -> Result<Self, ir::Error> {
        operator.check(&iter_dims, device)?;
        Ok(Instruction { operator, id, iter_dims })
    }

    /// Returns an iterator over the operands of this instruction.
    pub fn operands(&self) -> Vec<&Operand<'a, L>> { self.operator.operands() }

    /// Returns the type of the value produced by an instruction.
    pub fn t(&self) -> Option<Type> { self.operator.t() }

    /// Returns the operator of the instruction.
    pub fn operator(&self) -> &Operator<'_, L> { &self.operator }

    /// Returns the `InstId` representing the instruction.
    pub fn id(&self) -> InstId { self.id }

    /// Returns true if the instruction has side effects.
    pub fn has_side_effects(&self) -> bool { self.operator.has_side_effects() }

    /// Applies the lowering of a layout to the instruction.
    pub fn lower_layout(&mut self,
                        ld_idx: Operand<'a, L>, ld_pattern: ir::AccessPattern<'a>,
                        st_idx: Operand<'a, L>, st_pattern: ir::AccessPattern<'a>)
    where L: Clone {
        self.operator = match self.operator.clone() {
            Operator::TmpLd(t, id2) => {
                assert_eq!(ld_pattern.mem_block(), id2);
                Operator::Ld(t, ld_idx, ld_pattern)
            },
            Operator::TmpSt(val, id2) => {
                assert_eq!(st_pattern.mem_block(), id2);
                Operator::St(st_idx, val, false, st_pattern)
            },
            _ => panic!("Only TmpLd/TmpSt are changed on a layout lowering")
        };
    }

    /// Indicates the operands for wich a `DimMap` must be lowered if lhs and rhs are
    /// not mapped.
    pub fn dim_maps_to_lower(&self, lhs: ir::DimId, rhs: ir::DimId) -> Vec<usize> {
        self.operator().operands().iter().enumerate()
            .filter(|&(_, op)| op.should_lower_map(lhs, rhs))
            .map(|(id, _)| id).collect()
    }

    /// Returns 'self' if it is a memory instruction.
    pub fn as_mem_inst(&self) -> Option<&Instruction<'_, L>> {
        self.operator.mem_used().map(|_| self)
    }

    /// Indicates if the instruction performs a reduction.
    pub fn as_reduction(&self) -> Option<(InstId, &ir::DimMap, &[ir::DimId])> {
        at_most_one(self.operands().iter().flat_map(|x| x.as_reduction()))
    }

    /// Returns 'true' if `self` is a reduction initialized by init, and if 'dim' should
    /// have the same nesting with 'init' that with 'self'.
    pub fn is_reduction_common_dim(&self, init: InstId, dim: ir::DimId) -> bool {
        self.as_reduction().map(|(i, map, rd)| {
            i == init && !rd.contains(&dim) && map.iter().all(|&(_, rhs)| dim != rhs)
        }).unwrap_or(false)
    }

    /// Rename a dimension to another ID.
    pub fn merge_dims(&mut self, lhs: ir::DimId, rhs: ir::DimId) {
        self.operator.merge_dims(lhs, rhs);
    }

    /// The list of dimensions the instruction must be nested in.
    pub fn iteration_dims(&self) -> &HashSet<ir::DimId> { &self.iter_dims }

    /// Adds a new iteration dimension. Indicates if the dimension was not already an
    /// iteration dimension.
    pub fn add_iteration_dimension(&mut self, dim: ir::DimId) -> bool {
        self.iter_dims.insert(dim)
    }
}

impl<'a> Instruction<'a, ()> {
    pub fn freeze(self, cnt: &mut ir::Counter) -> Instruction<'a> {
        Instruction {
            operator: self.operator.freeze(cnt),
            id: self.id,
            iter_dims: self.iter_dims,
        }
    }
}

impl<'a> Instruction<'a> {
    /// Lowers the `DimMap` of an operand into an access to a temporary memory.
    pub fn lower_dim_map(&mut self, op_id: usize, new_src: InstId,
                         new_dim_map: ir::DimMap) {
        let operand = &mut *self.operator.operands_mut()[op_id];
        match *operand {
            Operand::Inst(ref mut src, _, ref mut dim_map, ref mut can_lower) => {
                *src = new_src;
                *dim_map = new_dim_map;
                *can_lower = DimMapScope::Local;
            },
            _ => panic!()
        }
    }
}

impl<'a> BasicBlock<'a> for Instruction<'a> {
    fn bb_id(&self) -> BBId { self.id.into() }

    fn as_inst(&self) -> Option<&Instruction<'a>> { Some(self) }
}
