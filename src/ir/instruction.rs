//! Describes the instructions.
use device::Device;
use ir::{self, BasicBlock, BBId, Operand, Operator, Type, DimMapScope};
use std;
use utils::*;

/// Uniquely identifies an instruction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct InstId(pub u32);

impl std::fmt::Display for InstId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "instruction {}", self.0)
    }
}

/// Represents an instruction.
#[derive(Clone, Debug)]
pub struct Instruction<'a> {
    operator: Operator<'a>,
    id: InstId,
    iter_dims: HashSet<ir::dim::Id>,
}

impl<'a> Instruction<'a> {
    /// Creates a new instruction and type-check the operands.
    pub fn new(operator: Operator<'a>, id: InstId, iter_dims: HashSet<ir::dim::Id>,
               device: &Device) -> Result<Instruction<'a>, ir::Error> {
        operator.check(&iter_dims, device)?;
        Ok(Instruction { operator, id, iter_dims })
    }

    /// Returns the list of operands of an `Instruction`.
    pub fn operands(&self) -> Vec<&Operand<'a>> { self.operator.operands() }

    /// Returns the type of the value produced by an instruction.
    pub fn t(&self) -> Option<Type> { self.operator.t() }

    /// Returns the operator of the instruction.
    pub fn operator(&self) -> &Operator { &self.operator }

    /// Returns the `InstId` representing the instruction.
    pub fn id(&self) -> InstId { self.id }

    /// Returns true if the instruction has side effects.
    pub fn has_side_effects(&self) -> bool { self.operator.has_side_effects() }

    /// Applies the lowering of a layout to the instruction.
    pub fn lower_layout(&mut self,
                        ld_idx: Operand<'a>, ld_pattern: ir::AccessPattern<'a>,
                        st_idx: Operand<'a>, st_pattern: ir::AccessPattern<'a>) {
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

    /// Indicates the operands for wich a `DimMap` must be lowered if lhs and rhs are
    /// not mapped.
    pub fn dim_maps_to_lower(&self, lhs: ir::dim::Id, rhs: ir::dim::Id) -> Vec<usize> {
        self.operator().operands().iter().enumerate()
            .filter(|&(_, op)| op.should_lower_map(lhs, rhs))
            .map(|(id, _)| id).collect()
    }

    /// Returns 'self' if it is a memory instruction.
    pub fn as_mem_inst(&self) -> Option<&Instruction> {
        self.operator.mem_used().map(|_| self)
    }

    /// Indicates if the instruction performs a reduction.
    pub fn as_reduction(&self) -> Option<(InstId, &ir::DimMap, &[ir::dim::Id])> {
        at_most_one(self.operands().iter().flat_map(|x| x.as_reduction()))
    }

    /// Returns 'true' if `self` is a reduction initialized by init, and if 'dim' should
    /// have the same nesting with 'init' that with 'self'.
    pub fn is_reduction_common_dim(&self, init: InstId, dim: ir::dim::Id) -> bool {
        self.as_reduction().map(|(i, map, rd)| {
            i == init && !rd.contains(&dim) && map.iter().all(|&(_, rhs)| dim != rhs)
        }).unwrap_or(false)
    }

    /// Rename a dimension to another ID.
    pub fn merge_dims(&mut self, lhs: ir::dim::Id, rhs: ir::dim::Id) {
        self.operator.merge_dims(lhs, rhs);
    }

    /// The list of dimensions the instruction must be nested in.
    pub fn iteration_dims(&self) -> &HashSet<ir::dim::Id> { &self.iter_dims }

    /// Adds a new iteration dimension. Indicates if the dimension was not already an
    /// iteration dimension.
    pub fn add_iteration_dimension(&mut self, dim: ir::dim::Id) -> bool {
        self.iter_dims.insert(dim)
    }
}

impl<'a> BasicBlock<'a> for Instruction<'a> {
    fn bb_id(&self) -> BBId { self.id.into() }

    fn as_inst(&self) -> Option<&Instruction<'a>> { Some(self) }
}
