//! Describes the different kinds of operands an instruction can have.
use ir::{self, Instruction, InstId, Parameter, DimMap, Type, mem};
use num::bigint::BigInt;
use num::rational::Ratio;
use num::traits::{Signed, Zero};
use self::Operand::*;
use utils::HashMap;

#[derive(Clone, Debug)]
pub struct LoweringMap {
        /// Memory ID to use for the temporary array
        mem_id: ir::mem::InternalId,
        /// Instruction ID to use for the `store` instruction when
        /// lowering.
        st_inst: ir::InstId,
        /// Maps the lhs dimensions in `map` to their lowered dimension.
        st_map: HashMap<ir::DimId, ir::DimId>,
        /// Instruction ID to use for the `load` instruction when
        /// lowering.
        ld_inst: ir::InstId,
        /// Maps the rhs dimensions in `map` to their lowered dimension.
        ld_map: HashMap<ir::DimId, ir::DimId>,
}

impl LoweringMap {
    /// Creates a new lowering map from an existing dimension map and
    /// a counter. This allocates new IDs for the new
    /// dimensions/instructions/memory locations that will be used
    /// when lowering the DimMap.
    pub fn for_dim_map(dim_map: &DimMap, cnt: &mut ir::Counter) -> LoweringMap {
        let mem_id = cnt.next_mem();
        let st_inst = cnt.next_inst();
        let ld_inst = cnt.next_inst();
        let (st_map, ld_map) = dim_map.iter().cloned().map(|(src, dst)| {
            let st_dim = cnt.next_dim();
            let ld_dim = cnt.next_dim();
            ((src, st_dim), (dst, ld_dim))
        }).unzip();
        
        LoweringMap {
            mem_id,
            st_inst,
            st_map,
            ld_inst,
            ld_map,
        }
    }

    /// Returns lowering information about the dim_map. The returned
    /// `LoweredDimMap` object should not be used immediately: it
    /// refers to fresh IDs that are not activated in the
    /// ir::Function. The appropriate instructions need to be built
    /// and stored with the corresponding IDs.
    pub(crate) fn lower(&self, map: &DimMap) -> ir::LoweredDimMap {
        ir::LoweredDimMap {
            mem: self.mem_id,
            store: self.st_inst,
            load: self.ld_inst,
            dimensions: map.iter().map(|(src, dst)| {
                (unwrap!(self.st_map.get(src)).clone(),
                 unwrap!(self.ld_map.get(dst)).clone())
            }).collect(),
        }
    }
}

/// Indicates how dimensions can be mapped. The `L` type indicates how
/// to lower mapped dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimMapScope<L> {
    /// The dimensions are mapped within registers, without producing syncthreads.
    Local,
    /// The dimensions are mapped within registers.
    Thread,
    /// The dimensions are mapped, possibly using temporary
    /// memory. The parameter `L` is used to indicate how the mapping
    /// should be lowered. It is `()` when building the function
    /// (lowering is not possible at that time), and a `LoweringMap`
    /// instance when exploring which indicates what IDs to use for
    /// the new objects.
    Global(L),
}

/// Represents an instruction operand.
#[derive(Clone, Debug)]
pub enum Operand<'a, L = LoweringMap> {
    /// An integer constant, on a given number of bits.
    Int(BigInt, u16),
    /// A float constant, on a given number of bits.
    Float(Ratio<BigInt>, u16),
    /// A value produced by an instruction. The boolean indicates if the `DimMap` can be
    /// lowered.
    Inst(InstId, Type, DimMap, DimMapScope<L>),
    /// The current index in a loop.
    Index(ir::DimId),
    /// A parameter of the function.
    Param(&'a Parameter),
    /// The address of a memory block.
    Addr(mem::InternalId),
    /// The value of the current instruction at a previous iteration.
    Reduce(InstId, Type, DimMap, Vec<ir::DimId>),
    /// A variable increased by a fixed amount at every step of some loops.
    InductionVar(ir::IndVarId, Type),
}

impl<'a, L> Operand<'a, L> {
    /// Returns the type of the `Operand`.
    pub fn t(&self) -> Type {
        match *self {
            Int(_, n_bit) => Type::I(n_bit),
            Float(_, n_bit) => Type::F(n_bit),
            Addr(mem) => ir::Type::PtrTo(mem.into()),
            Index(..) => Type::I(32),
            Param(p) => p.t,
            Inst(_, t, ..) | Reduce(_, t, ..) | InductionVar(_, t) => t,
        }
    }

    /// Create an operand from an instruction.
    pub fn new_inst(inst: &Instruction<L>, dim_map: DimMap, mut scope: DimMapScope<L>)
            -> Self {
        // A temporary array can only be generated if the type size is known.
        if let DimMapScope::Global(_) = scope {
            if unwrap!(inst.t()).len_byte().is_none() {
                scope = DimMapScope::Thread;
            }
        }

        Inst(inst.id(), unwrap!(inst.t()), dim_map, scope)
    }

    /// Creates a reduce operand from an instruction and a set of dimensions to reduce on.
    pub fn new_reduce(init: &Instruction<L>, dim_map: DimMap, dims: Vec<ir::DimId>)
            -> Self {
        Reduce(init.id(), unwrap!(init.t()), dim_map, dims)
    }

    /// Creates a new Int operand and checks its number of bits.
    pub fn new_int(val: BigInt, len: u16) -> Self {
        assert!(num_bits(&val) <= len);
        Int(val, len)
    }

    /// Creates a new Float operand.
    pub fn new_float(val: Ratio<BigInt>, len: u16) -> Self {
        Float(val, len)
    }

    /// Renames a basic block id.
    pub fn merge_dims(&mut self, lhs: ir::DimId, rhs: ir::DimId) {
        match *self {
            Inst(_, _, ref mut dim_map, _) |
            Reduce(_, _, ref mut dim_map, _) => { dim_map.merge_dims(lhs, rhs); },
            _ => (),
        }
    }

    /// Indicates if a `DimMap` should be lowered if lhs and rhs are not mapped.
    pub fn should_lower_map(&self, lhs: ir::DimId, rhs: ir::DimId) -> bool {
        match *self {
            Inst(_, _, ref dim_map, _) |
            Reduce(_, _, ref dim_map, _) => {
                dim_map.iter().any(|&pair| pair == (lhs, rhs) || pair == (rhs, lhs))
            },
            _ => false,
        }
    }

    /// If the operand is a reduction, returns the instruction initializing the reduction.
    pub fn as_reduction(&self) -> Option<(InstId, &DimMap, &[ir::DimId])> {
        if let Reduce(id, _, ref dim_map, ref dims) = *self {
            Some((id, dim_map, dims))
        } else { None }
    }

    /// Indicates if the operand stays constant during the execution.
    pub fn is_constant(&self) -> bool {
        match *self {
            Int(..) | Float(..) | Addr(..) | Param(..) => true,
            Index(..) | Inst(..) | Reduce(..) | InductionVar(..) => false,
        }
    }
}

impl<'a> Operand<'a, ()> {
    pub fn freeze(self, cnt: &mut ir::Counter) -> Operand<'a> {
        match self {
            Int(val, len) => Int(val, len),
            Float(val, len) => Float(val, len),
            Inst(id, t, dim_map, DimMapScope::Global(())) => {
                let lowering_map = LoweringMap::for_dim_map(
                    &dim_map, cnt);
                Inst(id, t, dim_map, DimMapScope::Global(lowering_map))
            },
            Inst(id, t, dim_map, DimMapScope::Local) =>
                Inst(id, t, dim_map, DimMapScope::Local),
            Inst(id, t, dim_map, DimMapScope::Thread) =>
                Inst(id, t, dim_map, DimMapScope::Thread),
            Index(id) => Index(id),
            Param(param) => Param(param),
            Addr(id) => Addr(id),
            Reduce(id, t, dim_map, dims) => Reduce(id, t, dim_map, dims),
            InductionVar(id, t) => InductionVar(id, t),
        }
    }
}

/// Returns the number of bits necessary to encode a `BigInt`.
fn num_bits(val: &BigInt) -> u16 {
    let mut num_bits = if val.is_negative() { 1 } else { 0 };
    let mut rem = val.abs();
    while !rem.is_zero() {
        rem = rem >> 1;
        num_bits += 1;
    }
    num_bits
}
