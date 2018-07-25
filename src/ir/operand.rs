//! Describes the different kinds of operands an instruction can have.
use ir::{self, Instruction, InstId, Parameter, DimMap, Type, dim, mem};
use num::bigint::BigInt;
use num::rational::Ratio;
use num::traits::{Signed, Zero};
use self::Operand::*;

/// Indicates how dimensions can be mapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimMapScope {
    /// The dimensions are mapped within registers, without producing syncthreads.
    Local,
    /// The dimensions are mapped within registers.
    Thread,
    /// The dimensions are mapped, possibly using temporary memory.
    Global,
}

/// Represents an instruction operand.
#[derive(Clone, Debug)]
pub enum Operand<'a> {
    /// An integer constant, on a given number of bits.
    Int(BigInt, u16),
    /// A float constant, on a given number of bits.
    Float(Ratio<BigInt>, u16),
    /// A value produced by an instruction. The boolean indicates if the `DimMap` can be
    /// lowered.
    Inst(InstId, Type, DimMap, DimMapScope),
    /// The current index in a loop.
    Index(dim::Id),
    /// A parameter of the function.
    Param(&'a Parameter),
    /// The address of a memory block.
    Addr(mem::InternalId),
    /// The value of the current instruction at a previous iteration.
    Reduce(InstId, Type, DimMap, Vec<ir::dim::Id>),
    /// A variable increased by a fixed amount at every step of some loops.
    InductionVar(ir::IndVarId, Type),
}

impl<'a> Operand<'a> {
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
    pub fn new_inst(inst: &Instruction, dim_map: DimMap, mut scope: DimMapScope)
            -> Operand<'a> {
        // A temporary arry can only be generated if the type size is known.
        if scope == DimMapScope::Global && unwrap!(inst.t()).len_byte().is_none() {
            scope = DimMapScope::Thread
        }
        Inst(inst.id(), unwrap!(inst.t()), dim_map, scope)
    }

    /// Creates a reduce operand from an instruction and a set of dimensions to reduce on.
    pub fn new_reduce(init: &Instruction, dim_map: DimMap, dims: Vec<ir::dim::Id>)
            -> Operand<'a> {
        Reduce(init.id(), unwrap!(init.t()), dim_map, dims)
    }

    /// Creates a new Int operand and checks its number of bits.
    pub fn new_int(val: BigInt, len: u16) -> Operand<'a> {
        assert!(num_bits(&val) <= len);
        Int(val, len)
    }

    /// Creates a new Float operand.
    pub fn new_float(val: Ratio<BigInt>, len: u16) -> Operand<'a> {
        Float(val, len)
    }

    /// Renames a basic block id.
    pub fn merge_dims(&mut self, lhs: ir::dim::Id, rhs: ir::dim::Id) {
        match *self {
            Inst(_, _, ref mut dim_map, _) |
            Reduce(_, _, ref mut dim_map, _) => { dim_map.merge_dims(lhs, rhs); },
            _ => (),
        }
    }

    /// Indicates if a `DimMap` should be lowered if lhs and rhs are not mapped.
    pub fn should_lower_map(&self, lhs: ir::dim::Id, rhs: ir::dim::Id) -> bool {
        match *self {
            Inst(_, _, ref dim_map, _) |
            Reduce(_, _, ref dim_map, _) => {
                dim_map.iter().any(|&pair| pair == (lhs, rhs) || pair == (rhs, lhs))
            },
            _ => false,
        }
    }

    /// If the operand is a reduction, returns the instruction initializing the reduction.
    pub fn as_reduction(&self) -> Option<(InstId, &DimMap, &[ir::dim::Id])> {
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
