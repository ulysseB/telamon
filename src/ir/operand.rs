//! Describes the different kinds of operands an instruction can have.
use self::Operand::*;
use ir::{self, Parameter, Type};
use num::bigint::BigInt;
use num::rational::Ratio;
use num::traits::{Signed, Zero};

/// Represents an instruction operand.
#[derive(Clone, Debug)]
pub enum Operand<'a> {
    /// An integer constant, on a given number of bits.
    Int(BigInt, u16),
    /// A float constant, on a given number of bits.
    Float(Ratio<BigInt>, u16),
    /// The current index in a loop.
    Index(ir::DimId),
    /// A parameter of the function.
    Param(&'a Parameter),
    /// The address of a memory block.
    Addr(ir::ArrayId),
    /// A variable increased by a fixed amount at every step of some loops.
    InductionVar(ir::IndVarId, Type),
    /// A variable, stored in register.
    Variable(ir::VarId, Type),
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
            Variable(_, t) => t,
            InductionVar(_, t) => t,
        }
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

    /// Indicates if the operand stays constant during the execution.
    pub fn is_constant(&self) -> bool {
        match self {
            Int(..) | Float(..) | Addr(..) | Param(..) => true,
            Index(..) | InductionVar(..) | Variable(..) => false,
        }
    }

    /// Ensures the operand respects correctness constraints.
    pub fn check<L>(&self, fun: &ir::Function<L>) -> Result<(), ir::Error> {
        fun.device().check_type(self.t())?;
        if let Variable(var, ..) = self {
            if !fun.variable(*var).use_mode().allow_direct_use() {
                Err(ir::Error::ForbiddenVarUse { var: *var })?;
            }
        }
        Ok(())
    }
}

/// Returns the number of bits necessary to encode a `BigInt`.
fn num_bits(val: &BigInt) -> u16 {
    let mut num_bits = if val.is_negative() { 1 } else { 0 };
    let mut rem = val.abs();
    while !rem.is_zero() {
        rem >>= 1;
        num_bits += 1;
    }
    num_bits
}
