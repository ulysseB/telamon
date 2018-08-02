//! Defines operators.
use device::Device;
use ir::{self, AccessPattern, Operand, Type};
use itertools::Itertools;
use std;
use std::borrow::Cow;
use self::Operator::*;
use utils::*;

/// The rounding mode of an arithmetic operation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub enum Rounding {
    /// No rounding occurs.
    Exact,
    /// Rounds toward the nearest number.
    Nearest,
    /// Rounds toward zero.
    Zero,
    /// Rounds toward positive infinite.
    Positive,
    /// Rounds toward negative infinite.
    Negative,
}

impl std::fmt::Display for Rounding {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let name = match self {
            Rounding::Exact => "exact",
            Rounding::Nearest => "toward nearest",
            Rounding::Zero => "toward zero",
            Rounding::Positive => "toward +inf",
            Rounding::Negative => "toward -inf",
        };
        write!(f, "{}", name)
    }
}

impl Rounding {
    /// Ensures the rounding policy applies to the given type.
    fn check(&self, t: ir::Type) -> Result<(), ir::TypeError> {
        if t.is_float() ^ (*self == Rounding::Exact) { Ok(()) } else {
            Err(ir::TypeError::InvalidRounding { rounding: *self, t })
        }
    }
}

/// Represents binary arithmetic operators.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum BinOp {
    /// Adds two operands.
    Add,
    /// Substracts two operands.
    Sub,
    /// Divides two operands,
    Div,
}

impl BinOp {
    /// Returns a string representing the operator.
    fn as_str(&self) -> &'static str {
        match *self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Div => "div",
        }
    }
}

/// The operation performed by an instruction.
#[derive(Clone, Debug)]
pub enum Operator<'a> {
    /// A binary arithmetic operator.
    BinOp(BinOp, Operand<'a>, Operand<'a>, Rounding),
    /// Performs a multiplication with the given return type.
    Mul(Operand<'a>, Operand<'a>, Rounding, Type),
    /// Performs s multiplication between the first two operands and adds the
    /// result to the third.
    Mad(Operand<'a>, Operand<'a>, Operand<'a>, Rounding),
    /// Moves a value into a register.
    Mov(Operand<'a>),
    /// Loads a value of the given type from the given address.
    Ld(Type, Operand<'a>, AccessPattern<'a>),
    /// Stores the second operand at the address given by the first.
    /// The boolean specifies if the instruction has side effects. A store has no side
    /// effects when it writes into a cell that previously had an undefined value.
    St(Operand<'a>, Operand<'a>, bool, AccessPattern<'a>),
    /// Represents a load from a temporary memory that is not fully defined yet.
    TmpLd(Type, ir::MemId),
    /// Represents a store to a temporary memory that is not fully defined yet.
    TmpSt(Operand<'a>, ir::MemId),
    /// Casts a value into another type.
    Cast(Operand<'a>, Type),
}

impl<'a> Operator<'a> {
    /// Ensures the types of the operands are valid.
    pub fn check(&self, iter_dims: &HashSet<ir::DimId>, device: &Device)
        -> Result<(), ir::Error>
    {
        self.t().map(|t| device.check_type(t)).unwrap_or(Ok(()))?;
        for operand in self.operands() { device.check_type(operand.t())?; }
        match *self {
            BinOp(_, ref lhs, ref rhs, rounding) => {
                rounding.check(lhs.t())?;
                ir::TypeError::check_equals(lhs.t(), rhs.t())?;
            },
            Mul(ref lhs, ref rhs, rounding, res_type) => {
                rounding.check(lhs.t())?;
                ir::TypeError::check_equals(lhs.t(), rhs.t())?;
                match (lhs.t(), res_type) {
                    (x, z) if x == z => (),
                    (Type::I(32), Type::I(64)) |
                    (Type::I(32), Type::PtrTo(_)) => (),
                    (_, t) => Err(ir::TypeError::UnexpectedType { t })?,
                }
            }
            Mad(ref mul_lhs, ref mul_rhs, ref add_rhs, rounding) => {
                rounding.check(mul_lhs.t())?;
                ir::TypeError::check_equals(mul_lhs.t(), mul_rhs.t())?;
                match (mul_lhs.t(), add_rhs.t()) {
                    (ref x, ref z) if x == z => (),
                    (Type::I(32), Type::I(64)) |
                    (Type::I(32), Type::PtrTo(_)) => (),
                    (_, t) => Err(ir::TypeError::UnexpectedType { t })?,
                }
            },
            Ld(_, ref addr, ref pattern) => {
                pattern.check(iter_dims)?;
                ir::TypeError::check_equals(addr.t(), Type::PtrTo(pattern.mem_block()))?;
            },
            St(ref addr, _, _, ref pattern) => {
                pattern.check(iter_dims)?;
                ir::TypeError::check_equals(addr.t(), Type::PtrTo(pattern.mem_block()))?;
            },
            TmpLd(..) | Cast(..) | Mov(..) | TmpSt(..) => (),
        }
        Ok(())
    }

    /// Returns the type of the value produced.
    pub fn t(&self) -> Option<Type> {
        match *self {
            BinOp(_, ref op, ..) |
            Mov(ref op) |
            Mad(_, _, ref op, _) => Some(op.t()),
            Ld(t, ..) | TmpLd(t, _) | Cast(_, t) | Mul(.., t) => Some(t),
            St(..) | TmpSt(..) => None,
        }
    }

    /// Retruns the list of operands.
    pub fn operands(&self) -> Vec<&Operand<'a>> {
        match *self {
            BinOp(_, ref lhs, ref rhs, _) |
            Mul(ref lhs, ref rhs, _, _) |
            St(ref lhs, ref rhs, _, _) => vec![lhs, rhs],
            Mad(ref mul_lhs, ref mul_rhs, ref add_rhs, _) =>
                vec![mul_lhs, mul_rhs, add_rhs],
            Mov(ref op) |
            Ld(_, ref op, _) |
            TmpSt(ref op, _) |
            Cast(ref op, _) => vec![op],
            TmpLd(..) => vec![],
        }
    }

    /// Retruns the list of mutable references to operands.
    pub fn operands_mut<'b>(&'b mut self) -> Vec<&'b mut Operand<'a>> {
        match *self {
            BinOp(_, ref mut lhs, ref mut rhs, _) |
            Mul(ref mut lhs, ref mut rhs, _, _) |
            St(ref mut lhs, ref mut rhs, _, _) => vec![lhs, rhs],
            Mad(ref mut mul_lhs, ref mut mul_rhs, ref mut add_rhs, _) =>
                vec![mul_lhs, mul_rhs, add_rhs],
            Mov(ref mut op) |
            Ld(_, ref mut op, _) |
            TmpSt(ref mut op, _) |
            Cast(ref mut op, _) => vec![op],
            TmpLd(..) => vec![],
        }
    }

    /// Returns true if the operator has side effects.
    pub fn has_side_effects(&self) -> bool {
        match *self {
            St(_, _, b, _) => b,
            BinOp(..) |
            Mul(..) |
            Mad(..) |
            Mov(..) |
            Ld(..) |
            TmpLd(..) |
            TmpSt(..) |
            Cast(..) => false
        }
    }

    /// Renames a basic block.
    pub fn merge_dims(&mut self, lhs: ir::DimId, rhs: ir::DimId) {
        self.operands_mut().iter_mut().foreach(|x| x.merge_dims(lhs, rhs));
    }

    /// Returns the pattern of access to the memory by the instruction, if any.
    pub fn mem_access_pattern(&self) -> Option<Cow<AccessPattern>> {
        match *self {
            Ld(_, _, ref pattern) |
            St(_, _, _, ref pattern) => Some(Cow::Borrowed(pattern)),
            TmpLd(_, mem_id) |
            TmpSt(_, mem_id) => {
                Some(Cow::Owned(AccessPattern::Unknown { mem_id }))
            },
            _ => None,
        }
    }

    /// Returns the memory blocks referenced by the instruction.
    pub fn mem_used(&self) -> Option<ir::MemId> {
        self.mem_access_pattern().map(|p| p.mem_block())
    }

    /// Indicates if the operator supports non-coherent memory accesses.
    pub fn supports_nc_access(&self) -> bool {
        if let Ld(..) = *self { true } else { false }
    }
}

impl<'a> std::fmt::Display for Operator<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let name = match *self {
            BinOp(op, ..) => op.as_str(),
            Mul(..) => "mul",
            Mad(..) => "mad",
            Mov(..) => "mov",
            Ld(..) => "ld",
            St(..) => "st",
            TmpLd(..) => "tmp_ld",
            TmpSt(..) => "tmp_st",
            Cast(..) => "cast",
        };
        write!(f, "{}", name)
    }
}
