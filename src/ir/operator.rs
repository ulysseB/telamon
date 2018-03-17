//! Defines operators.
use device::Device;
use ir::{self, AccessPattern, Dimension, mem, Operand, Stride, Type};
use itertools::Itertools;
use std::borrow::Cow;
use std::fmt;
use self::Operator::*;

/// The rounding mode of an arithmetic operation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

/// The operation performed by an instruction.
#[derive(Clone, Debug)]
pub enum Operator<'a> {
    /// Adds two operands.
    Add(Operand<'a>, Operand<'a>, Rounding),
    /// Substracts two operands.
    Sub(Operand<'a>, Operand<'a>, Rounding),
    /// Performs a multiplication with the given return type.
    Mul(Operand<'a>, Operand<'a>, Rounding, Type),
    /// Performs s multiplication between the first two operands and adds the
    /// result to the third.
    Mad(Operand<'a>, Operand<'a>, Operand<'a>, Rounding),
    /// Performs a division.
    Div(Operand<'a>, Operand<'a>, Rounding),
    /// Moves a value into a register.
    Mov(Operand<'a>),
    /// Loads a value of the given type from the given address.
    Ld(Type, Operand<'a>, AccessPattern),
    /// Stores the second operand at the address given by the first.
    /// The boolean specifies if the instruction has side effects. A store has no side
    /// effects when it writes into a cell that previously had an undefined value.
    St(Operand<'a>, Operand<'a>, bool, AccessPattern),
    /// Represents a load from a temporary memory that is not fully defined yet.
    TmpLd(Type, mem::Id),
    /// Represents a store to a temporary memory that is not fully defined yet.
    TmpSt(Operand<'a>, mem::Id),
    /// Casts a value into another type.
    Cast(Operand<'a>, Type),
}

impl<'a> Operator<'a> {
    /// Ensures the types of the operands are valid.
    pub fn type_check(&self, device: &Device) {
        assert!(device.is_valid_type(&self.t()));
        // Check operand types.
        for operand in self.operands() {
            let t = operand.t();
            assert_ne!(t, Type::Void);
            assert!(device.is_valid_type(&t));
        }
        match *self {
            Add(ref lhs, ref rhs, rounding) |
            Sub(ref lhs, ref rhs, rounding) |
            Div(ref lhs, ref rhs, rounding) => {
                assert!(lhs.t().is_float() ^ (rounding == Rounding::Exact));
                assert_eq!(lhs.t(), rhs.t());
            },
            Mul(ref lhs, ref rhs, rounding, ref res_type) => {
                assert!(lhs.t().is_float() ^ (rounding == Rounding::Exact));
                match (lhs.t(), rhs.t(), res_type) {
                    (ref x, ref y, z) if x == y && y == z => (),
                    (Type::I(32), Type::I(32), &Type::I(64)) |
                    (Type::I(32), Type::I(32), &Type::PtrTo(_)) => (),
                    _ => panic!(),
                }
            }
            Mad(ref mul_lhs, ref mul_rhs, ref add_rhs, rounding) => {
                assert!(mul_lhs.t().is_float() ^ (rounding == Rounding::Exact));
                match (mul_lhs.t(), mul_rhs.t(), add_rhs.t()) {
                    (ref x, ref y, ref z) if x == y && y == z => (),
                    (Type::I(32), Type::I(32), Type::I(64)) |
                    (Type::I(32), Type::I(32), Type::PtrTo(_)) => (),
                    _ => panic!(),
                }
            },
            Ld(ref t, ref addr, ref pattern) => {
                assert_ne!(*t, Type::Void);
                assert_eq!(addr.t(), Type::PtrTo(pattern.mem_block()));
            },
            St(ref addr, _, _, ref pattern) =>
                assert_eq!(addr.t(), Type::PtrTo(pattern.mem_block())),
            TmpLd(ref t, _) => assert_ne!(*t, Type::Void),
            Cast(ref src, ref t) => {
                assert_ne!(src.t(), Type::Void);
                assert_ne!(*t, Type::Void);
            },
            Mov(..) | TmpSt(..) => (),
        }
    }

    /// Returns the type of the value produced.
    pub fn t(&self) -> Type {
        match *self {
            Add(ref op, _, _) |
            Sub(ref op, _, _) |
            Mov(ref op) |
            Mad(_, _, ref op, _) |
            Div(ref op, _, _) => op.t(),
            Ld(t, ..) | TmpLd(t, _) | Cast(_, t) | Mul(.., t) => t,
            St(..) | TmpSt(..) => Type::Void,
        }
    }

    /// Retruns the list of operands.
    pub fn operands<'b>(&'b self) -> Vec<&'b Operand<'a>> {
        match *self {
            Add(ref lhs, ref rhs, _) |
            Sub(ref lhs, ref rhs, _) |
            Mul(ref lhs, ref rhs, _, _) |
            Div(ref lhs, ref rhs, _) |
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
            Add(ref mut lhs, ref mut rhs, _) |
            Sub(ref mut lhs, ref mut rhs, _) |
            Mul(ref mut lhs, ref mut rhs, _, _) |
            Div(ref mut lhs, ref mut rhs, _) |
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
            Add(..) |
            Sub(..) |
            Mul(..) |
            Mad(..) |
            Mov(..) |
            Div(..) |
            Ld(..) |
            TmpLd(..) |
            TmpSt(..) |
            Cast(..) => false
        }
    }

    /// Returns true if the operator is vectorizable on the given dimension.
    pub fn is_vectorizable(&self, dim: &Dimension) -> bool {
        // TODO(search_space): compute vectorizable info for tmp Ld/St. On vectorization,
        // the layout must be constrained.
        match *self {
            St(_, ref operand, _, ref pattern) => {
                if let Some(type_len) = operand.t().len_byte() {
                    dim.size().as_int().into_iter().any(|x| x == 2 || x == 4) &&
                        pattern.stride(dim.id()) == Stride::Int(type_len as i32)
                } else { false }
            },
            Ld(ref t, _, ref pattern) => {
                if let Some(type_len) = t.len_byte() {
                    dim.size().as_int().into_iter().any(|x| x == 2 || x == 4) &&
                        pattern.stride(dim.id()) == Stride::Int(type_len as i32)
                } else { false }
            },
            Add(..) |
            Sub(..) |
            Mul(..) |
            Mad(..) |
            Div(..) |
            Mov(..) |
            Cast(..) => false,
            TmpLd(..) |
            TmpSt(..) => dim.size().as_int().into_iter().any(|x| x == 2 || x == 4),
        }
    }

    /// Renames a basic block.
    pub fn merge_dims(&mut self, lhs: ir::dim::Id, rhs: ir::dim::Id) {
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
    pub fn mem_used(&self) -> Option<mem::Id> {
        self.mem_access_pattern().map(|p| p.mem_block())
    }

    /// Indicates if the operator supports non-coherent memory accesses.
    pub fn supports_nc_access(&self) -> bool {
        if let Ld(..) = *self { true } else { false }
    }
}

impl<'a> fmt::Display for Operator<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = match *self {
            Add(..) => "add",
            Sub(..) => "sub",
            Mul(..) => "mul",
            Mad(..) => "mad",
            Mov(..) => "mov",
            Div(..) => "div",
            Ld(..) => "ld",
            St(..) => "st",
            TmpLd(..) => "tmp_ld",
            TmpSt(..) => "tmp_st",
            Cast(..) => "cast",
        };
        write!(f, "{}", name)
    }
}
