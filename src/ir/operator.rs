//! Defines operators.
use self::Operator::*;
use ir::{self, AccessPattern, Operand, Type};
use std;
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
    fn check(self, t: ir::Type) -> Result<(), ir::TypeError> {
        if t.is_float() ^ (self == Rounding::Exact) {
            Ok(())
        } else {
            Err(ir::TypeError::InvalidRounding { rounding: self, t })
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
    /// Computes the bitwise AND operation.
    And,
    /// Computes the bitwise OR operation.
    Or,
    /// Computes `lhs < rhs`.
    Lt,
    /// Computes `lhs <= rhs`.
    Leq,
    /// Computes `lhs == rhs`.
    Equals,
}

impl BinOp {
    /// Returns a string representing the operator.
    fn name(&self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Div => "div",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Lt => "lt",
            BinOp::Leq => "leq",
            BinOp::Equals => "equals",
        }
    }

    /// Returns the type of the binay operator given the type of its operands.
    pub fn t(&self, operand_type: ir::Type) -> ir::Type {
        match self {
            BinOp::Lt | BinOp::Leq | BinOp::Equals => ir::Type::I(1),
            _ => operand_type,
        }
    }

    /// Indicates if the result must be rounded when operating on floats.
    fn requires_rounding(&self) -> bool {
        match self {
            BinOp::Lt | BinOp::Leq | BinOp::Equals => false,
            _ => true,
        }
    }
}

/// Arithmetic operators with a single operand.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum UnaryOp {
    /// Simply copy the input.
    Mov,
    /// Casts the input to another type.
    Cast(ir::Type),
}

impl UnaryOp {
    /// Gives the return type of the operand given its input type.
    fn t(&self, op_type: ir::Type) -> ir::Type {
        match self {
            UnaryOp::Mov => op_type,
            UnaryOp::Cast(t) => *t,
        }
    }

    /// Returns the name of the operand.
    fn name(&self) -> &'static str {
        match self {
            UnaryOp::Mov => "mov",
            UnaryOp::Cast(..) => "cast",
        }
    }
}

/// The operation performed by an instruction.
#[derive(Clone, Debug)]
pub enum Operator<'a> {
    /// A binary arithmetic operator.
    BinOp(BinOp, Operand<'a>, Operand<'a>, Rounding),
    /// Unary arithmetic operator.
    UnaryOp(UnaryOp, Operand<'a>),
    /// Performs a multiplication with the given return type.
    Mul(Operand<'a>, Operand<'a>, Rounding, Type),
    /// Performs s multiplication between the first two operands and adds the
    /// result to the third.
    Mad(Operand<'a>, Operand<'a>, Operand<'a>, Rounding),
    /// Loads a value of the given type from the given address.
    Ld(Type, Operand<'a>, AccessPattern<'a>),
    /// Stores the second operand at the address given by the first.
    /// The boolean specifies if the instruction has side effects. A store has no side
    /// effects when it writes into a cell that previously had an undefined value.
    St(Operand<'a>, Operand<'a>, bool, AccessPattern<'a>),
    /// Starts a DMA request. The number of element to transfer is the vectorization
    /// factor.
    DmaStart {
        src_ptr: Operand<'a>,
        src_pattern: AccessPattern<'a>,
        dst_ptr: Operand<'a>,
        /// Indicates if the DMA can be executed multiple times without impacting the
        /// result.
        has_side_effects: bool,
        /// Points to the corresponding `DmaWait` instruction. Must be set before starting
        /// the exploration.
        dma_wait: Option<ir::InstId>,
    },
    /// Waits for a DMA request to finish. The vectorization pattern must match the one of
    /// the `DmaStart`. This instruction returns the result of the DMA.
    DmaWait {
        sync_flag: Operand<'a>,
        dst_pattern: AccessPattern<'a>,
        /// Indicates if the DMA can be executed multiple times without impacting the
        /// result. Must match the corresponding flag in `DmaStart`.
        has_side_effects: bool,
        /// Points to the corresponding `DmaStart` instruction.
        dma_start: ir::InstId,
    },
}

impl<'a> Operator<'a> {
    /// Ensures the types of the operands are valid.
    pub fn check<L>(
        &self,
        iter_dims: &HashSet<ir::DimId>,
        fun: &ir::Function<L>,
    ) -> Result<(), ir::Error> {
        self.t()
            .map(|t| fun.device().check_type(t))
            .unwrap_or(Ok(()))?;
        for operand in self.operands() {
            operand.check(fun)?;
        }
        match self {
            BinOp(operator, lhs, rhs, rounding) => {
                if operator.requires_rounding() {
                    rounding.check(lhs.t())?;
                } else if *rounding != Rounding::Exact {
                    Err(ir::TypeError::InvalidRounding {
                        rounding: *rounding,
                        t: lhs.t(),
                    })?;
                }
                ir::TypeError::check_equals(lhs.t(), rhs.t())?;
            }
            Mul(lhs, rhs, rounding, res_type) => {
                rounding.check(lhs.t())?;
                ir::TypeError::check_equals(lhs.t(), rhs.t())?;
                match (lhs.t(), *res_type) {
                    (x, z) if x == z => (),
                    (Type::I(32), Type::I(64)) | (Type::I(32), Type::PtrTo(_)) => (),
                    (_, t) => Err(ir::TypeError::UnexpectedType { t })?,
                }
            }
            Mad(mul_lhs, mul_rhs, add_rhs, rounding) => {
                rounding.check(mul_lhs.t())?;
                ir::TypeError::check_equals(mul_lhs.t(), mul_rhs.t())?;
                match (mul_lhs.t(), add_rhs.t()) {
                    (ref x, ref z) if x == z => (),
                    (Type::I(32), Type::I(64)) | (Type::I(32), Type::PtrTo(_)) => (),
                    (_, t) => Err(ir::TypeError::UnexpectedType { t })?,
                }
            }
            Ld(_, addr, pattern) => {
                pattern.check(iter_dims)?;
                let pointer_type = pattern.pointer_type(fun.device());
                ir::TypeError::check_equals(addr.t(), pointer_type)?;
            }
            St(addr, _, _, pattern) => {
                pattern.check(iter_dims)?;
                let pointer_type = pattern.pointer_type(fun.device());
                ir::TypeError::check_equals(addr.t(), pointer_type)?;
            }
            DmaStart {
                src_ptr,
                src_pattern,
                ..
            } => {
                let pointer_type = src_pattern.pointer_type(fun.device());
                ir::TypeError::check_equals(src_ptr.t(), pointer_type)?;
            }
            DmaWait { sync_flag, .. } => {
                ir::TypeError::check_equals(sync_flag.t(), ir::Type::SyncFlag)?;
            }
            UnaryOp(..) => (),
        }
        Ok(())
    }

    /// Returns the type of the value produced.
    pub fn t(&self) -> Option<Type> {
        match self {
            Mad(_, _, op, _) => Some(op.t()),
            Ld(t, ..) | Mul(.., t) => Some(*t),
            BinOp(operator, lhs, ..) => Some(operator.t(lhs.t())),
            UnaryOp(operator, operand) => Some(operator.t(operand.t())),
            St(..) | DmaWait { .. } => None,
            DmaStart { .. } => Some(ir::Type::SyncFlag),
        }
    }

    /// Retruns the list of operands.
    pub fn operands(&self) -> Vec<&Operand<'a>> {
        match self {
            BinOp(_, lhs, rhs, _) | Mul(lhs, rhs, _, _) | St(lhs, rhs, _, _) => {
                vec![lhs, rhs]
            }
            Mad(mul_lhs, mul_rhs, add_rhs, _) => vec![mul_lhs, mul_rhs, add_rhs],
            UnaryOp(_, op) | Ld(_, op, _) => vec![op],
            DmaStart {
                src_ptr, dst_ptr, ..
            } => vec![src_ptr, dst_ptr],
            DmaWait { sync_flag, .. } => vec![sync_flag],
        }
    }

    /// Retruns the list of mutable references to operands.
    pub fn operands_mut<'b>(&'b mut self) -> Vec<&'b mut Operand<'a>> {
        match self {
            BinOp(_, lhs, rhs, _) | Mul(lhs, rhs, _, _) | St(lhs, rhs, _, _) => {
                vec![lhs, rhs]
            }
            Mad(mul_lhs, mul_rhs, add_rhs, _) => vec![mul_lhs, mul_rhs, add_rhs],
            UnaryOp(_, op, ..) | Ld(_, op, ..) => vec![op],
            DmaStart {
                src_ptr, dst_ptr, ..
            } => vec![src_ptr, dst_ptr],
            DmaWait { sync_flag, .. } => vec![sync_flag],
        }
    }

    /// Returns true if the operator has side effects.
    pub fn has_side_effects(&self) -> bool {
        match self {
            St(_, _, has_side_effects, _)
            | DmaStart {
                has_side_effects, ..
            }
            | DmaWait {
                has_side_effects, ..
            } => *has_side_effects,
            _ => false,
        }
    }

    /// Indicates if the operator accesses memory.
    pub fn is_mem_access(&self) -> bool {
        match self {
            St(..) | Ld(..) | DmaStart { .. } | DmaWait { .. } => true,
            _ => false,
        }
    }

    /// Returns the pattern of access to the memory by the instruction, if any.
    pub fn mem_access_pattern(&self) -> Option<&AccessPattern> {
        match self {
            Ld(_, _, pattern)
            | St(_, _, _, pattern)
            | DmaStart {
                src_pattern: pattern,
                ..
            }
            | DmaWait {
                dst_pattern: pattern,
                ..
            } => Some(pattern),
            _ => None,
        }
    }

    /// Indicates which in-memory variable is accessed by the instruction, if any.
    pub fn accessed_mem_var(&self) -> Option<ir::VarId> {
        self.loaded_mem_var().or(self.stored_mem_var())
    }

    /// Indicates which in-memory variable is used by the instruction, if any.
    pub fn loaded_mem_var(&self) -> Option<ir::VarId> {
        match self {
            Ld(_, _, pattern)
            | DmaStart {
                src_pattern: pattern,
                ..
            } => {
                if let ir::ArrayId::Variable(id) = pattern.accessed_array() {
                    Some(id)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Indicates which in-memory variable is stored by the instruction, if any.
    pub fn stored_mem_var(&self) -> Option<ir::VarId> {
        match self {
            St(.., pattern)
            | DmaWait {
                dst_pattern: pattern,
                ..
            } => {
                if let ir::ArrayId::Variable(id) = pattern.accessed_array() {
                    Some(id)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl<'a> std::fmt::Display for Operator<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let name = match *self {
            BinOp(op, ..) => op.name(),
            UnaryOp(op, ..) => op.name(),
            Mul(..) => "mul",
            Mad(..) => "mad",
            Ld(..) => "ld",
            St(..) => "st",
            DmaStart { .. } => "dma.start",
            DmaWait { .. } => "dma.wait",
        };
        write!(f, "{}", name)
    }
}
