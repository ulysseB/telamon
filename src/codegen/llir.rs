///! This module defines a Low-Level IR
use std::borrow::Cow;
use std::convert::{TryFrom, TryInto};
use std::num::NonZeroU32;
use std::{error, fmt, iter};

use itertools::Itertools;
use num::bigint::BigInt;
use num::rational::Ratio;

use crate::ir;
use crate::search_space::{InstFlag, MemSpace};

/// Checks that all the types in `types` are identical, and returns that type.
///
/// # Errors
///
/// Fails if `types` contains several different types.
///
/// # Panics
///
/// Panics if `types` is empty.
fn unify_type(types: impl IntoIterator<Item = ir::Type>) -> Result<ir::Type, TypeError> {
    let mut types = types.into_iter();
    let first = types.next().unwrap_or_else(|| panic!("no types provided"));
    if let Some(other) = types.find(|&other| other != first) {
        Err(TypeError::mismatch(first, other))
    } else {
        Ok(first)
    }
}

/// Checks that all the types in `types` are the same integer type, and returns it.
///
/// # Errors
///
/// Fails if `types` contains different types, or non-integer types
///
/// # Panics
///
/// Panics if `types` is empty.
fn unify_itype(types: impl IntoIterator<Item = ir::Type>) -> Result<ir::Type, TypeError> {
    unify_type(types).and_then(|t| {
        if t.is_integer() {
            Ok(t)
        } else {
            Err(TypeError::not_integer(t))
        }
    })
}

/// Checks that all the types inf `types` are the same floating-point type, and returns it.
///
/// # Errors
///
/// Fails if `types` contains different types, or non floating-point types.
///
/// # Panics
///
/// Panics if `types` is empty.
fn unify_ftype(types: impl IntoIterator<Item = ir::Type>) -> Result<ir::Type, TypeError> {
    unify_type(types).and_then(|t| {
        if t.is_float() {
            Ok(t)
        } else {
            Err(TypeError::not_float(t))
        }
    })
}

/// A named register.
///
/// Registers are typed, and should only be used in instructions expecting the appropriate type.
#[derive(Debug, Copy, Clone)]
pub struct Register<'a> {
    name: &'a str,
    t: ir::Type,
}

impl fmt::Display for Register<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(self.name)
    }
}

impl<'a> Register<'a> {
    /// Create a new named register with given type.
    pub fn new(name: &'a str, t: ir::Type) -> Self {
        Register { name, t }
    }

    /// Name of the register
    pub fn name(self) -> &'a str {
        self.name
    }

    /// The type of values stored in the register.
    pub fn t(self) -> ir::Type {
        self.t
    }

    /// Converts the register to an operand for use as a value
    pub fn into_operand(self) -> Operand<'a> {
        Operand::Register(self)
    }
}

/// An operand which can be used as input argument of an instruction.
#[derive(Debug, Clone)]
pub enum Operand<'a> {
    Register(Register<'a>),
    IntLiteral(Cow<'a, BigInt>, u16),
    FloatLiteral(Cow<'a, Ratio<BigInt>>, u16),
}

impl fmt::Display for Operand<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Register(register) => fmt::Display::fmt(register, fmt),
            &Operand::IntLiteral(ref value, bits) => write!(fmt, "{}i{}", value, bits),
            &Operand::FloatLiteral(ref value, bits) => {
                write!(fmt, "({}) as f{}", value, bits)
            }
        }
    }
}

impl<'a> From<Register<'a>> for Operand<'a> {
    fn from(register: Register<'a>) -> Operand<'a> {
        Operand::Register(register)
    }
}

impl<'a> Operand<'a> {
    /// The register represented by this operand, if there is one.
    pub fn to_register(&self) -> Option<Register<'a>> {
        match *self {
            Operand::Register(register) => Some(register),
            _ => None,
        }
    }

    /// The type of this operand.
    pub fn t(&self) -> ir::Type {
        match *self {
            Operand::Register(register) => register.t(),
            Operand::IntLiteral(_, bits) => ir::Type::I(bits),
            Operand::FloatLiteral(_, bits) => ir::Type::F(bits),
        }
    }
}

/// Trait to convert integer literals to operands
pub trait IntLiteral<'a>: ir::IntLiteral<'a> + Sized {
    /// Converts this value into an integer literal operand with the same number of bits
    fn int_literal(self) -> Operand<'a> {
        let (value, bits) = self.decompose();
        Operand::IntLiteral(value, bits)
    }

    /// Converts this value into an integer literal operand that is possibly wider.
    ///
    /// # Errors
    ///
    /// Fails if `t` is not an integral type, or `t` does not have a large enough bitwidth to
    /// represent the value.
    fn typed_int_literal(
        self,
        t: ir::Type,
    ) -> Result<Operand<'a>, InvalidTypeForIntLiteral> {
        let (value, bits) = self.decompose();
        match t {
            ir::Type::I(tbits) if bits <= tbits => Ok(Operand::IntLiteral(value, tbits)),
            _ => Err(InvalidTypeForIntLiteral(value.to_string(), bits, t)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InvalidTypeForIntLiteral(String, u16, ir::Type);

impl fmt::Display for InvalidTypeForIntLiteral {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            fmt,
            "unable to use `{}i{}` as literal of type `{}`",
            self.0, self.1, self.2
        )
    }
}

impl error::Error for InvalidTypeForIntLiteral {}

impl<'a, T: ir::IntLiteral<'a>> IntLiteral<'a> for T {}

/// Trait to convert floating-point literals to operands
pub trait FloatLiteral<'a>: ir::FloatLiteral<'a> + Sized {
    /// Converts this value into a floating-point literal operand with the same width
    fn float_literal(self) -> Operand<'a> {
        let (value, bits) = self.decompose();
        Operand::FloatLiteral(value, bits)
    }
}

impl<'a, T: ir::FloatLiteral<'a>> FloatLiteral<'a> for T {}

/// An address operand.
///
/// Addresses are separated from other operands because they have a slightly more complex
/// representation, notably due to being able to add immediate offsets.
///
/// Unlike regular operands, they are also only allowed as arguments of memory instructions.
#[derive(Debug, Copy, Clone)]
pub enum Address<'a> {
    /// A register with an immediate offset
    Register(Register<'a>, i32),
}

impl fmt::Display for Address<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Address::Register(reg, offset) if offset == 0 => write!(fmt, "[{}]", reg),
            Address::Register(reg, offset) => write!(fmt, "[{}+0x{:x}]", reg, offset),
        }
    }
}

impl<'a> TryFrom<Operand<'a>> for Address<'a> {
    type Error = InvalidOperandAsAddressError;

    fn try_from(operand: Operand<'a>) -> Result<Self, Self::Error> {
        match operand {
            Operand::Register(reg) if reg.t().is_integer() => {
                Ok(Address::Register(reg, 0))
            }
            _ => Err(InvalidOperandAsAddressError(operand.to_string())),
        }
    }
}

/// The error type used when an operand cannot be used as an address.
#[derive(Debug, Clone)]
pub struct InvalidOperandAsAddressError(String);

impl fmt::Display for InvalidOperandAsAddressError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "invalid operand used as address: `{}`", self.0)
    }
}

impl error::Error for InvalidOperandAsAddressError {}

/// Wrapper type for representing either scalar or vector operands or registers.
///
/// This should usually not be used directly but rather through the `RegVec` and `OpVec` type
/// aliases.
#[derive(Debug, Clone)]
pub enum ScalarOrVector<T> {
    Scalar(T),
    Vector(Vec<T>),
}

impl<T: fmt::Display> fmt::Display for ScalarOrVector<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarOrVector::Scalar(scalar) => fmt::Display::fmt(scalar, fmt),
            ScalarOrVector::Vector(vec) => write!(fmt, "{{{}}}", vec.iter().format(", ")),
        }
    }
}

impl<T> From<T> for ScalarOrVector<T> {
    fn from(scalar: T) -> Self {
        ScalarOrVector::Scalar(scalar)
    }
}

/// Either a single register or a vector of registers
///
/// Empty vectors are invalid, and so are vectors with registers of different types.
pub type RegVec<'a> = ScalarOrVector<Register<'a>>;

impl<'a> RegVec<'a> {
    /// The type of the register.
    ///
    /// In case of a vector register, it is assumed that all elements have the same type, and only
    /// the first one is returned.
    ///
    /// # Panics
    ///
    /// Panics if `self` is an empty vector.
    pub fn t(&self) -> ir::Type {
        match self {
            ScalarOrVector::Scalar(reg) => reg.t(),
            ScalarOrVector::Vector(regs) => regs[0].t(),
        }
    }
}

/// Either a single operand or a vector of operands
///
/// Empty vectors are invalid, and so are vectors with operands of different types.
pub type OpVec<'a> = ScalarOrVector<Operand<'a>>;

impl<'a> OpVec<'a> {
    /// The type of the operand.
    ///
    /// In case of a vector operand, it is assumed that all elements have the same type, and only
    /// the first one is returned.
    ///
    /// # Panics
    ///
    /// Panics if `self` is an empty vector.
    pub fn t(&self) -> ir::Type {
        match self {
            ScalarOrVector::Scalar(oper) => oper.t(),
            ScalarOrVector::Vector(opers) => opers[0].t(),
        }
    }
}

/// A typed unary operator.
#[derive(Debug, Copy, Clone)]
pub enum UnOp {
    Move { t: ir::Type },
    Cast { src_t: ir::Type, dst_t: ir::Type },
    // Natural exponential
    Exp { t: ir::Type },
}

impl fmt::Display for UnOp {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnOp::Move { t } => write!(fmt, "move.{}", t),
            UnOp::Cast { src_t, dst_t } => write!(fmt, "cast.{}.{}", dst_t, src_t),
            UnOp::Exp { t } => write!(fmt, "exp.{}", t),
        }
    }
}

impl UnOp {
    /// Converts an `ir::UnaryOp` with a given operand type to an unary operator.
    ///
    /// # Errors
    ///
    /// Fails if the `ir::UnaryOp` is not compatible with the requested `arg_t`.
    pub fn from_ir(op: ir::UnaryOp, arg_t: ir::Type) -> Result<Self, InstructionError> {
        Ok(match op {
            ir::UnaryOp::Mov => UnOp::Move { t: arg_t },
            ir::UnaryOp::Cast(dst_t) => UnOp::Cast {
                src_t: arg_t,
                dst_t,
            },
            ir::UnaryOp::Exp(t) => UnOp::Exp {
                t: Self::unify_type(Some(t), [arg_t])?,
            },
        })
    }

    /// The expected argument type for this operator.
    pub fn arg_t(self) -> [ir::Type; 1] {
        match self {
            UnOp::Move { t } | UnOp::Cast { src_t: t, .. } | UnOp::Exp { t } => [t],
        }
    }

    /// The resulting type when this operator is applied.
    pub fn ret_t(self) -> ir::Type {
        match self {
            UnOp::Move { t } | UnOp::Cast { dst_t: t, .. } | UnOp::Exp { t } => t,
        }
    }

    fn unify_type(d: Option<ir::Type>, a: [ir::Type; 1]) -> Result<ir::Type, TypeError> {
        unify_type(d.into_iter().chain(a.iter().copied()))
    }

    /// Create a `move` operator based on its destination and argument types.
    ///
    /// # Errors
    ///
    /// Fails if `d` and `a` are different types.
    pub fn infer_move(
        d: Option<ir::Type>,
        a: [ir::Type; 1],
    ) -> Result<Self, InstructionError> {
        Ok(Self::unify_type(d, a).map(|t| UnOp::Move { t })?)
    }

    /// Create a `cast` operator based on its destination and argument types.
    ///
    /// # Errors
    ///
    /// Fails if `dst_t` and `d` are different types.
    pub fn infer_cast(
        dst_t: ir::Type,
        d: Option<ir::Type>,
        [a]: [ir::Type; 1],
    ) -> Result<Self, InstructionError> {
        Ok(Self::unify_type(d, [dst_t]).map(|dst_t| UnOp::Cast { dst_t, src_t: a })?)
    }

    /// Create an `exp` operator based on its destination and argument types.
    ///
    /// # Errors
    ///
    /// Fails if `d` and `a` are different types.
    pub fn infer_exp(
        d: Option<ir::Type>,
        a: [ir::Type; 1],
    ) -> Result<Self, InstructionError> {
        Ok(Self::unify_type(d, a).map(|t| UnOp::Exp { t })?)
    }
}

/// Comparison operators
#[derive(Debug, Copy, Clone)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl fmt::Display for CmpOp {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(match self {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
        })
    }
}

/// A typed binary operator
#[derive(Debug, Copy, Clone)]
pub enum BinOp {
    // Integer Arithmetic Instructions
    IAdd { arg_t: ir::Type },
    ISub { arg_t: ir::Type },
    IDiv { arg_t: ir::Type },
    IMul { arg_t: ir::Type, spec: MulSpec },
    // Floating-Point Instructions
    FAdd { t: ir::Type, rounding: FpRounding },
    FSub { t: ir::Type, rounding: FpRounding },
    FMul { t: ir::Type, rounding: FpRounding },
    FDiv { t: ir::Type, rounding: FpRounding },
    FMax { t: ir::Type },
    FMin { t: ir::Type },
    // Comparison and Selection Instructions
    Set { op: CmpOp, arg_t: ir::Type },
    // Logic and Shift Instructions
    And { t: ir::Type },
    Or { t: ir::Type },
    Xor { t: ir::Type },
}

impl fmt::Display for BinOp {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use BinOp::*;

        match self {
            // Integer Arithmetic Instructions
            IAdd { arg_t } => write!(fmt, "add.{}", arg_t),
            ISub { arg_t } => write!(fmt, "sub.{}", arg_t),
            IDiv { arg_t } => write!(fmt, "div.{}", arg_t),
            IMul { arg_t, spec } => write!(fmt, "mul.{}.{}", spec, arg_t),
            // Floating-Point Instructions
            FAdd { t, rounding } => write!(fmt, "add.{}.{}", rounding, t),
            FSub { t, rounding } => write!(fmt, "sub.{}.{}", rounding, t),
            FMul { t, rounding } => write!(fmt, "mul.{}.{}", rounding, t),
            FDiv { t, rounding } => write!(fmt, "div.{}.{}", rounding, t),
            FMax { t } => write!(fmt, "max.{}", t),
            FMin { t } => write!(fmt, "min.{}", t),
            // Comparison and Selection Instructions
            Set { op, arg_t } => write!(fmt, "set.{}.{}", op, arg_t),
            // Logic and Shift Instructions
            And { t } => write!(fmt, "and.{}", t),
            Or { t } => write!(fmt, "or.{}", t),
            Xor { t } => write!(fmt, "xor.{}", t),
        }
    }
}

// Helper macro to reduce the boilerplate of writing `infer_[op]` functions
macro_rules! infer_binops {
    ($($infer:ident, $con:ident { $t:ident $(, $arg:ident: $argTy:ty)* }, $unify:ident;)*) => {
        $(pub fn $infer($($arg: $argTy ,)* d: Option<ir::Type>, ab: [ir::Type; 2]) -> Result<Self, InstructionError> {
            Ok(Self::$unify(d, ab).map(|$t| BinOp::$con { $t $(, $arg)* })?)
        })*
    };
}

impl BinOp {
    /// Converts an `ir::BinOp` with given rounding mode and operand types to a binary operator.
    ///
    /// # Errors
    ///
    /// Fails if the `ir::BinOp` is not compatible with the requested rounding mode and operand
    /// types.
    pub fn from_ir(
        op: ir::BinOp,
        rounding: ir::op::Rounding,
        lhs_t: ir::Type,
        rhs_t: ir::Type,
    ) -> Result<Self, InstructionError> {
        use ir::{BinOp as iop, Type as ity};

        let arg_t = Self::unify_type(None, [lhs_t, rhs_t])?;
        match arg_t {
            ity::I(_) if rounding != ir::op::Rounding::Exact => {
                return Err(InstructionError::invalid_rounding_for_type(rounding, arg_t))
            }
            ity::F(_) => match op {
                iop::Add | iop::Sub | iop::Div => (),
                iop::Max => {
                    if rounding != ir::op::Rounding::Exact {
                        return Err(InstructionError::invalid_rounding_for_op(
                            op, rounding,
                        ));
                    }
                }
                _ => return Err(InstructionError::invalid_binop_for_type(op, arg_t)),
            },
            // Type is not lowered
            _ => return Err(InstructionError::invalid_type(arg_t)),
        }

        Ok(match (op, arg_t) {
            (iop::Add, ity::I(_)) => BinOp::IAdd { arg_t },
            (iop::Sub, ity::I(_)) => BinOp::ISub { arg_t },
            (iop::Div, ity::I(_)) => BinOp::IDiv { arg_t },
            (iop::And, ity::I(_)) => BinOp::And { t: arg_t },
            (iop::Or, ity::I(_)) => BinOp::Or { t: arg_t },
            (iop::Add, ity::F(_)) => BinOp::FAdd {
                t: arg_t,
                rounding: rounding.into(),
            },
            (iop::Sub, ity::F(_)) => BinOp::FSub {
                t: arg_t,
                rounding: rounding.into(),
            },
            (iop::Div, ity::F(_)) => BinOp::FDiv {
                t: arg_t,
                rounding: rounding.into(),
            },
            (iop::Lt, _) => BinOp::Set {
                op: CmpOp::Lt,
                arg_t,
            },
            (iop::Leq, _) => BinOp::Set {
                op: CmpOp::Le,
                arg_t,
            },
            (iop::Equals, _) => BinOp::Set {
                op: CmpOp::Eq,
                arg_t,
            },
            (iop::Max, ity::F(_)) => BinOp::FMax { t: arg_t },
            _ => return Err(InstructionError::invalid_binop_for_type(op, arg_t)),
        })
    }

    /// Create a new multiplication operator based on its rounding mode, argument types, and return
    /// type.
    ///
    /// # Errors
    ///
    /// Fails if a rounding mode is provided for integer multiplication, or if the argument and
    /// result types are not compatibles.
    pub fn from_ir_mul(
        rounding: ir::op::Rounding,
        lhs_t: ir::Type,
        rhs_t: ir::Type,
        ret_t: ir::Type,
    ) -> Result<Self, InstructionError> {
        match lhs_t {
            ir::Type::I(_) => {
                if rounding != ir::op::Rounding::Exact {
                    Err(InstructionError::invalid_rounding_for_type(rounding, lhs_t))
                } else {
                    Ok(BinOp::IMul {
                        arg_t: lhs_t,
                        spec: MulSpec::from_ir(lhs_t, rhs_t, ret_t)?,
                    })
                }
            }
            ir::Type::F(_) => Ok(BinOp::FMul {
                t: Self::unify_ftype(Some(ret_t), [lhs_t, rhs_t])?,
                rounding: rounding.into(),
            }),
            // Type is not lowered
            _ => Err(InstructionError::invalid_type(lhs_t)),
        }
    }

    /// The expected argument types for this operator.
    pub fn arg_t(self) -> [ir::Type; 2] {
        use BinOp::*;

        match self {
            IAdd { arg_t }
            | ISub { arg_t }
            | IDiv { arg_t }
            | IMul { arg_t, .. }
            | Set { arg_t, .. } => [arg_t, arg_t],
            FAdd { t, .. }
            | FSub { t, .. }
            | FMul { t, .. }
            | FDiv { t, .. }
            | FMax { t }
            | FMin { t }
            | And { t }
            | Or { t }
            | Xor { t } => [t, t],
        }
    }

    /// The resulting type when this operator is applied.
    pub fn ret_t(self) -> ir::Type {
        use BinOp::*;

        match self {
            IAdd { arg_t } | ISub { arg_t } | IDiv { arg_t } => arg_t,
            IMul { arg_t, spec } => spec.ret_t(arg_t),
            Set { .. } => ir::Type::I(1),
            FAdd { t, .. }
            | FSub { t, .. }
            | FMul { t, .. }
            | FDiv { t, .. }
            | FMax { t }
            | FMin { t }
            | And { t }
            | Or { t }
            | Xor { t } => t,
        }
    }

    fn unify_itype(
        d: Option<ir::Type>,
        ab: [ir::Type; 2],
    ) -> Result<ir::Type, TypeError> {
        unify_itype(d.into_iter().chain(ab.iter().copied()))
    }

    fn unify_ftype(
        d: Option<ir::Type>,
        ab: [ir::Type; 2],
    ) -> Result<ir::Type, TypeError> {
        unify_ftype(d.into_iter().chain(ab.iter().copied()))
    }

    fn unify_type(d: Option<ir::Type>, ab: [ir::Type; 2]) -> Result<ir::Type, TypeError> {
        unify_type(d.into_iter().chain(ab.iter().copied()))
    }

    infer_binops! {
        infer_iadd, IAdd { arg_t }, unify_itype;
        infer_isub, ISub { arg_t }, unify_itype;
        infer_idiv, IDiv { arg_t }, unify_itype;
        infer_fadd, FAdd { t, rounding: FpRounding }, unify_ftype;
        infer_fsub, FSub { t, rounding: FpRounding }, unify_ftype;
        infer_fdiv, FDiv { t, rounding: FpRounding }, unify_ftype;
        infer_fmul, FMul { t, rounding: FpRounding }, unify_ftype;
        infer_fmax, FMax { t }, unify_ftype;
        infer_fmin, FMin { t }, unify_ftype;
        infer_and, And { t }, unify_itype;
        infer_xor, Xor { t }, unify_itype;
        infer_or, Or { t }, unify_itype;
    }

    pub fn infer_imul(
        spec: MulSpec,
        d: Option<ir::Type>,
        ab: [ir::Type; 2],
    ) -> Result<Self, InstructionError> {
        let arg_t = Self::unify_itype(None, ab)?;
        unify_itype(d.into_iter().chain(iter::once(spec.ret_t(arg_t))))?;
        Ok(BinOp::IMul { arg_t, spec })
    }

    pub fn infer_set(
        op: CmpOp,
        d: Option<ir::Type>,
        ab: [ir::Type; 2],
    ) -> Result<Self, InstructionError> {
        let arg_t = Self::unify_type(None, ab)?;
        unify_itype(d.into_iter().chain(iter::once(ir::Type::I(1))))?;
        Ok(BinOp::Set { op, arg_t })
    }
}

/// A typed ternary operator (e.g. fma)
#[derive(Debug, Copy, Clone)]
pub enum TernOp {
    IMad { arg_t: ir::Type, spec: MulSpec },
    FFma { t: ir::Type, rounding: FpRounding },
}

impl fmt::Display for TernOp {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TernOp::IMad { arg_t, spec } => write!(fmt, "mad.{}.{}", spec, arg_t),
            TernOp::FFma { t, rounding } => write!(fmt, "fma.{}.{}", rounding, t),
        }
    }
}

impl TernOp {
    /// Create a new fma operator based on its rounding mode and argument types.
    ///
    /// # Errors
    ///
    /// Fails if a rounding mode is provided for an integer mad, or if the argument types are not
    /// compatible.
    pub fn from_ir_mad(
        rounding: ir::op::Rounding,
        mlhs_t: ir::Type,
        mrhs_t: ir::Type,
        arhs_t: ir::Type,
    ) -> Result<Self, InstructionError> {
        match mlhs_t {
            ir::Type::I(_) => {
                if rounding != ir::op::Rounding::Exact {
                    Err(InstructionError::invalid_rounding_for_type(
                        rounding, mlhs_t,
                    ))
                } else {
                    Ok(TernOp::IMad {
                        arg_t: mlhs_t,
                        spec: MulSpec::from_ir(mlhs_t, mrhs_t, arhs_t)?,
                    })
                }
            }
            ir::Type::F(_) => Ok(TernOp::FFma {
                t: Self::unify_ftype(None, [mlhs_t, mrhs_t, arhs_t])?,
                rounding: rounding.into(),
            }),
            // Type is not lowered
            _ => Err(InstructionError::invalid_type(mlhs_t)),
        }
    }

    /// Create a `imad` operator based on its destination and argument types.
    ///
    /// # Errors
    ///
    /// Fails if `d` and `c` are different types, or `a`, `b` and `c` are not compatible with
    /// `spec`.
    pub fn infer_imad(
        spec: MulSpec,
        d: Option<ir::Type>,
        [a, b, c]: [ir::Type; 3],
    ) -> Result<Self, InstructionError> {
        let arg_t = unify_itype(iter::once(a).chain(iter::once(b)))?;
        unify_itype(
            d.into_iter()
                .chain(iter::once(c))
                .chain(iter::once(spec.ret_t(arg_t))),
        )?;
        Ok(TernOp::IMad { arg_t, spec })
    }

    fn unify_ftype(
        d: Option<ir::Type>,
        abc: [ir::Type; 3],
    ) -> Result<ir::Type, TypeError> {
        unify_ftype(d.into_iter().chain(abc.iter().copied()))
    }

    /// Create a `ffma` operator based on its destination and argument types.
    ///
    /// # Errors
    ///
    /// Fails if `d`, `a`, `b` and `c` are not the same type.
    pub fn infer_ffma(
        rounding: FpRounding,
        d: Option<ir::Type>,
        abc: [ir::Type; 3],
    ) -> Result<Self, InstructionError> {
        Ok(Self::unify_ftype(d, abc).map(|t| TernOp::FFma { t, rounding })?)
    }

    /// The expected argument types for this operator.
    pub fn arg_t(self) -> [ir::Type; 3] {
        match self {
            TernOp::IMad { arg_t, spec } => [arg_t, arg_t, spec.ret_t(arg_t)],
            TernOp::FFma { t, .. } => [t, t, t],
        }
    }

    /// The resulting type when this operator is applied.
    pub fn ret_t(self) -> ir::Type {
        match self {
            TernOp::IMad { arg_t, spec } => spec.ret_t(arg_t),
            TernOp::FFma { t, .. } => t,
        }
    }
}

/// A (possibly vectorized) instruction to execute.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Instruction<'a> {
    Unary(UnOp, RegVec<'a>, [OpVec<'a>; 1]),
    Binary(BinOp, RegVec<'a>, [OpVec<'a>; 2]),
    Ternary(TernOp, RegVec<'a>, [OpVec<'a>; 3]),
    Load(LoadSpec, RegVec<'a>, Address<'a>),
    Store(StoreSpec, Address<'a>, [OpVec<'a>; 1]),
    Jump(Label<'a>),
    Sync,
}

impl fmt::Display for Instruction<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Instruction::*;

        match self {
            Unary(op, d, [a]) => write!(fmt, "{} = {}({})", d, op, a),
            Binary(op, d, [a, b]) => write!(fmt, "{} = {}({}, {})", d, op, a, b),
            Ternary(op, d, [a, b, c]) => {
                write!(fmt, "{} = {}({}, {}, {})", d, op, a, b, c)
            }
            Load(spec, d, a) => write!(fmt, "{} = {}({})", d, spec, a),
            Store(spec, a, [b]) => write!(fmt, "{}({}, {})", spec, a, b),
            Jump(label) => write!(fmt, "jump {}", label),
            Sync => write!(fmt, "sync"),
        }
    }
}

/// Helper macro to define instruction constructors.  See examples of use in the `impl Instruction`
/// block below.
macro_rules! new_instruction {
    () => {};

    (
        $inst:ident$([$($pre:ident: $preTy:ty),*])?(
            $d:ident
            $(, $a:ident)*
        ),
        $infer:path,
        $arity:ident;
        $($rest:tt)*
    ) => {
        #[allow(non_camel_case_types)]
        pub fn $inst<
            $d
            $(, $a)*
        >(
            $($($pre: $preTy, )*)?
            $d: $d
            $(, $a: $a)*
        ) -> Result<Self, InstructionError>
        where
            $d: Into<RegVec<'a>>,
            $($a: Into<OpVec<'a>>,)*
        {
            let $d = $d.into();
            $(let $a = $a.into();)*

            Self::$arity(
                $infer($($($pre ,)*)? Some($d.t()), [$($a.t()),*])?,
                $d
                $(, $a)*
            )
        }

        new_instruction!($($rest)*);
    };

    (
        $inst:ident$([$($pre:ident: $preTy:ty),*])?(
            $d:ident
            $(, $a:ident)*
        ) = $alias:ident$([$($aPre:expr),*])?;
        $($rest:tt)*
    ) => {
        #[allow(non_camel_case_types)]
        pub fn $inst<
            $d
            $(, $a)*
        >(
            $($($pre: $preTy, )*)?
            $d: $d
            $(, $a: $a)*
        ) -> Result<Self, InstructionError>
        where
            $d: Into<RegVec<'a>>,
            $($a: Into<OpVec<'a>>,)*
        {
            Self::$alias($($($aPre ,)*)? $d $(, $a)*)
        }

        new_instruction!($($rest)*);
    };
}

impl<'a> Instruction<'a> {
    /// Create a new unary instruction.
    ///
    /// # Errors
    ///
    /// Fails if the destination and argument types are not compatible with the provided operator.
    pub fn unary(
        op: UnOp,
        dest: RegVec<'a>,
        arg: OpVec<'a>,
    ) -> Result<Self, InstructionError> {
        if op.arg_t() != [arg.t()] || dest.t() != op.ret_t() {
            return Err(InstructionError::incompatible_types());
        }

        Ok(Instruction::Unary(op, dest, [arg]))
    }

    new_instruction! {
        mov(d, a), UnOp::infer_move, unary;
        cast[dst_t: ir::Type](d, a), UnOp::infer_cast, unary;
        exp(d, a), UnOp::infer_exp, unary;
    }

    /// Create a new binary instruction.
    ///
    /// # Errors
    ///
    /// Fails if the destination and argument types are not compatible with the provided operator.
    pub fn binary(
        op: BinOp,
        d: RegVec<'a>,
        a: OpVec<'a>,
        b: OpVec<'a>,
    ) -> Result<Self, InstructionError> {
        if op.arg_t() != [a.t(), b.t()] || op.ret_t() != d.t() {
            return Err(InstructionError::incompatible_types());
        }

        Ok(Instruction::Binary(op, d, [a, b]))
    }

    new_instruction! {
        iadd(d, a, b), BinOp::infer_iadd, binary;
        isub(d, a, b), BinOp::infer_isub, binary;
        imul_ex[spec: MulSpec](d, a, b), BinOp::infer_imul, binary;
        imul_low(d, a, b) = imul_ex[MulSpec::Low];
        imul_high(d, a, b) = imul_ex[MulSpec::High];
        imul_wide(d, a, b) = imul_ex[MulSpec::Wide];
        idiv(d, a, b), BinOp::infer_idiv, binary;

        fadd_ex[rounding: FpRounding](d, a, b), BinOp::infer_fadd, binary;
        fadd(d, a, b) = fadd_ex[FpRounding::NearestEven];
        fsub_ex[rounding: FpRounding](d, a, b), BinOp::infer_fsub, binary;
        fsub(d, a, b) = fsub_ex[FpRounding::NearestEven];
        fmul_ex[rounding: FpRounding](d, a, b), BinOp::infer_fmul, binary;
        fmul(d, a, b) = fmul_ex[FpRounding::NearestEven];
        fdiv_ex[rounding: FpRounding](d, a, b), BinOp::infer_fdiv, binary;
        fdiv(d, a, b) = fdiv_ex[FpRounding::NearestEven];
        fmax(d, a, b), BinOp::infer_fmax, binary;
        fmin(d, a, b), BinOp::infer_fmin, binary;

        set[op: CmpOp](d, a, b), BinOp::infer_set, binary;
        set_eq(d, a, b) = set[CmpOp::Eq];
        set_ne(d, a, b) = set[CmpOp::Ne];
        set_lt(d, a, b) = set[CmpOp::Lt];
        set_le(d, a, b) = set[CmpOp::Le];
        set_gt(d, a, b) = set[CmpOp::Gt];
        set_ge(d, a, b) = set[CmpOp::Ge];

        and(d, a, b), BinOp::infer_and, binary;
        xor(d, a, b), BinOp::infer_xor, binary;
        or(d, a, b), BinOp::infer_or, binary;
    }

    pub fn imul<D, A, B>(d: D, a: A, b: B) -> Result<Self, InstructionError>
    where
        D: Into<RegVec<'a>>,
        A: Into<OpVec<'a>>,
        B: Into<OpVec<'a>>,
    {
        let (d, a, b) = (d.into(), a.into(), b.into());
        Self::imul_ex(MulSpec::from_ir(a.t(), b.t(), d.t())?, d, a, b)
    }

    /// Create a new ternary instruction.
    ///
    /// # Errors
    ///
    /// Fails if the destination and argument types are not compatible with the provided operator.
    pub fn ternary(
        op: TernOp,
        d: RegVec<'a>,
        a: OpVec<'a>,
        b: OpVec<'a>,
        c: OpVec<'a>,
    ) -> Result<Self, InstructionError> {
        if op.arg_t() != [a.t(), b.t(), c.t()] || op.ret_t() != d.t() {
            return Err(InstructionError::incompatible_types());
        }

        Ok(Instruction::Ternary(op, d, [a, b, c]))
    }

    new_instruction! {
        imad_ex[spec: MulSpec](d, a, b, c), TernOp::infer_imad, ternary;
        imad_low(d, a, b, c) = imad_ex[MulSpec::Low];
        imad_high(d, a, b, c) = imad_ex[MulSpec::High];
        imad_wide(d, a, b, c) = imad_ex[MulSpec::Wide];

        ffma_ex[rounding: FpRounding](d, a, b, c), TernOp::infer_ffma, ternary;
        ffma(d, a, b, c) = ffma_ex[FpRounding::NearestEven];
    }

    pub fn imad<D, A, B, C>(d: D, a: A, b: B, c: C) -> Result<Self, InstructionError>
    where
        D: Into<RegVec<'a>>,
        A: Into<OpVec<'a>>,
        B: Into<OpVec<'a>>,
        C: Into<OpVec<'a>>,
    {
        let (a, b, c) = (a.into(), b.into(), c.into());
        Self::imad_ex(MulSpec::from_ir(a.t(), b.t(), c.t())?, d, a, b, c)
    }

    /// Create a new load instruction.
    pub fn load(
        spec: LoadSpec,
        d: RegVec<'a>,
        a: Address<'a>,
    ) -> Result<Self, InstructionError> {
        if spec.t() != d.t() {
            return Err(InstructionError::incompatible_types());
        }

        Ok(Instruction::Load(spec, d, a))
    }

    /// Create a new store instruction.
    pub fn store(
        spec: StoreSpec,
        a: Address<'a>,
        b: OpVec<'a>,
    ) -> Result<Self, InstructionError> {
        if spec.t() != b.t() {
            return Err(InstructionError::incompatible_types());
        }

        Ok(Instruction::Store(spec, a, [b]))
    }

    /// Create a new `jump` instruction.
    pub fn jump(label: Label<'a>) -> Self {
        Instruction::Jump(label)
    }

    /// Create a new `sync` instruction.
    pub fn sync() -> Self {
        Instruction::Sync
    }

    /// Create a new predicated instruction.
    ///
    /// This function takes a `Into<Option<Register<'a>>>` so that both
    /// `instruction.predicated(reg)` and `instruction.predicated(Some(reg))` (where `reg` is a
    /// register) are valid, as well as `instruction.predicated(None)`.
    pub fn predicated(
        self,
        predicate: impl Into<Option<Register<'a>>>,
    ) -> PredicatedInstruction<'a> {
        PredicatedInstruction {
            predicate: predicate.into(),
            instruction: self,
        }
    }
}

/// A predicated instruction, wrapping both an instruction and optional predicate.
///
/// The predicate must be a register to ensure that it can be efficiently printed in all backends.
pub struct PredicatedInstruction<'a> {
    pub predicate: Option<Register<'a>>,
    pub instruction: Instruction<'a>,
}

impl<'a> From<Instruction<'a>> for PredicatedInstruction<'a> {
    fn from(instruction: Instruction<'a>) -> Self {
        PredicatedInstruction {
            predicate: None,
            instruction,
        }
    }
}

impl fmt::Display for PredicatedInstruction<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            fmt,
            "{}{}",
            self.predicate
                .into_iter()
                .format_with("", |predicate, f| f(&format_args!("@{} ", predicate,))),
            self.instruction,
        )
    }
}

/// A (named) loop label
#[derive(Debug, Copy, Clone)]
pub struct Label<'a> {
    name: &'a str,
}

impl fmt::Display for Label<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{}:", self.name)
    }
}

impl<'a> Label<'a> {
    /// Create a new loop label
    pub fn new(name: &'a str) -> Self {
        Label { name }
    }

    /// The name of the label
    pub fn name(self) -> &'a str {
        self.name
    }
}

/// Rounding mode for floating-point instructions
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FpRounding {
    /// Mantissa LSB rounds to nearest even
    NearestEven,
    /// Mantissa LSB rounds towards zero
    Zero,
    /// Mantissa LSB rounds towards negative infinity
    NegativeInfinite,
    /// Mantissa LSB rounds towards positive infinity
    PositiveInfinite,
}

impl fmt::Display for FpRounding {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(match self {
            FpRounding::NearestEven => "rn",
            FpRounding::Zero => "rz",
            FpRounding::NegativeInfinite => "rm",
            FpRounding::PositiveInfinite => "rp",
        })
    }
}

impl From<ir::op::Rounding> for FpRounding {
    fn from(ir: ir::op::Rounding) -> Self {
        match ir {
            ir::op::Rounding::Exact | ir::op::Rounding::Nearest => {
                FpRounding::NearestEven
            }
            ir::op::Rounding::Zero => FpRounding::Zero,
            ir::op::Rounding::Positive => FpRounding::PositiveInfinite,
            ir::op::Rounding::Negative => FpRounding::NegativeInfinite,
        }
    }
}

/// Integer multiplication width specification
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MulSpec {
    /// Keep the low bits of the result
    Low,
    /// Keep the high bits of the result
    High,
    /// Keep all the bits of the result.  Result type is twice as wide as the arguments.
    Wide,
}

impl fmt::Display for MulSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(match self {
            MulSpec::Low => "lo",
            MulSpec::High => "hi",
            MulSpec::Wide => "wide",
        })
    }
}

impl MulSpec {
    /// Infer the appropriate specification based on the arguments and return types.
    ///
    /// # Errors
    ///
    /// Fails if the argument types are not identical, are not integer types, or the result type is
    /// not an integer type twice as wide as the argument types.
    pub fn from_ir(
        lhs_t: ir::Type,
        rhs_t: ir::Type,
        ret_t: ir::Type,
    ) -> Result<Self, MulSpecError> {
        if lhs_t != rhs_t {
            return Err(MulSpecError {
                inner: MulSpecErrorInner::IncompatibleArgumentTypes(lhs_t, rhs_t),
            });
        }

        Ok(match (lhs_t, ret_t) {
            (ir::Type::I(a), ir::Type::I(b)) if b == 2 * a => MulSpec::Wide,
            (ir::Type::I(a), ir::Type::I(b)) if a == b => MulSpec::Low,
            (ir::Type::I(_), _) => {
                return Err(MulSpecError {
                    inner: MulSpecErrorInner::InvalidReturnType(lhs_t, ret_t),
                })
            }
            _ => {
                return Err(MulSpecError {
                    inner: MulSpecErrorInner::ArgumentIsNotInteger(lhs_t),
                })
            }
        })
    }

    fn ret_t(self, arg_t: ir::Type) -> ir::Type {
        let bits = match arg_t {
            ir::Type::I(bits) => bits,
            _ => panic!("MulSpec: argument type is not integer"),
        };

        ir::Type::I(match self {
            MulSpec::Low | MulSpec::High => bits,
            MulSpec::Wide => bits * 2,
        })
    }
}

/// The error type returned when inference of a multiplication width specification fails.
#[derive(Debug, Clone)]
pub struct MulSpecError {
    inner: MulSpecErrorInner,
}

#[derive(Debug, Clone)]
enum MulSpecErrorInner {
    IncompatibleArgumentTypes(ir::Type, ir::Type),
    InvalidReturnType(ir::Type, ir::Type),
    ArgumentIsNotInteger(ir::Type),
}

impl fmt::Display for MulSpecError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use MulSpecErrorInner::*;

        match self.inner {
            IncompatibleArgumentTypes(lhs, rhs) => write!(
                fmt,
                "integer multiplication: arguments have different types (`{}` and `{}`)",
                lhs, rhs
            ),
            InvalidReturnType(arg, ret) => write!(
                fmt,
                "integer multiplication: invalid return type `{}` for argument type `{}`",
                ret, arg,
            ),
            ArgumentIsNotInteger(t) => write!(
                fmt,
                "integer multiplication: argument type `{}` is not an integer type",
                t
            ),
        }
    }
}

impl error::Error for MulSpecError {}

/// Load instruction specification
///
/// This contains information about the type of values loaded, the state space from which the value
/// is loaded, the cache behavior to use, and a potential vectorization factor.
#[derive(Debug, Copy, Clone)]
pub struct LoadSpec {
    t: ir::Type,
    vec: NonZeroU32,
    ss: StateSpace,
    cop: LoadCacheOperator,
}

impl fmt::Display for LoadSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "load.{}.{}", self.ss, self.cop)?;
        if self.vec.get() > 1 {
            write!(fmt, ".v{}", self.vec)?;
        }
        write!(fmt, ".{}", self.t)
    }
}

impl LoadSpec {
    /// The state space from which the load is performed.
    pub fn state_space(self) -> StateSpace {
        self.ss
    }

    /// The cache behavior to use
    pub fn cache_operator(self) -> LoadCacheOperator {
        self.cop
    }

    /// The vectorization factor
    pub fn vector_factor(self) -> NonZeroU32 {
        self.vec
    }

    /// The type of values loaded
    pub fn t(self) -> ir::Type {
        self.t
    }

    /// Create the appropriate load specification based on the IR information.
    ///
    /// # Errors
    ///
    /// Fails if an outer vectorization factor is provided (this is currently not supported), and
    /// if the memory space or instruction flags are invalid or not fully specified.
    pub fn from_ir(
        vector_factors: [u32; 2],
        t: ir::Type,
        mem_space: MemSpace,
        inst_flag: InstFlag,
    ) -> Result<Self, InstructionError> {
        if vector_factors[0] != 1 {
            return Err(InstructionError::invalid_vector_factors(vector_factors));
        }

        if let Some(vec) = NonZeroU32::new(vector_factors[1]) {
            Ok(LoadSpec {
                t,
                vec,
                ss: mem_space.try_into()?,
                cop: inst_flag.try_into()?,
            })
        } else {
            Err(InstructionError::invalid_vector_factors(vector_factors))
        }
    }
}

/// Store instruction specification
///
/// This contains information about the type of values to store, the state space to store the value
/// into, the cache behavior to use, and a potential vectorization factor.
#[derive(Debug, Copy, Clone)]
pub struct StoreSpec {
    t: ir::Type,
    vec: NonZeroU32,
    ss: StateSpace,
    cop: StoreCacheOperator,
}

impl fmt::Display for StoreSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "store.{}.{}", self.ss, self.cop)?;
        if self.vec.get() > 1 {
            write!(fmt, ".v{}", self.vec)?;
        }
        write!(fmt, ".{}", self.t)
    }
}

impl StoreSpec {
    /// The type of values stored
    pub fn t(self) -> ir::Type {
        self.t
    }

    /// The state space into which the store is performed
    pub fn state_space(self) -> StateSpace {
        self.ss
    }

    /// The cache behavior to use
    pub fn cache_operator(self) -> StoreCacheOperator {
        self.cop
    }

    /// The vectorization factor
    pub fn vector_factor(self) -> NonZeroU32 {
        self.vec
    }

    /// Create the appropriate store specification based on the IR information.
    ///
    /// # Errors
    ///
    /// Fails if an outer vectorization factor is provided (this is currently not supported), and
    /// if the memory space or instruction flags are invalid or not fully specified.
    pub fn from_ir(
        vector_factors: [u32; 2],
        t: ir::Type,
        mem_space: MemSpace,
        inst_flag: InstFlag,
    ) -> Result<Self, InstructionError> {
        if vector_factors[0] != 1 {
            return Err(InstructionError::invalid_vector_factors(vector_factors));
        }

        if let Some(vec) = NonZeroU32::new(vector_factors[1]) {
            Ok(StoreSpec {
                t,
                vec,
                ss: mem_space.try_into()?,
                cop: inst_flag.try_into()?,
            })
        } else {
            Err(InstructionError::invalid_vector_factors(vector_factors))
        }
    }
}

/// Represent a state space, i.e. a storage area with particular characteristics.
///
/// The only guaranteed state space across all architectures is the `Global` space.
#[derive(Debug, Copy, Clone)]
pub enum StateSpace {
    /// Global memory, shared by all threads
    Global,
    /// Addressable memory shared between threads in the same block
    Shared,
}

impl fmt::Display for StateSpace {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(match self {
            StateSpace::Global => "global",
            StateSpace::Shared => "shared",
        })
    }
}

impl TryFrom<MemSpace> for StateSpace {
    type Error = UnconstrainedCandidateError;

    fn try_from(mem_space: MemSpace) -> Result<Self, Self::Error> {
        Ok(match mem_space {
            MemSpace::GLOBAL => StateSpace::Global,
            MemSpace::SHARED => StateSpace::Shared,
            _ => return Err(UnconstrainedCandidateError::new(&"MemSpace", &mem_space)),
        })
    }
}

/// The error type returned when an unconstrained value is encountered.
#[derive(Debug, Clone)]
pub struct UnconstrainedCandidateError(String, String);

impl UnconstrainedCandidateError {
    fn new(what: &dyn fmt::Display, values: &dyn fmt::Display) -> Self {
        UnconstrainedCandidateError(what.to_string(), values.to_string())
    }
}

impl fmt::Display for UnconstrainedCandidateError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            fmt,
            "unconstrained value of type {} was encountered: {}",
            self.0, self.1,
        )
    }
}

impl error::Error for UnconstrainedCandidateError {}

/// Cache operators for memory load instructions
#[derive(Debug, Copy, Clone)]
pub enum LoadCacheOperator {
    /// Cache at all levels.
    CacheAll,
    /// Cache at the global level
    CacheGlobal,
    /// Cache streaming, likely to be accessed once
    CacheStreaming,
    /// Cache at all levels, and also at the texture cache level.
    /// This looks somewhat weird, but it matches the behavior of the `ld.global.nc`.PTX
    /// instructions we generate.
    CacheAllAndTexture,
}

impl Default for LoadCacheOperator {
    fn default() -> Self {
        LoadCacheOperator::CacheAll
    }
}

impl fmt::Display for LoadCacheOperator {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use LoadCacheOperator::*;

        fmt.write_str(match self {
            CacheAll => "ca",
            CacheGlobal => "cg",
            CacheStreaming => "cs",
            CacheAllAndTexture => "nc",
        })
    }
}

impl TryFrom<InstFlag> for LoadCacheOperator {
    type Error = UnconstrainedCandidateError;

    fn try_from(inst_flag: InstFlag) -> Result<Self, Self::Error> {
        use LoadCacheOperator::*;

        Ok(match inst_flag {
            InstFlag::NO_CACHE => CacheStreaming,
            InstFlag::CACHE_GLOBAL => CacheGlobal,
            InstFlag::CACHE_SHARED => CacheAll,
            InstFlag::CACHE_READ_ONLY => CacheAllAndTexture,
            _ => return Err(UnconstrainedCandidateError::new(&"InstFlag", &inst_flag)),
        })
    }
}

/// Cache operators for memory store instructions
#[derive(Debug, Copy, Clone)]
pub enum StoreCacheOperator {
    /// Cache write-back all coherent levels
    WriteBack,
    /// Cache at global level
    CacheGlobal,
    /// Cache streaming, likely to be accessed once
    CacheStreaming,
}

impl Default for StoreCacheOperator {
    fn default() -> Self {
        StoreCacheOperator::WriteBack
    }
}

impl fmt::Display for StoreCacheOperator {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use StoreCacheOperator::*;

        fmt.write_str(match self {
            WriteBack => "wb",
            CacheGlobal => "cg",
            CacheStreaming => "cs",
        })
    }
}

impl TryFrom<InstFlag> for StoreCacheOperator {
    type Error = UnconstrainedCandidateError;

    fn try_from(inst_flag: InstFlag) -> Result<Self, Self::Error> {
        use StoreCacheOperator::*;

        Ok(match inst_flag {
            InstFlag::NO_CACHE => CacheStreaming,
            InstFlag::CACHE_GLOBAL => CacheGlobal,
            InstFlag::CACHE_SHARED => WriteBack,
            // TODO: InstFlag::CACHE_READ_ONLY is not "unconstrained" but invalid still
            _ => return Err(UnconstrainedCandidateError::new(&"InstFlag", &inst_flag)),
        })
    }
}

/// The error type returned when type errors are encountered.
#[derive(Debug, Clone)]
pub struct TypeError {
    inner: TypeErrorInner,
}

impl TypeError {
    fn mismatch(a: ir::Type, b: ir::Type) -> Self {
        TypeErrorInner::Mismatch(a, b).into()
    }

    fn not_integer(t: ir::Type) -> Self {
        TypeErrorInner::NotInteger(t).into()
    }

    fn not_float(t: ir::Type) -> Self {
        TypeErrorInner::NotFloat(t).into()
    }
}

#[derive(Debug, Clone)]
enum TypeErrorInner {
    Mismatch(ir::Type, ir::Type),
    NotInteger(ir::Type),
    NotFloat(ir::Type),
}

impl From<TypeErrorInner> for TypeError {
    fn from(inner: TypeErrorInner) -> Self {
        TypeError { inner }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TypeErrorInner::*;

        match self.inner {
            Mismatch(lhs, rhs) => {
                write!(fmt, "got incompatible types: `{}` and `{}`", lhs, rhs)
            }
            NotInteger(t) => write!(fmt, "expected an integer type; got `{}`", t),
            NotFloat(t) => write!(fmt, "expected a float type; got `{}`", t),
        }
    }
}

impl error::Error for TypeError {}

#[derive(Debug, Clone)]
pub struct InstructionError {
    inner: InstructionErrorInner,
}

impl fmt::Display for InstructionError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InstructionErrorInner::*;

        match &self.inner {
            InvalidRoundingForType(rounding, t) => {
                write!(fmt, "got unexpected rounding {} for type {}", rounding, t)
            }
            InvalidBinopForType(op, t) => write!(
                fmt,
                "got unexpected operator {} with argument type {}",
                op, t
            ),
            InvalidRoundingForOp(op, rounding) => {
                write!(fmt, "got unexpected rounding {} for op {}", rounding, op)
            }
            InvalidType(t) => write!(fmt, "got unexpected type {}", t),
            IncompatibleTypes => write!(fmt, "got incompatible types"),
            InvalidVectorFactors(factors) => write!(
                fmt,
                "got unexpected vectorization factors: {}x{}",
                factors[0], factors[1]
            ),
            TypeError(err) => fmt::Display::fmt(err, fmt),
            MulSpecError(err) => fmt::Display::fmt(err, fmt),
            UnconstrainedCandidateError(err) => fmt::Display::fmt(err, fmt),
        }
    }
}

impl error::Error for InstructionError {}

#[derive(Debug, Clone)]
enum InstructionErrorInner {
    InvalidRoundingForType(ir::op::Rounding, ir::Type),
    InvalidBinopForType(ir::BinOp, ir::Type),
    InvalidRoundingForOp(ir::BinOp, ir::op::Rounding),
    InvalidType(ir::Type),
    IncompatibleTypes,
    InvalidVectorFactors([u32; 2]),
    TypeError(TypeError),
    MulSpecError(MulSpecError),
    UnconstrainedCandidateError(UnconstrainedCandidateError),
}

impl From<InstructionErrorInner> for InstructionError {
    fn from(inner: InstructionErrorInner) -> Self {
        InstructionError { inner }
    }
}

impl InstructionError {
    fn invalid_rounding_for_type(rounding: ir::op::Rounding, t: ir::Type) -> Self {
        InstructionErrorInner::InvalidRoundingForType(rounding, t).into()
    }

    fn invalid_rounding_for_op(op: ir::BinOp, rounding: ir::op::Rounding) -> Self {
        InstructionErrorInner::InvalidRoundingForOp(op, rounding).into()
    }

    fn invalid_binop_for_type(op: ir::BinOp, t: ir::Type) -> Self {
        InstructionErrorInner::InvalidBinopForType(op, t).into()
    }

    fn invalid_type(t: ir::Type) -> Self {
        InstructionErrorInner::InvalidType(t).into()
    }

    fn incompatible_types() -> Self {
        InstructionErrorInner::IncompatibleTypes.into()
    }

    fn invalid_vector_factors(vector_factors: [u32; 2]) -> Self {
        InstructionErrorInner::InvalidVectorFactors(vector_factors).into()
    }
}

impl From<TypeError> for InstructionError {
    fn from(error: TypeError) -> Self {
        InstructionErrorInner::TypeError(error).into()
    }
}

impl From<MulSpecError> for InstructionError {
    fn from(error: MulSpecError) -> Self {
        InstructionErrorInner::MulSpecError(error).into()
    }
}

impl From<UnconstrainedCandidateError> for InstructionError {
    fn from(error: UnconstrainedCandidateError) -> Self {
        InstructionErrorInner::UnconstrainedCandidateError(error).into()
    }
}
