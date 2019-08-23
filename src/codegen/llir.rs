use std::borrow::Cow;

use num::bigint::BigInt;
use num::rational::Ratio;

use crate::ir;

#[derive(Copy, Clone)]
pub struct Register<'a> {
    name: &'a str,
    t: ir::Type,
}

impl<'a> Register<'a> {
    pub fn new(name: &'a str, t: ir::Type) -> Self {
        Register { name, t }
    }

    pub fn name(self) -> &'a str {
        self.name
    }

    pub fn t(self) -> ir::Type {
        self.t
    }

    pub fn into_operand(self) -> Operand<'a> {
        Operand::Register(self)
    }
}

#[derive(Clone)]
pub enum Operand<'a> {
    Register(Register<'a>),
    IntLiteral(Cow<'a, BigInt>, u16),
    FloatLiteral(Cow<'a, Ratio<BigInt>>, u16),
}

impl<'a> From<Register<'a>> for Operand<'a> {
    fn from(register: Register<'a>) -> Operand<'a> {
        Operand::Register(register)
    }
}

impl<'a> From<&'_ Register<'a>> for Operand<'a> {
    fn from(register: &'_ Register<'a>) -> Operand<'a> {
        Operand::Register(*register)
    }
}

impl<'a> Operand<'a> {
    pub fn int<T: ir::IntLiteral<'a>>(value: T) -> Self {
        let (value, bits) = value.decompose();
        Operand::IntLiteral(value, bits)
    }

    pub fn float<T: ir::FloatLiteral<'a>>(value: T) -> Self {
        let (value, bits) = value.decompose();
        Operand::FloatLiteral(value, bits)
    }

    pub fn to_register(&self) -> Option<Register<'a>> {
        match *self {
            Operand::Register(register) => Some(register),
            _ => None,
        }
    }

    pub fn t(&self) -> ir::Type {
        match *self {
            Operand::Register(register) => register.t(),
            Operand::IntLiteral(_, bits) => ir::Type::I(bits),
            Operand::FloatLiteral(_, bits) => ir::Type::F(bits),
        }
    }
}

#[derive(Clone)]
pub enum Vector<T> {
    Scalar(T),
    Vector(Vec<T>),
}

pub type VRegister<'a> = Vector<Register<'a>>;

pub type VOperand<'a> = Vector<Operand<'a>>;

pub trait IntoScalar: Sized {
    fn into_scalar(self) -> Vector<Self>;
}

impl<T> IntoScalar for T {
    fn into_scalar(self) -> Vector<Self> {
        Vector::Scalar(self)
    }
}

pub trait IntoVector<T>: Sized {
    fn into_vector(self) -> Vector<T>;
}

impl<T: IntoScalar> IntoVector<T> for T {
    fn into_vector(self) -> Vector<T> {
        self.into_scalar()
    }
}

impl<T> IntoVector<T> for Vector<T> {
    fn into_vector(self) -> Vector<T> {
        self
    }
}
