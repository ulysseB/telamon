//! This crate defines the `C99Display` trait which should be used to display constructs in
//! a C99-compatible syntax, and implements it for the various `llir` constructs.
//!
//! In the future, it should define additional shared constructs for C-based backends such as the
//! x86 and MPPA backends.

use std::fmt;

use telamon::codegen::llir;
use telamon::ir;

/// Formatting trait for C99 values.
///
/// This is similar to the standard library's `Display` trait, except that it prints values in a
/// syntax compatible with the C99 standard.
pub trait C99Display {
    /// Formats the value using the given formatter.
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result;

    /// Helper function to wrap `self` into a `Display` implementation which will call back into
    /// `C99Display::fmt`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use std::fmt;
    ///
    /// impl C99Display for String {
    ///     fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(fmt, "\"{}\"", self.escape_default())
    ///     }
    /// }
    ///
    /// assert_eq!(r#"x"y"#.c99().to_string(), r#""x\"y""#);
    /// ```
    fn c99(&self) -> DisplayC99<'_, Self> {
        DisplayC99 { inner: self }
    }
}

/// Helper struct for printing values in C99 syntax.
///
/// This `struct` implements the `Display` trait by using a `C99Display` implementation. It is
/// created by the `c99` method on `C99Display` instances.
pub struct DisplayC99<'a, T: ?Sized> {
    inner: &'a T,
}

impl<T: fmt::Debug + ?Sized> fmt::Debug for DisplayC99<'_, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.inner, fmt)
    }
}

impl<T: C99Display + ?Sized> fmt::Display for DisplayC99<'_, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        C99Display::fmt(self.inner, fmt)
    }
}

impl C99Display for llir::Register<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(self.name())
    }
}

impl C99Display for llir::Operand<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::Operand::*;

        match self {
            Register(register) => C99Display::fmt(register, fmt),
            &IntLiteral(ref val, bits) => {
                assert!(bits <= 64);
                fmt::Display::fmt(val, fmt)
            }
            &FloatLiteral(ref val, bits) => {
                use num::{Float, ToPrimitive};

                assert!(bits <= 64);
                let f = val.numer().to_f64().unwrap() / val.denom().to_f64().unwrap();

                // Print in C99 hexadecimal floating point representation
                let (mantissa, exponent, sign) = f.integer_decode();
                let signchar = if sign < 0 { "-" } else { "" };

                // Assume that floats and doubles in the C implementation have
                // 32 and 64 bits, respectively
                let floating_suffix = match bits {
                    32 => "f",
                    64 => "",
                    _ => panic!("Cannot print floating point value with {} bits", bits),
                };

                write!(
                    fmt,
                    "{}0x{:x}p{}{}",
                    signchar, mantissa, exponent, floating_suffix
                )
            }
        }
    }
}

impl<T: C99Display> C99Display for llir::ScalarOrVector<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            llir::ScalarOrVector::Scalar(scalar) => C99Display::fmt(scalar, fmt),
            llir::ScalarOrVector::Vector(..) => {
                panic!("x86 backend does not support vectors.")
            }
        }
    }
}

impl C99Display for llir::UnOp {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::UnOp;

        match self {
            UnOp::Move { .. } => Ok(()),
            UnOp::Cast { dst_t, .. } => write!(fmt, "({})", dst_t.c99()),
            UnOp::Exp { t: ir::Type::F(32) } => write!(fmt, "expf"),
            UnOp::Exp { .. } => panic!("{}: non-atomic C99 instruction", self),
        }
    }
}

fn is_infix(binop: llir::BinOp) -> bool {
    use llir::BinOp::*;

    match binop {
        // Integer Arithmetic Instructions
        IAdd { .. }
        | ISub { .. }
        | IDiv { .. }
        | IMul {
            spec: llir::MulSpec::Low,
            ..
        }
        | FAdd { .. }
        | FSub { .. }
        | FMul { .. }
        | FDiv { .. }
        | Set { .. }
        | And { .. }
        | Or { .. }
        | Xor { .. } => true,
        _ => false,
    }
}

impl C99Display for llir::BinOp {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::BinOp::*;

        match self {
            // Integer Arithmetic Instructions
            IAdd { .. } => write!(fmt, "+"),
            ISub { .. } => write!(fmt, "-"),
            IDiv { .. } => write!(fmt, "/"),
            IMul {
                spec: llir::MulSpec::Low,
                ..
            } => write!(fmt, "*"),
            IMul { spec, arg_t } => {
                write!(fmt, "__mul{}{}", arg_t.bitwidth().unwrap(), spec.c99())
            }
            IMax { .. } => write!(fmt, "__max"),
            // Floating-Point Instructions
            FAdd { .. } => write!(fmt, "+"),
            FSub { .. } => write!(fmt, "-"),
            FMul { .. } => write!(fmt, "*"),
            FDiv { .. } => write!(fmt, "/"),
            FMax { .. } => write!(fmt, "__max"),
            FMin { .. } => write!(fmt, "__min"),
            // Comparison and Selection Instructions
            Set { op, .. } => write!(fmt, "{}", op.c99()),
            // Logic and Shift Instructions
            And { .. } => write!(fmt, "&"),
            Or { .. } => write!(fmt, "|"),
            Xor { .. } => write!(fmt, "^"),
        }
    }
}

impl C99Display for llir::TernOp {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::TernOp::*;

        match self {
            IMad { spec, arg_t } => {
                write!(fmt, "__mad{}{}", arg_t.bitwidth().unwrap(), spec.c99())
            }
            FFma { .. } => write!(fmt, "__fma"),
        }
    }
}

impl C99Display for llir::CmpOp {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(match self {
            llir::CmpOp::Eq => "==",
            llir::CmpOp::Ne => "!=",
            llir::CmpOp::Lt => "<",
            llir::CmpOp::Le => "<=",
            llir::CmpOp::Gt => ">",
            llir::CmpOp::Ge => ">=",
        })
    }
}

impl C99Display for llir::MulSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(match self {
            llir::MulSpec::Low => "",
            llir::MulSpec::High => "Hi",
            llir::MulSpec::Wide => "Wide",
        })
    }
}

impl C99Display for llir::Address<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::Address;

        match *self {
            Address::Register(reg, offset) if offset == 0 => write!(fmt, "{}", reg),
            Address::Register(reg, offset) => write!(fmt, "{}{:+#x}", reg, offset),
        }
    }
}

impl C99Display for llir::Label<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{}:", self.name())
    }
}

impl C99Display for llir::LoadSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "({}*)", self.t().c99())
    }
}

impl C99Display for llir::StoreSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "({}*)", self.t().c99())
    }
}

impl C99Display for llir::PredicatedInstruction<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(predicate) = self.predicate {
            write!(fmt, "if ({}) ", predicate.c99())?;
        }

        write!(fmt, "{};", self.instruction.c99())
    }
}

impl C99Display for llir::Instruction<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::Instruction::*;

        match self {
            Unary(op, d, [a]) => write!(
                fmt,
                "{d} = {op}({a})",
                op = op.c99(),
                d = d.c99(),
                a = a.c99()
            ),
            Binary(op, d, [a, b]) if is_infix(*op) => write!(
                fmt,
                "{d} = {a} {op} {b}",
                op = op.c99(),
                d = d.c99(),
                a = a.c99(),
                b = b.c99()
            ),
            Binary(op, d, [a, b]) => write!(
                fmt,
                "{d} = {op}({a}, {b})",
                op = op.c99(),
                d = d.c99(),
                a = a.c99(),
                b = b.c99()
            ),
            Ternary(op, d, [a, b, c]) => write!(
                fmt,
                "{d} = {op}({a}, {b}, {c})",
                op = op.c99(),
                d = d.c99(),
                a = a.c99(),
                b = b.c99(),
                c = c.c99()
            ),
            Load(spec, d, a) => write!(
                fmt,
                "{d} = *{spec}({a})",
                spec = spec.c99(),
                d = d.c99(),
                a = a.c99()
            ),
            Store(spec, a, [b]) => write!(
                fmt,
                "*{spec}({a}) = {b}",
                spec = spec.c99(),
                a = a.c99(),
                b = b.c99()
            ),
            Jump(label) => write!(fmt, "goto {label}", label = label.name()),
            Sync => write!(fmt, "__sync()"),
        }
    }
}

impl C99Display for ir::Type {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(match self {
            ir::Type::PtrTo(..) => "intptr_t",
            ir::Type::F(32) => "float",
            ir::Type::F(64) => "double",
            ir::Type::I(1) => "int8_t",
            ir::Type::I(8) => "int8_t",
            ir::Type::I(16) => "int16_t",
            ir::Type::I(32) => "int32_t",
            ir::Type::I(64) => "int64_t",
            _ => panic!("invalid C99 type: {}", self),
        })
    }
}
