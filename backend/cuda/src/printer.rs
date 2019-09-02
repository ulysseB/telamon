//! Provides functions to print PTX code.
use std::fmt::{self, Write as WriteFmt};
use std::io::Write;

use utils::*;

use itertools::Itertools;
use telamon::codegen::*;
use telamon::ir::{self, Type};
use telamon::search_space::{DimKind, Domain};

use crate::{Gpu, NameGenerator};

/// Formatting trait for PTX values.
///
/// This is similar to the standard library's `Display` trait, except that it prints values in PTX
/// syntax.
pub trait PTXDisplay {
    /// Formats the value using the given formatter.
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result;

    /// Helper function to wrap `self` into a `Display` implementation which will call back into
    /// `PTXDisplay::fmt`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use std::fmt;
    ///
    /// impl PTXDisplay for String {
    ///     fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(fmt, "\"{}\"", self.escape_default())
    ///     }
    /// }
    ///
    /// assert_eq!(r#"x"y"#.ptx().to_string(), r#""x\"y""#);
    /// ```
    fn ptx(&self) -> DisplayPTX<'_, Self> {
        DisplayPTX { inner: self }
    }
}

/// Helper struct for printing values in PTX syntax.
///
/// This `struct` implements the `Display` trait by using a `PTXDisplay` implementation. It is
/// created by the `ptx` method on `PTXDisplay` instances.
pub struct DisplayPTX<'a, T: ?Sized> {
    inner: &'a T,
}

impl<'a, T: fmt::Debug + ?Sized> fmt::Debug for DisplayPTX<'a, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.inner, fmt)
    }
}

impl<'a, T: PTXDisplay + ?Sized> fmt::Display for DisplayPTX<'a, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        PTXDisplay::fmt(self.inner, fmt)
    }
}

impl PTXDisplay for llir::Register<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "%{}", self.name())
    }
}

impl PTXDisplay for llir::Operand<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::Operand::*;

        match self {
            Register(register) => write!(fmt, "{}", register.ptx()),
            &IntLiteral(ref val, bits) => {
                assert!(bits <= 64);
                fmt::Display::fmt(val, fmt)
            }
            &FloatLiteral(ref val, bits) => {
                use num::ToPrimitive;
                assert!(bits <= 64);

                write!(
                    fmt,
                    "0D{:016X}",
                    (val.numer().to_f64().unwrap() / val.denom().to_f64().unwrap())
                        .to_bits()
                )
            }
        }
    }
}

impl<T: PTXDisplay> PTXDisplay for llir::ScalarOrVector<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            llir::ScalarOrVector::Scalar(scalar) => PTXDisplay::fmt(scalar, fmt),
            llir::ScalarOrVector::Vector(scalars) => write!(
                fmt,
                "{{{}}}",
                scalars.iter().map(PTXDisplay::ptx).format(", ")
            ),
        }
    }
}

#[derive(Default)]
pub(crate) struct CudaPrinter {
    buffer: String,
}

impl CudaPrinter {
    /// Prints the variables declared by the `NameGenerator`.
    fn var_decls(&mut self, namegen: &NameGenerator) -> String {
        let print_decl = |(&t, n)| {
            let prefix = NameGenerator::gen_prefix(t);
            format!(".reg.{} %{}<{}>;", t.ptx(), prefix, n)
        };
        namegen
            .num_var
            .iter()
            .map(print_decl)
            .collect_vec()
            .join("\n  ")
    }

    /// Declares block and thread indexes.
    fn decl_par_indexes(function: &Function, name_map: &mut NameMap<'_>) -> String {
        let mut decls = vec![];
        // Load block indexes.
        for (dim, dir) in function.block_dims().iter().zip(&["x", "y", "z"]) {
            let index = name_map.name_index(dim.id());
            decls.push(format!("mov.u32 {}, %ctaid.{};", index.ptx(), dir));
        }
        // Compute thread indexes.
        for (dim, dir) in function.thread_dims().iter().rev().zip(&["x", "y", "z"]) {
            decls.push(format!(
                "mov.s32 {}, %tid.{};",
                name_map.name_index(dim.id()).ptx(),
                dir
            ));
        }
        decls.join("\n  ")
    }

    /// Declares a shared memory block.
    fn shared_mem_decl(&mut self, block: &MemoryRegion, name_map: &mut NameMap<'_>) {
        unwrap!(writeln!(
            self.buffer,
            "\
  .shared.align 16 .u8 %shmem{id}[{size}];
  mov.u32 {name}, %shmem{id};",
            id = block.id().0,
            name = name_map.name_addr(block.id()).ptx(),
            size = unwrap!(block.alloc_size().as_int())
        ));
    }

    /// Prints a `Type` for the host.
    fn host_type(t: Type) -> &'static str {
        match t {
            Type::PtrTo(..) => "CUdeviceptr",
            Type::F(32) => "float",
            Type::F(64) => "double",
            Type::I(8) => "int8_t",
            Type::I(16) => "int16_t",
            Type::I(32) => "int32_t",
            Type::I(64) => "int64_t",
            ref t => panic!("invalid type for the host: {}", t),
        }
    }

    /// Returns the string representation of thread and block sizes on the host.
    fn host_3sizes<'a, IT>(dims: IT) -> [String; 3]
    where
        IT: Iterator<Item = &'a Dimension<'a>> + 'a,
    {
        let mut sizes = ["1".to_string(), "1".to_string(), "1".to_string()];
        for (i, d) in dims.enumerate() {
            assert!(i < 3);
            sizes[i] = Self::host_size(d.size())
        }
        sizes
    }

    /// Prints a size on the host.
    fn host_size(size: &Size) -> String {
        let dividend = size.dividend().iter().map(|p| format!("* {}", &p.name));
        format!(
            "{}{}/{}",
            size.factor(),
            dividend.format(""),
            size.divisor()
        )
    }

    /// Prints a parameter declaration.
    fn param_decl(&mut self, param: &ParamVal) -> String {
        format!(
            ".param .{t} {name}",
            t = param.t().ptx(),
            name = param.key().ident(),
        )
    }

    /// Prints a `Function`.
    pub fn function(&mut self, function: &Function, gpu: &Gpu) -> String {
        let mut namegen = NameGenerator::default();
        let interner = Interner::default();
        let name_map = &mut NameMap::new(&interner, function, &mut namegen);
        let param_decls = function
            .device_code_args()
            .map(|v| self.param_decl(v))
            .join(",\n  ");
        // LOAD PARAMETERS
        for val in function.device_code_args() {
            unwrap!(writeln!(
                self.buffer,
                "  ld.param.{t} {var_name}, [{name}];",
                t = val.t().ptx(),
                var_name = name_map.name_param_val(val.key()).ptx(),
                name = val.key().ident(),
            ));
        }
        // INDEX LOAD
        self.buffer.push_str(&"  ");
        let idx_loads = Self::decl_par_indexes(function, name_map);
        self.buffer.push_str(&idx_loads);
        self.buffer.push_str(&"\n");
        //MEM DECL
        for block in function.mem_blocks() {
            match block.alloc_scheme() {
                AllocationScheme::Shared => self.shared_mem_decl(block, name_map),
                AllocationScheme::PrivatisedGlobal => {
                    Printer::new(self, name_map).privatise_global_block(block, function)
                }
                AllocationScheme::Global => (),
            }
        }
        // Compute size casts
        for dim in function.dimensions() {
            if !dim.kind().intersects(DimKind::UNROLL | DimKind::LOOP) {
                continue;
            }
            for level in dim.induction_levels() {
                if let Some((_, ref incr)) = level.increment {
                    let reg = name_map.declare_size_cast(incr, level.t());
                    if let Some(reg) = reg {
                        let old_name = name_map.name_size(incr, Type::I(32));
                        self.print_inst(
                            llir::Instruction::cast(level.t(), reg, old_name)
                                .unwrap()
                                .into(),
                        );
                    }
                }
            }
        }
        let ind_levels = function.init_induction_levels().iter().chain(
            function
                .block_dims()
                .iter()
                .flat_map(|d| d.induction_levels()),
        );
        for level in ind_levels {
            Printer::new(self, name_map).parallel_induction_level(level);
        }
        Printer::new(self, name_map).cfg(function, function.cfg());
        let var_decls = self.var_decls(&namegen);
        let mut body = String::new();
        body.push_str(&var_decls);
        body.push_str(&"\n");
        body.push_str(&self.buffer);
        format!(
            include_str!("template/device.ptx"),
            sm_major = gpu.sm_major,
            sm_minor = gpu.sm_minor,
            addr_size = gpu.addr_size,
            name = function.name(),
            params = param_decls,
            num_thread = function.num_threads(),
            body = body
        )
    }

    pub fn host_function(&mut self, fun: &Function, gpu: &Gpu, out: &mut dyn Write) {
        let block_sizes = Self::host_3sizes(fun.block_dims().iter());
        let thread_sizes = Self::host_3sizes(fun.thread_dims().iter().rev());
        let extern_param_names = fun
            .space()
            .ir_instance()
            .signature()
            .params
            .iter()
            .map(|x| &x.name as &str)
            .collect_vec()
            .join(", ");
        let mut extra_def = vec![];
        let mut extra_cleanup = vec![];
        let params = fun
            .device_code_args()
            .map(|p| match *p {
                ParamVal::External(ref p, _) => format!("&{}", p.name),
                ParamVal::Size(ref size) => {
                    extra_def.push(format!(
                        "int32_t {} = {};",
                        p.key().ident(),
                        Self::host_size(size)
                    ));
                    format!("&{}", p.key().ident())
                }
                ParamVal::GlobalMem(_, ref size, _) => {
                    let size = Self::host_size(size);
                    extra_def.push(format!("CUDeviceptr {};", p.key().ident()));
                    extra_def.push(format!(
                        "CHECK_CUDA(cuMemAlloc(&{}, {}));",
                        p.key().ident(),
                        size
                    ));
                    extra_cleanup
                        .push(format!("CHECK_CUDA(cuMemFree({}));", p.key().ident()));
                    format!("&{}", p.key().ident())
                }
            })
            .collect_vec()
            .join(", ");
        let extern_params = fun
            .space()
            .ir_instance()
            .signature()
            .params
            .iter()
            .map(|p| format!("{} {}", Self::host_type(p.t), p.name))
            .collect_vec()
            .join(", ");
        let res = write!(
            out,
            include_str!("template/host.c"),
            name = fun.name(),
            ptx_code = self.function(fun, gpu).replace("\n", "\\n\\\n"),
            extern_params = extern_params,
            extern_param_names = extern_param_names,
            param_vec = format!("{{ {} }}", params),
            extra_def = extra_def.join("  \n"),
            extra_cleanup = extra_cleanup.join("  \n"),
            t_dim_x = &thread_sizes[0],
            t_dim_y = &thread_sizes[1],
            t_dim_z = &thread_sizes[2],
            b_dim_x = &block_sizes[0],
            b_dim_y = &block_sizes[1],
            b_dim_z = &block_sizes[2],
        );
        unwrap!(res);
    }
}

impl InstPrinter for CudaPrinter {
    fn print_label(&mut self, label: llir::Label<'_>) {
        unwrap!(writeln!(self.buffer, "{}:", label.name()));
    }

    fn print_inst(&mut self, inst: llir::PredicatedInstruction<'_>) {
        writeln!(self.buffer, "{};", inst.ptx()).unwrap();
    }
}

impl PTXDisplay for llir::UnOp {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::UnOp;

        match self {
            UnOp::Move { t } => write!(fmt, "mov.{}", t.ptx()),
            UnOp::Cast { src_t, dst_t } => {
                // Integer rounding is required for float-to-integer conversions, and for
                // same-size float-to-float conversions where the value is rounded to an
                // integer.  Integer rounding is illegal in all other instances. [1]
                //
                // When integer rounding is required, we choose to round to the nearest integer,
                // choosing even if source is equidistant between integers.
                //
                // Floating-point rounding is required for float-to-float conversions that result
                // in loss of precision, and for integer-to-float conversions.  Floating-point
                // rounding is illegal in all other instances.
                //
                // When floating-point rounding is required, we choose to round the mantissa LSB
                // towards even.
                //
                // [1]:
                // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt
                let rnd = match (src_t, dst_t) {
                    (ir::Type::F(_), ir::Type::I(_)) => ".rni",
                    (ir::Type::I(_), ir::Type::F(_)) => ".rn",
                    (ir::Type::F(src_bits), ir::Type::F(dst_bits))
                        if dst_bits < src_bits =>
                    {
                        ".rn"
                    }
                    _ => "",
                };
                write!(fmt, "cvt{}.{}.{}", rnd, dst_t.ptx(), src_t.ptx())
            }
            UnOp::Exp { .. } => panic!("{}: non-atomic PTX instruction", self),
        }
    }
}

impl PTXDisplay for llir::BinOp {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::BinOp::*;

        match self {
            // Integer Arithmetic Instructions
            IAdd { arg_t } => write!(fmt, "add.{}", arg_t.ptx()),
            ISub { arg_t } => write!(fmt, "sub.{}", arg_t.ptx()),
            IDiv { arg_t } => write!(fmt, "div.{}", arg_t.ptx()),
            IMul { arg_t, spec } => write!(fmt, "mul.{}.{}", spec.ptx(), arg_t.ptx()),
            // Floating-Point Instructions
            FAdd { t, rounding } => write!(fmt, "add.{}.{}", rounding.ptx(), t.ptx()),
            FSub { t, rounding } => write!(fmt, "sub.{}.{}", rounding.ptx(), t.ptx()),
            FMul { t, rounding } => write!(fmt, "mul.{}.{}", rounding.ptx(), t.ptx()),
            FDiv { t, rounding } => write!(fmt, "div.{}.{}", rounding.ptx(), t.ptx()),
            FMax { t } => write!(fmt, "max.{}", t.ptx()),
            FMin { t } => write!(fmt, "min.{}", t.ptx()),
            // Comparison and Selection Instructions
            Set { op, arg_t } => write!(fmt, "setp.{}.{}", op.ptx(), arg_t.ptx()),
            // Logic and Shift Instructions
            And { t } => write!(fmt, "and.{}", t.ptx()),
            Or { t } => write!(fmt, "or.{}", t.ptx()),
            Xor { t } => write!(fmt, "xor.{}", t.ptx()),
        }
    }
}

impl PTXDisplay for llir::TernOp {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::TernOp::*;

        match self {
            IMad { arg_t, spec } => write!(fmt, "mad.{}.{}", spec.ptx(), arg_t.ptx()),
            FFma { t, rounding } => write!(fmt, "fma.{}.{}", rounding.ptx(), t.ptx()),
        }
    }
}

impl PTXDisplay for llir::CmpOp {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(match self {
            llir::CmpOp::Eq => "eq",
            llir::CmpOp::Ne => "ne",
            llir::CmpOp::Lt => "lt",
            llir::CmpOp::Le => "le",
            llir::CmpOp::Gt => "gt",
            llir::CmpOp::Ge => "ge",
        })
    }
}

impl PTXDisplay for llir::FpRounding {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(match self {
            llir::FpRounding::NearestEven => "rn",
            llir::FpRounding::Zero => "rz",
            llir::FpRounding::NegativeInfinite => "rm",
            llir::FpRounding::PositiveInfinite => "rp",
        })
    }
}

impl PTXDisplay for llir::MulSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(match self {
            llir::MulSpec::Low => "lo",
            llir::MulSpec::High => "hi",
            llir::MulSpec::Wide => "wide",
        })
    }
}

impl PTXDisplay for llir::Address<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::Address;

        match *self {
            Address::Register(reg, offset) if offset == 0 => {
                write!(fmt, "[{}]", reg.ptx())
            }
            Address::Register(reg, offset) => {
                write!(fmt, "[{}{:+#x}]", reg.ptx(), offset)
            }
        }
    }
}

impl PTXDisplay for llir::Label<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(self.name())
    }
}

impl PTXDisplay for llir::StateSpace {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::StateSpace;

        fmt.write_str(match self {
            StateSpace::Global => "global",
            StateSpace::Shared => "shared",
        })
    }
}

impl PTXDisplay for llir::LoadCacheOperator {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::LoadCacheOperator::*;

        fmt.write_str(match self {
            CacheAll => "ca",
            CacheGlobal => "cg",
            CacheStreaming => "cs",
            CacheAllAndTexture => "nc",
        })
    }
}

impl PTXDisplay for llir::StoreCacheOperator {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::StoreCacheOperator::*;

        fmt.write_str(match self {
            WriteBack => "wb",
            CacheGlobal => "cg",
            CacheStreaming => "cs",
        })
    }
}

impl PTXDisplay for llir::LoadSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, ".{}.{}", self.state_space(), self.cache_operator())?;
        if self.vector_factor().get() > 1 {
            write!(fmt, ".v{}", self.vector_factor())?;
        }
        write!(fmt, ".{}", self.t())
    }
}

impl PTXDisplay for llir::StoreSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, ".{}.{}", self.state_space(), self.cache_operator())?;
        if self.vector_factor().get() > 1 {
            write!(fmt, ".v{}", self.vector_factor())?;
        }
        write!(fmt, ".{}", self.t())
    }
}

impl PTXDisplay for llir::PredicatedInstruction<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            fmt,
            "{}{}",
            self.predicate
                .into_iter()
                .format_with("", |predicate, f| f(&format_args!(
                    "@{} ",
                    predicate.ptx()
                ))),
            self.instruction.ptx()
        )
    }
}

impl PTXDisplay for llir::Instruction<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use llir::Instruction::*;

        match self {
            // The expression exp(x) needs to be decomposed into
            // 2^(log2(e)*x).
            //
            // Only implemented for f32, since other types require
            // significant software emulation
            Unary(llir::UnOp::Exp { t }, d, [a]) if t == &ir::Type::F(32) => {
                writeln!(
                    fmt,
                    "mul.{t} {d}, 0f3fb8aa3b, {a}; // 0f3fb8aa3b = log2(e)",
                    t = t.ptx(),
                    d = d.ptx(),
                    a = a.ptx()
                )?;
                write!(fmt, "ex2.approx.{t} {d}, {d}", t = t.ptx(), d = d.ptx())
            }
            Unary(op, d, [a]) => write!(fmt, "{} {}, {}", op.ptx(), d.ptx(), a.ptx()),
            Binary(op, d, [a, b]) => {
                write!(fmt, "{} {}, {}, {}", op.ptx(), d.ptx(), a.ptx(), b.ptx())
            }
            Ternary(op, d, [a, b, c]) => write!(
                fmt,
                "{} {}, {}, {}, {}",
                op.ptx(),
                d.ptx(),
                a.ptx(),
                b.ptx(),
                c.ptx()
            ),
            Load(spec, d, a) => write!(fmt, "ld{} {}, {}", spec.ptx(), d.ptx(), a.ptx()),
            Store(spec, a, [b]) => {
                write!(fmt, "st{} {}, {}", spec.ptx(), a.ptx(), b.ptx())
            }
            Jump(label) => write!(fmt, "bra.uni {}", label.ptx()),
            Sync => write!(fmt, "bar.sync 0"),
        }
    }
}

impl PTXDisplay for ir::Type {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::I(1) => write!(fmt, "pred"),
            Type::I(size) => write!(fmt, "s{size}", size = size),
            Type::F(size) => write!(fmt, "f{size}", size = size),
            _ => panic!("unexpected PTX type: {}", self),
        }
    }
}
