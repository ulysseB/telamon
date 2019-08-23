//! Provides functions to print PTX code.
use std::fmt::{self, Write as WriteFmt};
use std::io::Write;

use utils::*;

use itertools::Itertools;
use telamon::codegen::llir::IntoVector;
use telamon::codegen::*;
use telamon::ir::{self, op, Type};
use telamon::search_space::{DimKind, Domain, InstFlag, MemSpace};

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
        fmt.write_str(self.name())
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

impl<T: PTXDisplay> PTXDisplay for llir::Vector<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            llir::Vector::Scalar(scalar) => PTXDisplay::fmt(scalar, fmt),
            llir::Vector::Vector(ts) => {
                write!(fmt, "{{{}}}", ts.iter().map(PTXDisplay::ptx).format(", "))
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct CudaPrinter {
    buffer: String,
}

impl CudaPrinter {
    fn mul_mode(mode: MulMode) -> &'static str {
        match mode {
            MulMode::Wide => ".wide",
            MulMode::Low => ".lo",
            MulMode::High => ".hi",
            MulMode::Empty => "",
        }
    }

    fn get_inst_type(mode: MulMode, ret_type: Type) -> Type {
        match mode {
            MulMode::Wide => {
                if let Type::I(n) = ret_type {
                    Type::I(n / 2)
                } else {
                    panic!("get_inst_type should only be called with integer types")
                }
            }
            _ => ret_type,
        }
    }

    /// Prints a load operator.
    fn ld_operator(space: MemSpace, flag: InstFlag) -> &'static str {
        if space == MemSpace::SHARED {
            "ld.shared"
        } else {
            match flag {
                InstFlag::CACHE_SHARED => "ld.global.ca",
                InstFlag::CACHE_GLOBAL => "ld.global.cg",
                InstFlag::CACHE_READ_ONLY => "ld.global.nc",
                InstFlag::NO_CACHE => "ld.global.cs",
                _ => panic!("invalid load flag {:?}", flag),
            }
        }
    }

    /// Prints a store operator.
    fn st_operator(space: MemSpace, flag: InstFlag) -> &'static str {
        if space == MemSpace::SHARED {
            "st.shared"
        } else {
            match flag {
                InstFlag::CACHE_SHARED => "st.global.wb",
                InstFlag::CACHE_GLOBAL => "st.global.cg",
                InstFlag::NO_CACHE => "st.global.cs",
                _ => panic!("invalid store flag {:?}", flag),
            }
        }
    }

    /// Prints the variables declared by the `NameGenerator`.
    fn var_decls(&mut self, namegen: &NameGenerator) -> String {
        let print_decl = |(&t, n)| {
            let prefix = NameGenerator::gen_prefix(t);
            format!(".reg.{} %{}<{}>;", Self::get_type(t), prefix, n)
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
        let ptr_type_name = Self::get_type(Type::I(32));
        let name = name_map.name_addr(block.id()).name();
        unwrap!(writeln!(
            self.buffer,
            "  .shared.align 16 .u8 {vec_name}[{size}];\
             \n  mov.{t} {name}, {vec_name};",
            vec_name = &name[1..],
            name = name,
            t = ptr_type_name,
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

    fn binary_op(op: ir::BinOp) -> &'static str {
        match op {
            ir::BinOp::Add => "add",
            ir::BinOp::Sub => "sub",
            ir::BinOp::Div => "div",
            ir::BinOp::And => "and",
            ir::BinOp::Or => "or",
            ir::BinOp::Lt => "setp.lt",
            ir::BinOp::Leq => "setp.le",
            ir::BinOp::Equals => "setp.eq",
            ir::BinOp::Max => "max",
        }
    }

    /// Prints a parameter decalartion.
    fn param_decl(&mut self, param: &ParamVal, name_map: &NameMap<'_>) -> String {
        format!(
            ".param .{t}{attr} {name}",
            t = Self::get_type(param.t()),
            attr = if param.is_pointer() {
                ".ptr.global.align 16"
            } else {
                ""
            },
            name = name_map.name_param(param.key()),
        )
    }
    /// Prints a rounding mode selector.
    fn rounding(rounding: op::Rounding) -> &'static str {
        match rounding {
            op::Rounding::Exact => "",
            op::Rounding::Nearest => ".rn",
            op::Rounding::Zero => ".rz",
            op::Rounding::Positive => ".rp",
            op::Rounding::Negative => ".rm",
        }
    }

    /// Prints a `Function`.
    pub fn function(&mut self, function: &Function, gpu: &Gpu) -> String {
        let mut namegen = NameGenerator::default();
        let interner = Interner::default();
        let name_map = &mut NameMap::new(&interner, function, &mut namegen);
        let param_decls = function
            .device_code_args()
            .map(|v| self.param_decl(v, name_map))
            .collect_vec()
            .join(",\n  ");
        // LOAD PARAMETERS
        for val in function.device_code_args() {
            unwrap!(writeln!(
                self.buffer,
                "  ld.param.{t} {var_name}, [{name}];",
                t = Self::get_type(val.t()),
                var_name = name_map.name_param_val(val.key()).ptx(),
                name = name_map.name_param(val.key())
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
                        self.print_unary_op(
                            [1, 1],
                            ir::UnaryOp::Cast(level.t()),
                            Type::I(32),
                            reg.into_vector(),
                            old_name.into_vector(),
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
        let mut next_extra_var_id = 0;
        let mut extra_def = vec![];
        let mut extra_cleanup = vec![];
        let params = fun
            .device_code_args()
            .map(|p| match *p {
                ParamVal::External(ref p, _) => format!("&{}", p.name),
                ParamVal::Size(ref size) => {
                    let extra_var = format!("_extra_{}", next_extra_var_id);
                    next_extra_var_id += 1;
                    extra_def.push(format!(
                        "int32_t {} = {};",
                        extra_var,
                        Self::host_size(size)
                    ));
                    format!("&{}", extra_var)
                }
                ParamVal::GlobalMem(_, ref size, _) => {
                    let extra_var = format!("_extra_{}", next_extra_var_id);
                    next_extra_var_id += 1;
                    let size = Self::host_size(size);
                    extra_def.push(format!("CUDeviceptr {};", extra_var));
                    extra_def.push(format!(
                        "CHECK_CUDA(cuMemAlloc(&{}, {}));",
                        extra_var, size
                    ));
                    extra_cleanup.push(format!("CHECK_CUDA(cuMemFree({}));", extra_var));
                    format!("&{}", extra_var)
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

    /// Print a type in the backend
    fn get_type(t: Type) -> String {
        match t {
            Type::I(1) => "pred".to_string(),
            Type::I(size) => format!("s{size}", size = size),
            Type::F(size) => format!("f{size}", size = size),
            _ => panic!(),
        }
    }
}

impl InstPrinter for CudaPrinter {
    /// Print result = op1 op op2
    fn print_binop(
        &mut self,
        vector_factors: [u32; 2],
        op: ir::BinOp,
        operands_type: Type,
        rounding: op::Rounding,
        result: llir::VRegister<'_>,
        lhs: llir::VOperand<'_>,
        rhs: llir::VOperand<'_>,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        let op = Self::binary_op(op);
        let rounding = Self::rounding(rounding);
        let operands_type = Self::get_type(operands_type);
        unwrap!(writeln!(
            self.buffer,
            "  {}{}.{} {}, {}, {};",
            op,
            rounding,
            operands_type,
            result.ptx(),
            lhs.ptx(),
            rhs.ptx()
        ));
    }

    /// Prints result = operator operand.
    fn print_unary_op(
        &mut self,
        vector_factors: [u32; 2],
        operator: ir::UnaryOp,
        operand_type: ir::Type,
        result: llir::VRegister<'_>,
        operand: llir::VOperand<'_>,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        let t_str = Self::get_type(operand_type);

        match operator {
            ir::UnaryOp::Mov => unwrap!(writeln!(
                self.buffer,
                "  mov.{} {}, {};",
                t_str,
                result.ptx(),
                operand.ptx()
            )),
            ir::UnaryOp::Exp(..) => {
                // The expression exp(x) needs to be decomposed into
                // 2^(log2(e)*x).
                //
                // Only implemented for f32, since other types require
                // significant software emulation
                match operand_type {
                    ir::Type::F(32) => {
                        unwrap!(writeln!(
                            self.buffer,
                            "mul.{t} {reg}, 0f3fb8aa3b, {operand}; // 0f3fb8aa3b = log2(e)\n
                             ex2.approx.{t} {reg}, {reg};",
                            t = t_str,
                            reg = result.ptx(),
                            operand = operand.ptx(),
                        ));
                    }
                    _ => panic!("No implementation of exp for type {}", operand_type),
                }
            }
            ir::UnaryOp::Cast(cast_type) => {
                let rounding = match (cast_type, operand_type) {
                    (ir::Type::F(_), ir::Type::I(_))
                    | (ir::Type::I(_), ir::Type::F(_)) => ir::op::Rounding::Nearest,
                    (ir::Type::F(x), ir::Type::F(y)) if x < y => {
                        ir::op::Rounding::Nearest
                    }
                    _ => ir::op::Rounding::Exact,
                };
                let rounding = Self::rounding(rounding);
                unwrap!(writeln!(
                    self.buffer,
                    "cvt{}.{}.{} {}, {};",
                    rounding,
                    Self::get_type(cast_type),
                    t_str,
                    result.ptx(),
                    operand.ptx()
                ));
            }
        };
    }

    /// Print result = op1 * op2
    fn print_mul(
        &mut self,
        vector_factors: [u32; 2],
        return_type: Type,
        round: op::Rounding,
        mul_mode: MulMode,
        result: llir::VRegister<'_>,
        lhs: llir::VOperand<'_>,
        rhs: llir::VOperand<'_>,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        let operator = if round == op::Rounding::Exact {
            format!("mul{}", Self::mul_mode(mul_mode))
        } else {
            format!("mul{}", Self::rounding(round))
        };
        let t = Self::get_type(Self::get_inst_type(mul_mode, return_type));
        unwrap!(writeln!(
            self.buffer,
            "  {}.{} {}, {}, {};",
            operator,
            t,
            result.ptx(),
            lhs.ptx(),
            rhs.ptx()
        ));
    }

    /// Print result = mlhs * mrhs + arhs
    fn print_mad(
        &mut self,
        vector_factors: [u32; 2],
        return_type: Type,
        round: op::Rounding,
        mul_mode: MulMode,
        result: llir::VRegister<'_>,
        mlhs: llir::VOperand<'_>,
        mrhs: llir::VOperand<'_>,
        arhs: llir::VOperand<'_>,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        let operator = if round == op::Rounding::Exact {
            format!("mad{}", Self::mul_mode(mul_mode))
        } else {
            format!("fma{}", Self::rounding(round))
        };
        let t = Self::get_type(Self::get_inst_type(mul_mode, return_type));
        unwrap!(writeln!(
            self.buffer,
            "  {}.{} {}, {}, {}, {};",
            operator,
            t,
            result.ptx(),
            mlhs.ptx(),
            mrhs.ptx(),
            arhs.ptx(),
        ));
    }

    /// Print result = load [addr]
    fn print_ld(
        &mut self,
        vector_factors: [u32; 2],
        return_type: Type,
        mem_space: MemSpace,
        flag: InstFlag,
        result: llir::VRegister<'_>,
        addr: llir::Operand<'_>,
    ) {
        let operator = Self::ld_operator(mem_space, flag);
        let vector = match vector_factors {
            [1, 1] => "",
            [1, 2] => ".v2",
            [1, 4] => ".v4",
            p => panic!("invalid vector pattern: {:?}", p),
        };
        unwrap!(writeln!(
            self.buffer,
            "  {}{}.{} {}, [{}];",
            operator,
            vector,
            Self::get_type(return_type),
            result.ptx(),
            addr.ptx(),
        ));
    }

    /// Print store val [addr]
    fn print_st(
        &mut self,
        vector_factors: [u32; 2],
        val_type: Type,
        mem_space: MemSpace,
        mem_flag: InstFlag,
        predicate: Option<llir::Register<'_>>,
        addr: llir::Operand<'_>,
        val: llir::VOperand<'_>,
    ) {
        let vector = match vector_factors {
            [1, 1] => "",
            [1, 2] => ".v2",
            [1, 4] => ".v4",
            p => panic!("invalid vector pattern: {:?}", p),
        };
        if let Some(predicate) = predicate {
            unwrap!(write!(self.buffer, "@{} ", predicate.ptx()));
        }
        let operator = Self::st_operator(mem_space, mem_flag);
        unwrap!(writeln!(
            self.buffer,
            "  {}{}.{} [{}], {};",
            operator,
            vector,
            Self::get_type(val_type),
            addr.ptx(),
            val.ptx(),
        ));
    }

    /// print a label where to jump
    fn print_label(&mut self, label_id: &str) {
        unwrap!(writeln!(self.buffer, "LOOP_{}:", label_id));
    }

    /// Print if (cond) jump label(label_id)
    fn print_cond_jump(&mut self, label_id: &str, cond: &str) {
        unwrap!(writeln!(
            self.buffer,
            "  @{} bra.uni LOOP_{};",
            cond, label_id
        ));
    }

    /// Print wait on all threads
    fn print_sync(&mut self) {
        unwrap!(writeln!(self.buffer, "  bar.sync 0;"));
    }
}
