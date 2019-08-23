use crate::NameGenerator;
use itertools::Itertools;
use std::fmt::{self, Write as WriteFmt};
use telamon::codegen::llir::IntoVector;
use telamon::codegen::*;
use telamon::ir::{self, op, Type};
use telamon::search_space::{DimKind, Domain, InstFlag, MemSpace};
use utils::unwrap;
// TODO(cc_perf): avoid concatenating strings.

#[derive(Default)]
pub(crate) struct X86printer {
    buffer: String,
}

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

impl<T: C99Display> C99Display for llir::Vector<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            llir::Vector::Scalar(scalar) => C99Display::fmt(scalar, fmt),
            llir::Vector::Vector(_) => panic!("x86 backend does not support vectors."),
        }
    }
}

impl X86printer {
    /// Declares all parameters of the function with the appropriate type
    fn param_decl(&mut self, param: &ParamVal, namegen: &NameMap<'_>) -> String {
        let name = namegen.name_param(param.key());
        match param {
            ParamVal::External(_, par_type) => {
                format!("{} {}", Self::get_type(*par_type), name)
            }
            ParamVal::Size(_) => format!("uint32_t {}", name),
            ParamVal::GlobalMem(_, _, par_type) => {
                format!("{} {}", Self::get_type(*par_type), name)
            }
        }
    }

    /// Declared all variables that have been required from the namegen
    fn var_decls(&mut self, namegen: &NameGenerator) -> String {
        let print_decl = |(&t, &n)| {
            if let Type::PtrTo(..) = t {
                unreachable!("Type PtrTo are never inserted in this map");
            }
            let prefix = NameGenerator::gen_prefix(t);
            let mut s = format!("{} ", Self::get_type(t));
            s.push_str(
                &(0..n)
                    .map(|i| format!("{}{}", prefix, i))
                    .collect_vec()
                    .join(", "),
            );
            s.push_str(";\n  ");
            s
        };
        let other_var_decl = namegen
            .num_var
            .iter()
            .map(print_decl)
            .collect_vec()
            .join("\n  ");
        if namegen.num_glob_ptr == 0 {
            other_var_decl
        } else {
            format!(
                "intptr_t {};\n{}",
                &(0..namegen.num_glob_ptr)
                    .map(|i| format!("ptr{}", i))
                    .collect_vec()
                    .join(", "),
                other_var_decl,
            )
        }
    }

    /// Declares block and thread indexes.
    fn decl_par_indexes(
        &mut self,
        function: &Function,
        namegen: &mut NameMap<'_>,
    ) -> String {
        assert!(function.block_dims().is_empty());
        let mut decls = vec![];
        // Compute thread indexes.
        for (ind, dim) in function.thread_dims().iter().enumerate() {
            decls.push(format!(
                "{} = tid.t{};\n",
                namegen.name_index(dim.id()).name(),
                ind
            ));
        }
        decls.join("\n  ")
    }

    /// Prints a `Function`.
    pub fn function(&mut self, function: &Function) -> String {
        let mut namegen = NameGenerator::default();
        let interner = Interner::default();
        let name_map = &mut NameMap::new(&interner, function, &mut namegen);
        let param_decls = function
            .device_code_args()
            .map(|v| self.param_decl(v, name_map))
            .collect_vec()
            .join(",\n  ");
        // SIGNATURE AND OPEN BRACKET
        let mut return_string = if function.device_code_args().count() == 0 {
            format!(
                include_str!("template/signature_no_arg.c.template"),
                name = function.name(),
            )
        } else {
            format!(
                include_str!("template/signature.c.template"),
                name = function.name(),
                params = param_decls
            )
        };
        // INDEX LOADS
        let idx_loads = self.decl_par_indexes(function, name_map);
        unwrap!(writeln!(self.buffer, "{}", idx_loads));
        // LOAD PARAM
        for val in function.device_code_args() {
            unwrap!(writeln!(
                self.buffer,
                "{var_name} = {name};// LD_PARAM",
                var_name = name_map.name_param_val(val.key()).c99(),
                name = name_map.name_param(val.key())
            ));
        }
        // MEM DECL
        for block in function.mem_blocks() {
            match block.alloc_scheme() {
                AllocationScheme::Shared => panic!("No shared mem in cpu!!"),
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
        // INIT
        let ind_levels = function.init_induction_levels().iter().chain(
            function
                .block_dims()
                .iter()
                .flat_map(|d| d.induction_levels()),
        );
        for level in ind_levels {
            Printer::new(self, name_map).parallel_induction_level(level);
        }
        // BODY
        Printer::new(self, name_map).cfg(function, function.cfg());
        let var_decls = self.var_decls(&namegen);
        return_string.push_str(&var_decls);
        return_string.push_str(&self.buffer);
        // Close function bracket
        return_string.push('}');
        return_string
    }

    /// Function takes parameters as an array of void* pointers
    /// This function converts back these pointers into their original types
    fn fun_params_cast(&mut self, function: &Function) -> String {
        function
            .device_code_args()
            .enumerate()
            .map(|(i, v)| match v {
                ParamVal::External(..) if v.is_pointer() => {
                    format!("intptr_t p{i} = (intptr_t)*(args + {i})", i = i)
                }
                ParamVal::External(_, par_type) => format!(
                    "{t} p{i} = *({t}*)*(args + {i})",
                    t = Self::get_type(*par_type),
                    i = i
                ),
                ParamVal::Size(_) => {
                    format!("uint32_t p{i} = *(uint32_t*)*(args + {i})", i = i)
                }
                // Are we sure we know the size at compile time ? I think we do
                ParamVal::GlobalMem(_, _, par_type) => format!(
                    "{t} p{i} = ({t})*(args + {i})",
                    t = Self::get_type(*par_type),
                    i = i
                ),
            })
            .collect_vec()
            .join(";\n  ")
    }

    /// Declares the variables that will be used in C function call
    fn params_call(&mut self, function: &Function) -> String {
        function
            .device_code_args()
            .enumerate()
            .map(|x| x.0)
            .map(|i| format!("p{}", i))
            .collect_vec()
            .join(", ")
    }

    /// Build the right call for a nested loop on dimensions with linearized accesses
    /// that is, for a 3 dimensions arrays a[2][5][3] returns d0 + d1 * 3 + d2 * 5
    fn build_index_call(&mut self, func: &Function) -> String {
        let mut vec_ret = vec![];
        let dims = func.thread_dims();
        let n = dims.len();
        for i in 0..n {
            let start = format!("d{}", i);
            let mut vec_str = vec![start];
            for dim in dims.iter().take(i) {
                vec_str.push(format!("{}", unwrap!(dim.size().as_int())));
            }
            vec_ret.push(vec_str.join(" * "));
        }
        vec_ret.join(" + ")
    }

    /// Helper for building a structure containing as many thread id (one id per dim)
    /// as required.
    fn build_thread_id_struct(&mut self, func: &Function) -> String {
        let mut ret = String::new();
        if func.num_threads() == 1 {
            return String::from("int t0;\n");
        }
        for (ind, _dim) in func.thread_dims().iter().enumerate() {
            ret.push_str(&format!("int t{};\n", ind));
        }
        ret
    }

    /// Prints code that generates the required number of threads, stores the handles in an array
    fn thread_gen(&mut self, func: &Function) -> String {
        if func.num_threads() == 1 {
            return include_str!("template/monothread_init.c.template").to_string();
        }
        let mut loop_decl = String::new();
        let mut ind_vec = Vec::new();
        let mut loop_nesting = 0;

        for (ind, dim) in func.thread_dims().iter().enumerate() {
            ind_vec.push(format!("d{}", ind));
            unwrap!(writeln!(
                loop_decl,
                "for(d{ind} = 0; d{ind} < {size}; d{ind}++) {{",
                ind = ind,
                size = unwrap!(dim.size().as_int()),
            ));
            loop_nesting += 1;
        }
        let ind_dec_inter = ind_vec.join(", ");
        let ind_var_decl = format!("int {};", ind_dec_inter);
        let loop_ends = "}\n".repeat(loop_nesting);
        let mut tid_struct = String::new();
        for (ind, _) in func.thread_dims().iter().enumerate() {
            tid_struct.push_str(&format!(
                "thread_args[{index}].tid.t{dim_id} = d{dim_id};\n",
                index = self.build_index_call(func),
                dim_id = ind
            ));
        }
        format!(
            include_str!("template/multithread_init.c.template"),
            num_threads = func.num_threads(),
            ind = self.build_index_call(func),
            ind_var_decl = ind_var_decl,
            loop_init = loop_decl,
            tid_struct = tid_struct,
            loop_jump = loop_ends
        )
    }

    /// Prints code that joins all previously generated threads
    fn thread_join(&mut self, func: &Function) -> String {
        if func.num_threads() == 1 {
            return String::new();
        }
        let mut loop_decl = String::new();
        let mut loop_nesting = 0;

        for (ind, dim) in func.thread_dims().iter().enumerate() {
            unwrap!(writeln!(
                loop_decl,
                "for(d{ind} = 0; d{ind} < {size}; d{ind}++) {{",
                ind = ind,
                size = unwrap!(dim.size().as_int()),
            ));
            loop_nesting += 1;
        }

        let loop_ends = "}\n".repeat(loop_nesting);

        format!(
            include_str!("template/join_thread.c.template"),
            ind = self.build_index_call(func),
            loop_init = loop_decl,
            loop_jump = loop_ends
        )
    }

    /// wrap the kernel call into a function with a fixed interface
    pub fn wrapper_function(&mut self, func: &Function) -> String {
        let fun_str = self.function(func);
        if func.device_code_args().count() == 0 {
            format!(
                include_str!("template/host_no_arg.c.template"),
                fun_name = func.name(),
                fun_str = fun_str,
                gen_threads = self.thread_gen(func),
                dim_decl = self.build_thread_id_struct(func),
                thread_join = self.thread_join(func),
            )
        } else {
            let fun_params = self.params_call(func);
            format!(
                include_str!("template/host.c.template"),
                fun_name = func.name(),
                fun_str = fun_str,
                fun_params_cast = self.fun_params_cast(func),
                fun_params = fun_params,
                gen_threads = self.thread_gen(func),
                dim_decl = self.build_thread_id_struct(func),
                thread_join = self.thread_join(func),
            )
        }
    }

    fn get_type(t: Type) -> String {
        match t {
            Type::PtrTo(..) => String::from("intptr_t"),
            Type::F(32) => String::from("float"),
            Type::F(64) => String::from("double"),
            Type::I(1) => String::from("int8_t"),
            Type::I(8) => String::from("int8_t"),
            Type::I(16) => String::from("int16_t"),
            Type::I(32) => String::from("int32_t"),
            Type::I(64) => String::from("int64_t"),
            ref t => panic!("invalid type for the host: {}", t),
        }
    }
}

impl InstPrinter for X86printer {
    fn print_binop(
        &mut self,
        vector_factors: [u32; 2],
        op: ir::BinOp,
        _: Type,
        _: op::Rounding,
        result: llir::VRegister<'_>,
        lhs: llir::VOperand<'_>,
        rhs: llir::VOperand<'_>,
    ) {
        assert_eq!(vector_factors, [1, 1]);

        let (op, is_infix_op) = match op {
            ir::BinOp::Add => ("+", true),
            ir::BinOp::Sub => ("-", true),
            ir::BinOp::Div => ("/", true),
            ir::BinOp::And => ("&", true),
            ir::BinOp::Or => ("|", true),
            ir::BinOp::Lt => ("<", true),
            ir::BinOp::Leq => ("<=", true),
            ir::BinOp::Equals => ("==", true),
            ir::BinOp::Max => ("telamon_op_max", false),
        };

        if is_infix_op {
            unwrap!(writeln!(
                self.buffer,
                "{} = {} {} {};",
                result.c99(),
                lhs.c99(),
                op,
                rhs.c99()
            ));
        } else {
            unwrap!(writeln!(
                self.buffer,
                "{} = {}({}, {});",
                result.c99(),
                op,
                lhs.c99(),
                rhs.c99()
            ));
        }
    }

    fn print_unary_op(
        &mut self,
        vector_factors: [u32; 2],
        operator: ir::UnaryOp,
        _: Type,
        result: llir::VRegister<'_>,
        operand: llir::VOperand<'_>,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        unwrap!(write!(self.buffer, "{} = ", result.c99()));
        match operator {
            ir::UnaryOp::Mov => {
                unwrap!(writeln!(self.buffer, "{};", operand.c99()));
            }

            ir::UnaryOp::Cast(t) => {
                unwrap!(write!(
                    self.buffer,
                    "({}){};",
                    Self::get_type(t),
                    operand.c99()
                ));
            }

            ir::UnaryOp::Exp(t) => match t {
                ir::Type::F(32) => {
                    unwrap!(write!(self.buffer, "expf({});", operand.c99()))
                }
                _ => panic!("Exp not implemented for type {}", t),
            },
        };
    }

    fn print_mul(
        &mut self,
        vector_factors: [u32; 2],
        _: Type,
        _: op::Rounding,
        mode: MulMode,
        result: llir::VRegister<'_>,
        op1: llir::VOperand<'_>,
        op2: llir::VOperand<'_>,
    ) {
        assert_ne!(mode, MulMode::High);
        assert_eq!(vector_factors, [1, 1]);
        unwrap!(writeln!(
            self.buffer,
            "{} = {} * {};",
            result.c99(),
            op1.c99(),
            op2.c99()
        ));
    }

    fn print_mad(
        &mut self,
        vector_factors: [u32; 2],
        _: Type,
        _: op::Rounding,
        mode: MulMode,
        result: llir::VRegister<'_>,
        mlhs: llir::VOperand<'_>,
        mrhs: llir::VOperand<'_>,
        arhs: llir::VOperand<'_>,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        assert_ne!(mode, MulMode::High);
        unwrap!(writeln!(
            self.buffer,
            "{} = {} * {} + {};",
            result.c99(),
            mlhs.c99(),
            mrhs.c99(),
            arhs.c99(),
        ));
    }

    fn print_ld(
        &mut self,
        vector_factors: [u32; 2],
        return_type: Type,
        _: MemSpace,
        _: InstFlag,
        result: llir::VRegister<'_>,
        addr: llir::Operand<'_>,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        unwrap!(writeln!(
            self.buffer,
            "{} = *({}*){} ;",
            result.c99(),
            Self::get_type(return_type),
            addr.c99()
        ));
    }

    fn print_st(
        &mut self,
        vector_factors: [u32; 2],
        val_type: Type,
        _: MemSpace,
        _: InstFlag,
        predicate: Option<llir::Register<'_>>,
        addr: llir::Operand<'_>,
        val: llir::VOperand<'_>,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        if let Some(predicate) = predicate {
            unwrap!(write!(self.buffer, "if ({})", predicate.c99()));
        }
        unwrap!(writeln!(
            self.buffer,
            "*({}*){} = {} ;",
            Self::get_type(val_type),
            addr.c99(),
            val.c99(),
        ));
    }

    fn print_label(&mut self, label_id: &str) {
        unwrap!(writeln!(self.buffer, "LABEL_{}:", label_id));
    }

    fn print_cond_jump(&mut self, label_id: &str, cond: &str) {
        unwrap!(writeln!(
            self.buffer,
            "if({}) goto LABEL_{};",
            cond, label_id
        ));
    }

    fn print_sync(&mut self) {
        unwrap!(writeln!(self.buffer, "pthread_barrier_wait(tid.barrier);"));
    }
}
