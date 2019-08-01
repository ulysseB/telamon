use crate::ValuePrinter;
use itertools::Itertools;
use std::borrow::Cow;
use std::fmt::{self, Write as WriteFmt};
use telamon::codegen::*;
use telamon::ir::{self, op, Type};
use telamon::search_space::{DimKind, Domain, InstFlag, MemSpace};
use utils::unwrap;
// TODO(cc_perf): avoid concatenating strings.

#[derive(Default)]
pub(crate) struct X86printer {
    buffer: String,
}

impl X86printer {
    /// Declares all parameters of the function with the appropriate type
    fn param_decl(
        &mut self,
        param: &ParamVal,
        value_printer: &NameMap<ValuePrinter>,
    ) -> String {
        let name = value_printer.name_param(param.key());
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

    /// Declared all variables that have been required from the value_printer
    fn var_decls(&mut self, name_map: &NameMap<ValuePrinter>) -> String {
        let value_printer = name_map.value_printer();
        let print_decl = |(&t, &n)| {
            if let Type::PtrTo(..) = t {
                unreachable!("Type PtrTo are never inserted in this map");
            }
            let prefix = ValuePrinter::gen_prefix(t);
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
        let other_var_decl = value_printer
            .num_var
            .iter()
            .map(print_decl)
            .collect_vec()
            .join("\n  ");
        if value_printer.num_glob_ptr == 0 {
            other_var_decl
        } else {
            format!(
                "intptr_t {};\n{}",
                &(0..value_printer.num_glob_ptr)
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
        value_printer: &mut NameMap<ValuePrinter>,
    ) -> String {
        assert!(function.block_dims().is_empty());
        let mut decls = vec![];
        // Compute thread indexes.
        for (ind, dim) in function.thread_dims().iter().enumerate() {
            decls.push(format!(
                "{} = tid.t{};\n",
                value_printer.name_index(dim.id()),
                ind
            ));
        }
        decls.join("\n  ")
    }

    /// Prints a `Function`.
    pub fn function(&mut self, function: &Function) -> String {
        let mut value_printer = ValuePrinter::default();
        let name_map = &mut NameMap::new(function, &mut value_printer);
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
                var_name = name_map.name_param_val(val.key()),
                name = name_map.name_param(val.key())
            ));
        }
        // MEM DECL
        for block in function.mem_blocks() {
            match block.alloc_scheme() {
                AllocationScheme::Shared => panic!("No shared mem in cpu!!"),
                AllocationScheme::PrivatisedGlobal => {
                    self.privatise_global_block(block, name_map, function)
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
                    let name = name_map.declare_size_cast(incr, level.t());
                    if let Some(name) = name {
                        let old_name = name_map.name_size(incr, Type::I(32));
                        self.print_unary_op(
                            [1, 1],
                            ir::UnaryOp::Cast(level.t()),
                            Type::I(32),
                            &name,
                            &old_name,
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
            self.parallel_induction_level(level, name_map);
        }
        // BODY
        self.cfg(function, function.cfg(), name_map);
        let var_decls = self.var_decls(&name_map);
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
    type ValuePrinter = ValuePrinter;

    fn print_binop(
        &mut self,
        vector_factors: [u32; 2],
        op: ir::BinOp,
        _: Type,
        _: op::Rounding,
        result: &str,
        lhs: &str,
        rhs: &str,
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
                result, lhs, op, rhs
            ));
        } else {
            unwrap!(writeln!(
                self.buffer,
                "{} = {}({}, {});",
                result, op, lhs, rhs
            ));
        }
    }

    fn print_unary_op(
        &mut self,
        vector_factors: [u32; 2],
        operator: ir::UnaryOp,
        _: Type,
        result: &str,
        operand: &str,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        unwrap!(write!(self.buffer, "{} = ", result));
        match operator {
            ir::UnaryOp::Mov => {
                unwrap!(writeln!(self.buffer, "{};", operand));
            }

            ir::UnaryOp::Cast(t) => {
                unwrap!(write!(self.buffer, "({}){};", Self::get_type(t), operand));
            }

            ir::UnaryOp::Exp(t) => match t {
                ir::Type::F(32) => unwrap!(write!(self.buffer, "expf({});", operand)),
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
        result: &str,
        op1: &str,
        op2: &str,
    ) {
        assert_ne!(mode, MulMode::High);
        assert_eq!(vector_factors, [1, 1]);
        unwrap!(writeln!(self.buffer, "{} = {} * {};", result, op1, op2));
    }

    fn print_mad(
        &mut self,
        vector_factors: [u32; 2],
        _: Type,
        _: op::Rounding,
        mode: MulMode,
        result: &str,
        mlhs: &str,
        mrhs: &str,
        arhs: &str,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        assert_ne!(mode, MulMode::High);
        unwrap!(writeln!(
            self.buffer,
            "{} = {} * {} + {};",
            result, mlhs, mrhs, arhs
        ));
    }

    fn print_ld(
        &mut self,
        vector_factors: [u32; 2],
        return_type: Type,
        _: MemSpace,
        _: InstFlag,
        result: &str,
        addr: &str,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        unwrap!(writeln!(
            self.buffer,
            "{} = *({}*){} ;",
            result,
            Self::get_type(return_type),
            addr
        ));
    }

    fn print_st(
        &mut self,
        vector_factors: [u32; 2],
        val_type: Type,
        _: MemSpace,
        _: InstFlag,
        predicate: Option<&str>,
        addr: &str,
        val: &str,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        if let Some(predicate) = predicate {
            unwrap!(write!(self.buffer, "if ({})", predicate));
        }
        unwrap!(writeln!(
            self.buffer,
            "*({}*){} = {} ;",
            Self::get_type(val_type),
            addr,
            val
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

    fn name_operand<'a>(
        vector_dims: &[Vec<Dimension>; 2],
        op: &ir::Operand,
        value_printer: &'a NameMap<ValuePrinter>,
    ) -> Cow<'a, str> {
        assert!(vector_dims[0].is_empty());
        assert!(vector_dims[1].is_empty());
        value_printer.name_op(op)
    }

    fn name_inst<'a>(
        vector_dims: &[Vec<Dimension>; 2],
        inst: ir::InstId,
        value_printer: &'a NameMap<ValuePrinter>,
    ) -> Cow<'a, str> {
        assert!(vector_dims[0].is_empty());
        assert!(vector_dims[1].is_empty());
        value_printer.name_inst(inst).into()
    }

    fn print_loop(
        &mut self,
        fun: &Function<'_>,
        loop_: &Loop<'_, '_>,
        body: &[Cfg<'_>],
        namer: &mut NameMap<'_, '_, Self::ValuePrinter>,
    ) {
        struct LoopHeader<'a, 'b, 'c, 'd> {
            loop_: &'a Loop<'a, 'b>,
            namer: &'a mut NameMap<'c, 'd, ValuePrinter>,
        }

        impl<'a, 'b, 'c, 'd> fmt::Display for LoopHeader<'a, 'b, 'c, 'd> {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    fmt,
                    "for ({}; {} < {}; {})",
                    self.loop_
                        .inits
                        .iter()
                        .format_with(", ", |(target, inst), f| {
                            f(&target.name(self.namer))?;

                            match inst {
                                Inst::Move(_, op) => {
                                    f(&format_args!(" = {}", op.name(self.namer)))?
                                }
                                Inst::Add(_, lhs, rhs) => f(&format_args!(
                                    " = {} + {}",
                                    lhs.name(self.namer),
                                    rhs.name(self.namer)
                                ))?,
                                Inst::AddAssign(_, op) => {
                                    f(&format_args!(" += {}", op.name(self.namer)))?
                                }
                            }

                            Ok(())
                        }),
                    self.loop_.index.name(self.namer),
                    self.loop_.bound.name(self.namer),
                    self.loop_.increments.iter().format_with(
                        ", ",
                        |(target, inst), f| {
                            f(&target.name(self.namer))?;

                            match inst {
                                Inst::Move(_, op) => {
                                    f(&format_args!(" = {}", op.name(self.namer)))?
                                }
                                Inst::Add(_, lhs, rhs) => f(&format_args!(
                                    " = {} + {}",
                                    lhs.name(self.namer),
                                    rhs.name(self.namer)
                                ))?,
                                Inst::AddAssign(_, op) => {
                                    f(&format_args!(" += {}", op.name(self.namer)))?
                                }
                            }

                            Ok(())
                        }
                    ),
                )
            }
        }

        unwrap!(writeln!(self.buffer, "{} {{", LoopHeader { loop_, namer }));
        self.cfg_vec(fun, body, namer);
        unwrap!(writeln!(self.buffer, "}}"));
    }
}
