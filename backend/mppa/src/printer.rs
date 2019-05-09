use crate::ValuePrinter;
use itertools::Itertools;
use std::borrow::Cow;
use std::fmt::Write as WriteFmt;
use telamon::codegen::*;
use telamon::ir::{self, op, Type};
use telamon::search_space::{InstFlag, MemSpace};
use telamon_c::printer::CPrinter;
use utils::unwrap;
// TODO(cc_perf): avoid concatenating strings.

#[derive(Default)]
pub struct MppaPrinter {
    c_printer: CPrinter,
}

impl MppaPrinter {
    /// Prints code that generates the required number of threads,
    /// stores the handles in an array
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
                "tids[{index}].t{dim_id} = d{dim_id};\n",
                index = self.c_printer.build_index_call(func),
                dim_id = ind
            ));
        }
        format!(
            include_str!("template/multithread_init.c.template"),
            num_threads = func.num_threads(),
            ind = self.c_printer.build_index_call(func),
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
            ind = self.c_printer.build_index_call(func),
            loop_init = loop_decl,
            loop_jump = loop_ends
        )
    }

    /// Turns the argument of wrapper into an array of void pointers
    /// Necessary to call pthread with it
    fn build_ptr_struct(
        &self,
        func: &Function,
        name_map: &NameMap<ValuePrinter>,
    ) -> String {
        func.device_code_args()
            .enumerate()
            .map(|(i, arg)| {
                let name = name_map.name_param(arg.key());
                if arg.is_pointer() {
                    format!("args[{}] = (void *){}", i, name)
                } else {
                    format!("args[{}] = (void *)&{}", i, name)
                }
            })
            .join(";\n")
    }

    /// wrap the kernel call into a function with a fixed interface
    pub fn wrapper_function<'a: 'b, 'b>(
        &mut self,
        func: &'b Function<'a>,
        name_map: &mut NameMap<'b, '_, ValuePrinter>,
        id: usize,
    ) -> String {
        let fun_str = self.c_printer.function(func, name_map);
        let fun_params = self.c_printer.params_call(func);
        let (lower_bound, upper_n_arg) = func.device_code_args().size_hint();
        let n_args = if let Some(upper_bound) = upper_n_arg {
            assert_eq!(upper_bound, lower_bound);
            upper_bound
        } else {
            20
        };
        let cl_arg_def = func
            .device_code_args()
            .map(|v| self.c_printer.param_decl(v, name_map))
            .collect_vec()
            .join(",  ");
        format!(
            include_str!("template/host.c.template"),
            id = id,
            cl_arg_def = cl_arg_def,
            n_arg = n_args,
            build_ptr_struct = self.build_ptr_struct(func, name_map),
            fun_name = func.name(),
            fun_str = fun_str,
            fun_params_cast = self.c_printer.fun_params_cast(func),
            fun_params = fun_params,
            gen_threads = self.thread_gen(func),
            dim_decl = self.c_printer.build_thread_id_struct(func),
            thread_join = self.thread_join(func),
        )
    }

    /// Returns the name of a type.
    fn cl_type_name(t: ir::Type) -> &'static str {
        match t {
            ir::Type::PtrTo(..) => "__global void*",
            ir::Type::I(8) => "char",
            ir::Type::I(16) => "short",
            ir::Type::I(32) => "int",
            ir::Type::I(64) => "long",
            _ => ValuePrinter::get_string(t),
        }
    }
    /// Prints the OpenCL wrapper for a candidate implementation.
    pub fn print_ocl_wrapper(
        &mut self,
        fun: &Function,
        name_map: &mut NameMap<ValuePrinter>,
        id: usize,
    ) -> String {
        let arg_names = fun
            .device_code_args()
            .format_with(", ", |p, f| {
                f(&format_args!("{}", name_map.name_param(p.key())))
            })
            .to_string();
        let cl_arg_defs = fun
            .device_code_args()
            .format_with(", ", |p, f| {
                f(&format_args!(
                    "{} {}",
                    Self::cl_type_name(p.t()),
                    name_map.name_param(p.key())
                ))
            })
            .to_string();
        format!(
            include_str!("template/ocl_wrap.c.template"),
            fun_id = id,
            arg_names = arg_names,
            cl_arg_defs = cl_arg_defs,
        )
    }
}

impl InstPrinter for MppaPrinter {
    type ValuePrinter = ValuePrinter;

    fn print_binop(
        &mut self,
        vector_factors: [u32; 2],
        op: ir::BinOp,
        t: Type,
        rounding: op::Rounding,
        result: &str,
        lhs: &str,
        rhs: &str,
    ) {
        self.c_printer
            .print_binop(vector_factors, op, t, rounding, result, lhs, rhs);
    }

    fn print_unary_op(
        &mut self,
        vector_factors: [u32; 2],
        operator: ir::UnaryOp,
        t: Type,
        result: &str,
        operand: &str,
    ) {
        self.c_printer
            .print_unary_op(vector_factors, operator, t, result, operand);
    }

    fn print_mul(
        &mut self,
        vector_factors: [u32; 2],
        t: Type,
        rounding: op::Rounding,
        mode: MulMode,
        result: &str,
        op1: &str,
        op2: &str,
    ) {
        self.c_printer
            .print_mul(vector_factors, t, rounding, mode, result, op1, op2);
    }

    fn print_mad(
        &mut self,
        vector_factors: [u32; 2],
        t: Type,
        rounding: op::Rounding,
        mode: MulMode,
        result: &str,
        mlhs: &str,
        mrhs: &str,
        arhs: &str,
    ) {
        self.c_printer.print_mad(
            vector_factors,
            t,
            rounding,
            mode,
            result,
            mlhs,
            mrhs,
            arhs,
        );
    }

    fn print_ld(
        &mut self,
        vector_factors: [u32; 2],
        return_type: Type,
        mem_space: MemSpace,
        inst_flag: InstFlag,
        result: &str,
        addr: &str,
    ) {
        self.c_printer.print_ld(
            vector_factors,
            return_type,
            mem_space,
            inst_flag,
            result,
            addr,
        );
    }

    fn print_st(
        &mut self,
        vector_factors: [u32; 2],
        val_type: Type,
        mem_space: MemSpace,
        inst_flag: InstFlag,
        predicate: Option<&str>,
        addr: &str,
        val: &str,
    ) {
        self.c_printer.print_st(
            vector_factors,
            val_type,
            mem_space,
            inst_flag,
            predicate,
            addr,
            val,
        );
    }

    fn print_label(&mut self, label_id: &str) {
        self.c_printer.print_label(label_id);
    }

    fn print_cond_jump(&mut self, label_id: &str, cond: &str) {
        self.c_printer.print_cond_jump(label_id, cond);
    }

    fn print_sync(&mut self) {
        self.c_printer.print_sync();
    }

    fn name_operand<'a>(
        vector_dims: &[Vec<Dimension>; 2],
        op: &ir::Operand,
        name_map: &'a NameMap<ValuePrinter>,
    ) -> Cow<'a, str> {
        CPrinter::name_operand(vector_dims, op, name_map)
    }

    fn name_inst<'a>(
        vector_dims: &[Vec<Dimension>; 2],
        inst: ir::InstId,
        name_map: &'a NameMap<ValuePrinter>,
    ) -> Cow<'a, str> {
        CPrinter::name_inst(vector_dims, inst, name_map)
    }

    /// Prints a standard loop as a C for loop
    fn standard_loop(
        &mut self,
        fun: &Function,
        dim: &Dimension,
        cfgs: &[Cfg],
        namer: &mut NameMap<Self::ValuePrinter>,
    ) {
        self.c_printer.standard_loop(fun, dim, cfgs, namer);
    }
}
