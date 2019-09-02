use crate::NameGenerator;
use itertools::Itertools;
use std::fmt::Write as WriteFmt;
use telamon::codegen::*;
use telamon::ir::{self, Type};
use telamon::search_space::{DimKind, Domain};
use telamon_c::C99Display as _;
use utils::unwrap;
// TODO(cc_perf): avoid concatenating strings.

#[derive(Default)]
pub struct MppaPrinter {
    buffer: String,
}

fn param_t(param: &ParamVal) -> String {
    match param {
        &ParamVal::External(ref param, par_type) => {
            if let Some(elem_t) = param.elem_t {
                format!("{}*", elem_t.c99())
            } else {
                par_type.c99().to_string()
            }
        }
        ParamVal::Size(_) => "uint32_t".to_string(),
        ParamVal::GlobalMem(_, _, par_type) => format!("{}*", par_type.c99()),
    }
}

impl MppaPrinter {
    /// Declares all parameters of the function with the appropriate type
    fn param_decl(&self, param: &ParamVal) -> String {
        format!("{} {}", param_t(param), param.key().ident())
    }

    /// Declared all variables that have been required from the namegen
    fn var_decls(&self, namegen: &NameGenerator) -> String {
        let print_decl = |(&t, &n)| {
            // Type is never supposed to be PtrTo here as we handle ptr types in a different way
            if let ir::Type::PtrTo(..) = t {
                unreachable!("Type PtrTo are never inserted in this map");
            }
            let prefix = NameGenerator::gen_prefix(t);
            let mut s = format!("{} ", t.c99());
            s.push_str(
                &(0..n)
                    .map(|i| format!("{}{}", prefix, i))
                    .collect_vec()
                    .join(", "),
            );
            s.push_str(";\n  ");
            s
        };
        let other_var_decl = namegen.num_var.iter().map(print_decl).join("\n  ");
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
    fn decl_par_indexes(&self, function: &Function, name_map: &NameMap<'_>) -> String {
        assert!(function.block_dims().is_empty());
        let mut decls = vec![];
        // Compute thread indexes.
        for (ind, dim) in function.thread_dims().iter().enumerate() {
            decls.push(format!(
                "{} = tid->t{};\n",
                name_map.name_index(dim.id()).name(),
                ind
            ));
        }
        decls.join("\n  ")
    }

    /// Prints a `Function`.
    fn function<'a: 'b, 'b>(&mut self, function: &'b Function<'a>) -> String {
        let mut namegen = NameGenerator::default();
        let interner = Interner::default();
        let name_map = &mut NameMap::new(&interner, function, &mut namegen);

        let param_decls = function
            .device_code_args()
            .map(|v| self.param_decl(v))
            .join(",\n  ");
        // SIGNATURE AND OPEN BRACKET
        let mut return_string = format!(
            include_str!("template/signature.c.template"),
            name = function.name(),
            params = param_decls
        );
        // INDEX LOADS
        let idx_loads = self.decl_par_indexes(function, name_map);
        unwrap!(writeln!(self.buffer, "{}", idx_loads));
        // LOAD PARAM
        for val in function.device_code_args() {
            let var_name = name_map.name_param_val(val.key());
            unwrap!(writeln!(
                self.buffer,
                "{var_name} = {cast}{name}; // {param}",
                cast = if val.elem_t().is_some() {
                    format!("({})", var_name.t().c99())
                } else {
                    "".to_string()
                },
                var_name = var_name.c99(),
                name = val.key().ident(),
                param = val.key(),
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
                        self.print_inst(
                            llir::Instruction::cast(level.t(), reg, old_name)
                                .unwrap()
                                .into(),
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
    fn fun_params_cast(&self, function: &Function) -> String {
        function
            .device_code_args()
            .enumerate()
            .map(|(i, v)| match v {
                ParamVal::External(..) => format!(
                    "  {t} {p} = *({t}*)args[{i}];",
                    t = param_t(v),
                    p = v.key().ident(),
                    i = i
                ),
                ParamVal::Size(_) => format!(
                    r#"  uint32_t {p} = *(uint32_t*)(void *)args[{i}];
//printf("{p} = %d\n", {p});"#,
                    p = v.key().ident(),
                    i = i
                ),
                // Are we sure we know the size at compile time ? I think we do
                ParamVal::GlobalMem(_, _, par_type) => format!(
                    "  {t} {p} = *({t}*)args[{i}];",
                    p = v.key().ident(),
                    t = par_type.c99(),
                    i = i
                ),
            })
            .join("\n")
    }

    /// Declares the variables that will be used in C function call
    fn params_call(&mut self, function: &Function) -> String {
        function
            .device_code_args()
            .format_with(", ", |p, f| f(&format_args!("{}", p.key().ident())))
            .to_string()
    }

    /// Build the right call for a nested loop on dimensions with linearized accesses that is, for
    /// a 3 dimensional arrays a[2][5][3] returns d0 + d1 * 3 + d2 * 5
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

    /// Helper for building a structure containing as many thread ids (one id per dim) as required.
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
                "tids[{index}].t{dim_id} = d{dim_id};\n",
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

    /// Turns the argument of wrapper into an array of void pointers
    /// Necessary to call pthread with it
    fn build_ptr_struct(&self, func: &Function) -> String {
        func.device_code_args()
            .enumerate()
            .map(|(i, arg)| format!("args[{}] = (void *)&{}", i, arg.key().ident()))
            .join(";\n")
    }

    /// wrap the kernel call into a function with a fixed interface
    pub fn wrapper_function<'a: 'b, 'b>(
        &mut self,
        func: &'b Function<'a>,
        id: usize,
    ) -> String {
        let fun_str = self.function(func);
        let fun_params = self.params_call(func);
        let (lower_bound, upper_n_arg) = func.device_code_args().size_hint();
        let n_args = if let Some(upper_bound) = upper_n_arg {
            assert_eq!(upper_bound, lower_bound);
            upper_bound
        } else {
            20
        };
        let cl_arg_def = func
            .device_code_args()
            .map(|v| self.param_decl(v))
            .join(",  ");
        format!(
            include_str!("template/host.c.template"),
            id = id,
            cl_arg_def = cl_arg_def,
            n_arg = n_args,
            build_ptr_struct = self.build_ptr_struct(func),
            fun_name = func.name(),
            fun_str = fun_str,
            fun_params_cast = self.fun_params_cast(func),
            fun_params = fun_params,
            gen_threads = self.thread_gen(func),
            dim_decl = self.build_thread_id_struct(func),
            thread_join = self.thread_join(func),
        )
    }

    /// Returns the name of a type.
    fn type_name(t: ir::Type) -> &'static str {
        match t {
            ir::Type::PtrTo(..) => "void*",
            ir::Type::F(32) => "float",
            ir::Type::F(64) => "double",
            ir::Type::I(1) => "bool",
            ir::Type::I(8) => "uint8_t",
            ir::Type::I(16) => "uint16_t",
            ir::Type::I(32) => "uint32_t",
            ir::Type::I(64) => "uint64_t",
            _ => panic!("non-printable type"),
        }
    }

    /// Returns the name of a type.
    fn cl_type_name(t: ir::Type) -> &'static str {
        match t {
            ir::Type::PtrTo(..) => "__global void*",
            ir::Type::I(8) => "char",
            ir::Type::I(16) => "short",
            ir::Type::I(32) => "int",
            ir::Type::I(64) => "long",
            _ => Self::type_name(t),
        }
    }
    /// Prints the OpenCL wrapper for a candidate implementation.
    pub fn print_ocl_wrapper(&self, fun: &Function, id: usize) -> String {
        let arg_names = fun
            .device_code_args()
            .format_with(", ", |p, f| f(&format_args!("{}", p.key().ident())));
        let cl_arg_defs = fun.device_code_args().format_with(", ", |p, f| {
            f(&format_args!(
                "{} {}",
                Self::cl_type_name(p.t()),
                p.key().ident(),
            ))
        });
        format!(
            include_str!("template/ocl_wrap.c.template"),
            fun_id = id,
            arg_names = arg_names,
            cl_arg_defs = cl_arg_defs,
        )
    }
}

impl InstPrinter for MppaPrinter {
    fn print_label(&mut self, label: llir::Label<'_>) {
        writeln!(self.buffer, "{}", label.c99()).unwrap()
    }

    fn print_inst(&mut self, inst: llir::PredicatedInstruction<'_>) {
        writeln!(self.buffer, "{}", inst.c99()).unwrap();
    }
}
