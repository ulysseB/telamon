use crate::NameGenerator;
use itertools::Itertools;
use std::fmt::Write as WriteFmt;
use telamon::codegen::*;
use telamon::ir::Type;
use telamon::search_space::{DimKind, Domain};
use telamon_c::C99Display as _;
use utils::unwrap;
// TODO(cc_perf): avoid concatenating strings.

#[derive(Default)]
pub(crate) struct X86printer {
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

impl X86printer {
    /// Declares all parameters of the function with the appropriate type
    fn param_decl(&self, param: &ParamVal) -> String {
        format!("{} {}", param_t(param), param.key().ident())
    }

    /// Declared all variables that have been required from the namegen
    fn var_decls(&self, namegen: &NameGenerator) -> String {
        let print_decl = |(&t, &n)| {
            if let Type::PtrTo(..) = t {
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
                "{} = tid.t{};\n",
                name_map.name_index(dim.id()).name(),
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
        // SIGNATURE AND OPEN BRACKET
        let mut return_string = format!(
            "void {name}(
{params}
)
{{",
            name = function.name(),
            params = std::iter::once("thread_dim_id_t tid".to_string())
                .chain(function.device_code_args().map(|v| self.param_decl(v)),)
                .join(",\n  "),
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
                    "  uint32_t {p} = *(uint32_t*)args[{i}];",
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

    /// Helper for building a structure containing as many thread ids (one id per dim) as required.
    fn build_thread_id_struct(&self, func: &Function) -> String {
        if func.thread_dims().is_empty() {
            "".to_string()
        } else {
            format!(
                "int {};\n",
                (0..func.thread_dims().len())
                    .format_with(", ", |idx, f| f(&format_args!("t{}", idx)))
            )
        }
    }

    /// Prints code that generates the required number of threads, stores the handles in an array
    fn thread_gen(&self, func: &Function) -> String {
        let loop_decl =
            func.thread_dims()
                .iter()
                .enumerate()
                .format_with("\n", |(idx, dim), f| {
                    f(&format_args!(
                        "for (int d{idx} = 0; d{idx} < {size}; ++d{idx})",
                        idx = idx,
                        size = dim.size().as_int().unwrap(),
                    ))
                });

        format!(
            "pthread_t thread_ids[{num_threads}];
thread_arg_t thread_args[{num_threads}];
pthread_barrier_t barrier;

pthread_barrier_init(&barrier, NULL, {num_threads});

{loop_decl}
{{
    size_t tid = {tid};
    thread_args[tid].args = args;
    {tid_struct}
    thread_args[tid].tid.barrier = &barrier;
    pthread_create(&thread_ids[tid], NULL, exec_wrap, (void *)&thread_args[tid]);
}}

for (size_t tid = 0; tid < {num_threads}; ++tid) {{
    pthread_join(thread_ids[tid], NULL);
}}

pthread_barrier_destroy(&barrier);
",
            num_threads = func.num_threads(),
            // Build the right call for a nested loop on dimensions with linearized accesses that
            // is, for a 3 dimensions arrays a[2][5][3] returns d0 + 3 * (d1 + 5 * (d2 + 2 * (0)))
            tid = format!(
                "{}0{}",
                func.thread_dims()
                    .iter()
                    .enumerate()
                    .format_with("", |(idx, dim), f| {
                        f(&format_args!(
                            "d{} + {} * (",
                            idx,
                            dim.size().as_int().unwrap()
                        ))
                    }),
                ")".repeat(func.thread_dims().len())
            ),
            loop_decl = loop_decl,
            tid_struct = (0..func.thread_dims().len()).format_with("    ", |idx, f| {
                f(&format_args!(
                    "thread_args[tid].tid.t{idx} = d{idx};\n",
                    idx = idx,
                ))
            }),
        )
    }

    /// wrap the kernel call into a function with a fixed interface
    pub fn wrapper_function(&mut self, func: &Function) -> String {
        let fun_str = self.function(func);
        format!(
            include_str!("template/host.c.template"),
            fun_name = func.name(),
            fun_str = fun_str,
            fun_params_cast = self.fun_params_cast(func),
            // The first comma is for the `tid` argument
            fun_params = func
                .device_code_args()
                .format_with("", |p, f| f(&format_args!(", {}", p.key().ident()))),
            entry_point = self.thread_gen(func),
            dim_decl = self.build_thread_id_struct(func),
        )
    }
}

impl InstPrinter for X86printer {
    fn print_label(&mut self, label: llir::Label<'_>) {
        writeln!(self.buffer, "{}", label.c99()).unwrap()
    }

    fn print_inst(&mut self, inst: llir::PredicatedInstruction<'_>) {
        writeln!(self.buffer, "{}", inst.c99()).unwrap();
    }
}
