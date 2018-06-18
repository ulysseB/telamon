
use codegen::*;
use device::x86::Namer;
use ir::{self, op, Type};
use itertools::Itertools;
use search_space::{Domain, DimKind};
use std::fmt::Write as WriteFmt;
use utils::*;
use device::printer::{self, Printer};
// TODO(cc_perf): avoid concatenating strings.

struct X86_printer {
    out_function: String,
}

impl X86_printer {
    fn param_decl(&mut self, param: &ParamVal, namer: &NameMap) -> String {
        let name = namer.name_param(param.key());
        match param {
            ParamVal::External(_, par_type) => format!("{} {}", cpu_type(par_type), name),
            ParamVal::Size(_) => format!("uint32_t {}", name),
            ParamVal::GlobalMem(_, _, par_type) => format!("{} {}", cpu_type(par_type), name),
        }
    }





    fn var_decls(&mut self, namer: &Namer) -> String {
        let print_decl = |(&t, &n)| {
            match t {
                ir::Type::PtrTo(..) => String::new(),
                _ => {
                    let prefix = Namer::gen_prefix(&t);
                    let mut s = format!("{} ", cpu_type(&t));
                    s.push_str(&(0..n).map(|i| format!("{}{}", prefix, i)).collect_vec().join(", "));
                    s.push_str(";\n  ");
                    s
                }
            }
        };
        let mut ptr_decl = String::from("intptr_t  ");
        ptr_decl.push_str(&(0..namer.num_glob_ptr).map( |i| format!("ptr{}", i)).collect_vec().join(", "));
        ptr_decl.push_str(&";\n");
        let other_var_decl = namer.num_var.iter().map(print_decl).collect_vec().join("\n  ");
        ptr_decl.push_str(&other_var_decl);
        ptr_decl
    }

    /// Declares block and thread indexes.
    fn decl_par_indexes(&mut self, function: &Function, namer: &mut NameMap) -> String {
        assert!(function.block_dims().is_empty());
        let mut decls = vec![];
        // Compute thread indexes.
        for (ind, dim) in function.thread_dims().iter().enumerate() {
            //FIXME: fetch proper thread index
            decls.push(format!("{} = tid.t{};", namer.name_index(dim.id()), ind));
        }
        decls.join("\n  ")
    }


    /// Prints a `Function`.
    pub fn function(&mut self, function: &Function) -> String {
        let mut namer = Namer::default();
        let (param_decls, body, ld_params, idx_loads, mem_decls);
        let mut init = Vec::new();
        {
            let name_map = &mut NameMap::new(function, &mut namer);
            param_decls = function.device_code_args()
                .map(|v| Self::param_decl(v, name_map))
                .collect_vec().join(",\n  ");
            idx_loads = Self::decl_par_indexes(function, name_map);
            ld_params = function.device_code_args().map(|val| {
                format!("{var_name} = {name};",
                        var_name = name_map.name_param_val(val.key()),
                        name = name_map.name_param(val.key()))
            }).collect_vec().join("\n  ");
            mem_decls = function.mem_blocks().flat_map(|block| {
                match block.alloc_scheme() {
                    AllocationScheme::Shared =>
                        panic!("No shared mem in cpu!!"),
                    AllocationScheme::PrivatisedGlobal =>
                        Some(printer::privatise_global_block(self, block, name_map, function)),
                    AllocationScheme::Global => None,
                }
            }).format("\n  ").to_string();
            // Compute size casts
            for dim in function.dimensions() {
                if !dim.kind().intersects(DimKind::UNROLL | DimKind::LOOP) { continue; }
                for level in dim.induction_levels() {
                    if let Some((_, incr)) = level.increment {
                        let name = name_map.declare_size_cast(incr, level.t());
                        if let Some(name) = name {
                            let cpu_t = cpu_type(&level.t());
                            let old_name = name_map.name_size(incr, Type::I(32));
                            init.push(format!("{} = ({}){};", name, cpu_t, old_name));
                        }
                    }
                }
            }
            let ind_levels = function.init_induction_levels().into_iter()
                .chain(function.block_dims().iter().flat_map(|d| d.induction_levels()));
            init.extend(ind_levels.map(|level| parallel_induction_level(level, name_map)));
            body = cfg(function, function.cfg(), name_map);
        }
        let var_decls = var_decls(&namer);
        format!(include_str!("template/device.c.template"),
        name = function.name,
        idx_loads = idx_loads,
        ld_params = ld_params,
        params = param_decls,
        var_decls = var_decls,
        mem_decls = mem_decls,
        init = init.join("\n  "),
        body = body
        )
    }

    fn fun_params_cast(&mut self, function: &Function) -> String {
        function.device_code_args()
            .enumerate()
            .map(|(i, v)| match v {
                ParamVal::External(..) if v.is_pointer() => format!("intptr_t p{i} = (intptr_t)*(args + {i})", 
                                                                    i = i),
                                                                    ParamVal::External(_, par_type) => format!("{t} p{i} = *({t}*)*(args + {i})", 
                                                                                                               t = cpu_type(par_type), i = i),
                                                                    ParamVal::Size(_) => format!("uint32_t p{i} = *(uint32_t*)*(args + {i})", i = i),
                                                                    // Are we sure we know the size at compile time ? I think we do
                                                                    ParamVal::GlobalMem(_, _, par_type) => format!("{t} p{i} = ({t})*(args + {i})", 
                                                                                                                   t = cpu_type(par_type), i = i)
            }
            )
            .collect_vec()
            .join(";\n  ")
    } 

    fn params_call(&mut self, function: &Function) -> String {
        function.device_code_args()
            .enumerate().map(|x| x.0)
            .map(|i| format!("p{}", i))
            .collect_vec()
            .join(", ")
    }

    // Build the right call for a nested loop on dimensions
    fn build_index_call(&mut self, func: &Function) -> String {
        let mut vec_ret = vec![];
        let dims = func.thread_dims();
        let n = dims.len();
        for i in 0..n {
            let start = format!("d{}", i);
            let mut vec_str = vec![start];
            for j in 0.. i  {
                vec_str.push(format!("{}", unwrap!(dims[j].size().as_int())));
            }
            vec_ret.push(vec_str.join(" * "));
        }
        vec_ret.join(" + ")
    }

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

    fn thread_gen(&mut self, func: &Function) -> String {
        if func.num_threads() == 1 {
            let mut ret = format!("thread_arg_t thread_args;\n");
            ret.push_str(&format!(" thread_args.args = args;\n"));
            ret.push_str(&format!(" thread_args.tid.t0 = 0;\n"));
            ret.push_str(&format!(" thread_args.tid.barrier = &barrier;\n"));
            ret.push_str(&format!("pthread_barrier_init(&barrier, NULL,{});\n",   func.num_threads()));
            ret.push_str(&format!("exec_wrap((void *)&thread_args);\n"));
            return ret;
        }
        let mut ret = format!("pthread_t thr_ids[{}];\n", func.num_threads());
        let mut ind_var_decl = String::from("int ");
        let build_struct = format!("thread_arg_t thread_args[{}];\n", func.num_threads());
        let dim_tid_struct = format!("thread_dim_id_t thread_tids[{}];\n", func.num_threads());
        let barrier_init = format!("pthread_barrier_init(&barrier, NULL,{});\n",   func.num_threads() );
        let mut loop_decl = String::new();
        let mut ind_vec = Vec::new();
        let mut jmp_stack = Vec::new();
        for (ind, dim) in func.thread_dims().iter().enumerate() {
            let mut loop_jmp = String::new();
            ind_vec.push(format!("d{}", ind));
            loop_decl.push_str(&format!("d{}=0;\n", ind));
            loop_decl.push_str(&format!("LOOP_BEGIN_{}:\n", ind));
            loop_jmp.push_str(&format!("d{}++;\n", ind));
            loop_jmp.push_str(&format!("if (d{} < {})\n", ind, unwrap!(dim.size().as_int())));
            loop_jmp.push_str(&format!("    goto LOOP_BEGIN_{};\n", ind));
            jmp_stack.push(loop_jmp);
        }
        let ind_dec_inter = ind_vec.join(", ");
        ind_var_decl.push_str(&ind_dec_inter);
        ind_var_decl.push_str(&";\n");
        let mut loop_jmp = String::new(); 
        while let Some(j_str) = jmp_stack.pop() {
            loop_jmp.push_str(&j_str);
        }
        let arg_struct = format!("thread_args[{ind}].args = args;\n",  ind = build_index_call(func) );
        let mut tid_struct = String::new();
        for (ind, _) in func.thread_dims().iter().enumerate() {
            tid_struct.push_str(&format!("thread_args[{index}].tid.t{dim_id} = d{dim_id};\n",  index = build_index_call(func), dim_id = ind));
        }
        let barrier_str = format!("thread_args[{}].tid.barrier = &barrier;\n",  build_index_call(func) );
        let create_call = format!("pthread_create(&thr_ids[{}], NULL, exec_wrap, (void *)&thread_args[{ind}]);\n",  ind = build_index_call(func) );
        ret.push_str(&ind_var_decl);
        ret.push_str(&build_struct);
        ret.push_str(&dim_tid_struct);
        ret.push_str(&barrier_init);
        ret.push_str(&loop_decl);
        ret.push_str(&arg_struct);
        ret.push_str(&tid_struct);
        ret.push_str(&barrier_str);
        ret.push_str(&create_call);
        ret.push_str(&loop_jmp);
        ret
    }

    fn thread_join(&mut self, func: &Function) -> String {
        if func.num_threads() == 1 {
            return String::new();
        }
        let mut ret = String::new();
        let mut loop_decl = String::new();
        let mut jmp_stack = Vec::new();
        for (ind, dim) in func.thread_dims().iter().enumerate() {
            let mut loop_jmp = String::new();
            loop_decl.push_str(&format!("d{} = 0;\n", ind));
            loop_decl.push_str(&format!("JOIN_LOOP_BEGIN_{}:\n", ind));
            loop_jmp.push_str(&format!("d{}++;\n", ind));
            loop_jmp.push_str(&format!("if (d{} < {})\n", ind, unwrap!(dim.size().as_int())));
            loop_jmp.push_str(&format!("    goto JOIN_LOOP_BEGIN_{};\n", ind));
            jmp_stack.push(loop_jmp);
        }
        let mut loop_jmp = String::new();
        while let Some(j_str) = jmp_stack.pop() {
            loop_jmp.push_str(&j_str);
        }
        let join_call = format!("pthread_join(thr_ids[{}], NULL);\n", build_index_call(func) );
        let barrier_destroy = format!("pthread_barrier_destroy(&barrier);\n");
        ret.push_str(&loop_decl);
        ret.push_str(&join_call);
        ret.push_str(&loop_jmp);
        ret.push_str(&barrier_destroy);
        ret

    }

    pub fn wrapper_function(&mut self, func: &Function) -> String {
        let fun_str = function(func);
        let fun_params = params_call(func);
        format!(include_str!("template/host.c.template"),
        fun_name = func.name,
        fun_str = fun_str,
        fun_params_cast = fun_params_cast(func),
        fun_params = fun_params,
        gen_threads = thread_gen(func),
        dim_decl = build_thread_id_struct(func),
        thread_join = thread_join(func),
        )
    }
}

impl Printer for X86_printer {
    fn get_int(&self, n: u32) -> String {
        format!("{}", n)
    }

    fn get_type(&mut self, t: &ir::Type) -> &'static str {
        match *t {
            Type::Void => "void",
            //Type::PtrTo(..) => " uint8_t *",
            Type::PtrTo(..) => "intptr_t",
            Type::F(32) => "float",
            Type::F(64) => "double",
            Type::I(1) => "int8_t",
            Type::I(8) => "int8_t",
            Type::I(16) => "int16_t",
            Type::I(32) => "int32_t",
            Type::I(64) => "int64_t",
            ref t => panic!("invalid type for the host: {}", t)
        }
    }

    fn print_binop(&mut self, return_id: &str, op_type: ir::BinOp, op1: &str, op2: &str) {
        let push_str = match op_type {
            ir::BinOp::Add => format!("{} = {} + {};\n", return_id, op1, op2),
            ir::BinOp::Sub => format!("{} = {} - {};\n", return_id, op1, op2),
            ir::BinOp::Div => format!("{} = {} / {};\n", return_id, op1, op2),
        }
        self.out_function.push_str(&push_str);
    }

    fn print_mul(&mut self, return_id: &str, _: op::Rounding, op1: &str, op2: &str) {
        let push_str = format!("{} = {} * {};\n", return_id, op1, op2);
        self.out_function.push_str(&push_str);
    }

    fn print_mad(&mut self, return_id: &str, _: op::Rounding, op1: &str, op2: &str, op3: &str) {
        let push_str = format!("{} = {} * {} + {};\n", return_id, op1, op2, op3);
        self.out_function.push_str(&push_str);
    }

    fn print_mov(&mut self, return_id: &str, op: &str) {
        let push_str = format!("{} = {} ;\n", return_id, op);
        self.out_function.push_str(&push_str);
    }

    fn print_ld(&mut self, return_id: &str, cast_type: &str,  addr: &str) {
        let push_str = format!("{} = ({})*{} ;\n", return_id, cast_type, addr);
        self.out_function.push_str(&push_str);
    }

    fn print_st(&mut self, addr: &str, val: &str) {
        let push_str = format!("*{} = {} ;\n", addr, val);
        self.out_function.push_str(&push_str);
    }

    fn print_cond_st(&mut self, addr: &str, val: &str, cond: &str) {
        let push_str = format!("if ({}) *{} = {} ;\n", cond, addr, val);
        self.out_function.push_str(&push_str);
    }

    fn print_cast(&mut self, return_id: &str, op1: &str, t: &Type) {
        let push_str = format!("{} = ({}) {};\n", return_id, self.get_type(t), op1);
        self.out_function.push_str(&push_str);
    }

    fn print_label(&mut self, label_id: &str) {
        let push_str = format!("LABEL_{}:", label_id);
        self.out_function.push_str(&push_str);
    }

    fn print_and(&mut self, return_id: &str, op1: &str, op2: &str){
        let push_str = format!("{} = {} && {};\n", return_id, op1, op2);
        self.out_function.push_str(&push_str);
    }

    fn print_or(&mut self, return_id: &str, op1: &str, op2: &str){
        let push_str = format!("{} = {} || {};\n", return_id, op1, op2);
        self.out_function.push_str(&push_str);
    }

    fn print_equal(&mut self, return_id: &str, op1: &str, op2: &str){
        let push_str = format!("{} = {} == {};\n", return_id, op1, op2);
        self.out_function.push_str(&push_str);
    }

    fn print_lt(&mut self, return_id: &str, op1: &str, op2: &str){
        let push_str = format!("{} = {} < {};\n", return_id, op1, op2);
        self.out_function.push_str(&push_str);
    }

    fn print_gt(&mut self, return_id: &str, op1: &str, op2: &str){
        let push_str = format!("{} = {} > {};\n", return_id, op1, op2);
        self.out_function.push_str(&push_str);
    }

    fn print_cond_jump(&mut self, label_id: &str, cond: &str) {
        let push_str = format!("if({}) goto {};\n", cond, label_id);
        self.out_function.push_str(&push_str);
    }

    fn print_sync(&mut self) {
        self.out_function.push_str(&"pthread_barrier_wait(tid.barrier);\n");
    }
}

