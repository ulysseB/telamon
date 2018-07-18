//! Provides functions to print PTX code.
use device::cuda::{Gpu, Namer};
use codegen::Printer;
use codegen::*;
use ir::{self, op, Type};
use itertools::Itertools;
use search_space::{DimKind, Domain, InstFlag};
use std;
use std::io::Write;
use std::fmt::Write as WriteFmt;

pub struct CudaPrinter {
    out_function: String,
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
                    Type::I( n / 2)
                } else {
                    panic!("get_inst_type should only be called with integer types")
                }
            },
            _ => ret_type,
        }
    }

    /// Prints a load operator.
    fn ld_operator(flag: InstFlag) -> &'static str {
        match flag {
            InstFlag::MEM_SHARED => "ld.shared",
            InstFlag::MEM_CA => "ld.global.ca",
            InstFlag::MEM_CG => "ld.global.cg",
            InstFlag::MEM_CS => "ld.global.cs",
            InstFlag::MEM_NC => "ld.global.nc",
            _ => panic!("invalid load flag {:?}", flag),
        }
    }

    /// Prints a store operator.
    fn st_operator(flag: InstFlag) -> &'static str {
        match flag {
            InstFlag::MEM_SHARED => "st.shared",
            InstFlag::MEM_CA => "st.global.wb",
            InstFlag::MEM_CG => "st.global.cg",
            InstFlag::MEM_CS => "st.global.cs",
            _ => panic!("invalid store flag {:?}", flag),
        }
    }

    /// Prints the variables declared by the `Namer`.
    fn var_decls(&mut self, namer: &Namer) -> String {
        let print_decl = |(&t, n)| {
            let prefix = Namer::gen_prefix(&t);
            format!(".reg.{} %{}<{}>;", Self::get_type(t), prefix, n)
        };
        namer.num_var.iter().map(print_decl).collect_vec().join("\n  ")
    }

    /// Declares block and thread indexes.
    fn decl_par_indexes(function: &Function, namer: &mut NameMap) -> String {
        let mut decls = vec![];
        // Load block indexes.
        for (dim, dir) in function.block_dims().iter().zip(&["x", "y", "z"])  {
            let index = namer.name_index(dim.id());
            decls.push(format!("mov.u32 {}, %ctaid.{};", index, dir));
        }
        // Compute thread indexes.
        for (dim, dir) in function.thread_dims().iter().rev().zip(&["x", "y", "z"]) {
            decls.push(format!("mov.s32 {}, %tid.{};", namer.name_index(dim.id()), dir));
        }
        decls.join("\n  ")
    }

    /// Declares a shared memory block.
    fn shared_mem_decl(&mut self, block: &InternalMemBlock, namer: &mut NameMap)  {
        let ptr_type_name = Self::get_type(Type::I(32));
        let name = namer.name_addr(block.id());
        unwrap!(writeln!(self.out_function, ".shared.align 16 .u8 {vec_name}[{size}];\
            \n  mov.{t} {name}, {vec_name};\n",
            vec_name = &name[1..],
            name = name,
            t = ptr_type_name,
            size = unwrap!(block.alloc_size().as_int())));
    }

    pub fn new() -> Self {
        CudaPrinter{out_function: String::new() }
    }

    /// Prints a `Type` for the host.
    fn host_type(t: &Type) -> &'static str {
        match *t {
            Type::PtrTo(..) => "CUdeviceptr",
            Type::F(32) => "float",
            Type::F(64) => "double",
            Type::I(8) => "int8_t",
            Type::I(16) => "int16_t",
            Type::I(32) => "int32_t",
            Type::I(64) => "int64_t",
            ref t => panic!("invalid type for the host: {}", t)
        }
    }

    /// Returns the string representation of thread and block sizes on the host.
    fn host_3sizes<'a, IT>(dims: IT) -> [String; 3]
        where IT: Iterator<Item=&'a Dimension<'a>>  + 'a {
            let mut sizes = ["1".to_string(), "1".to_string(), "1".to_string()];
            for (i, d) in dims.into_iter().enumerate() {
                assert!(i < 3);
                sizes[i] = Self::host_size(d.size())
            }
            sizes
        }

    /// Prints a size on the host.
    fn host_size(size: &Size) -> String {
        let dividend = size.dividend().iter().map(|p| format!("* {}", &p.name));
        format!("{}{}/{}", size.factor(), dividend.format(""), size.divisor())
    }

    fn binary_op(op: ir::BinOp) -> &'static str {
        match op {
            ir::BinOp::Add => "add",
            ir::BinOp::Sub => "sub",
            ir::BinOp::Div => "div",
        }
    }

    /// Prints a parameter decalartion.
    fn param_decl(&mut self, param: &ParamVal, namer: &NameMap) -> String {
        format!(
            ".param .{t}{attr} {name}",
            t = Self::get_type(param.t()),
            attr = if param.is_pointer() { ".ptr.global.align 16" } else { "" },
            name = namer.name_param(param.key()),
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
        let mut namer = Namer::default();
        let param_decls;
        {
            let name_map = &mut NameMap::new(function, &mut namer);
            param_decls = function.device_code_args()
                .map(|v| self.param_decl(v, name_map))
                .collect_vec().join(",\n  ");
            // LOAD PARAMETERS
            for val in function.device_code_args() {
                unwrap!(writeln!(self.out_function, "ld.param.{t} {var_name}, [{name}];",
                        t = Self::get_type(val.t()),
                        var_name = name_map.name_param_val(val.key()),
                        name = name_map.name_param(val.key())));
            }
            // INDEX LOAD
            let idx_loads = Self::decl_par_indexes(function, name_map);
            self.out_function.push_str(&idx_loads);
            self.out_function.push_str(&"\n");
            //MEM DECL
            for block in function.mem_blocks() {
                match block.alloc_scheme() {
                    AllocationScheme::Shared =>
                        self.shared_mem_decl(block, name_map),
                    AllocationScheme::PrivatisedGlobal =>
                        self.privatise_global_block(block, name_map, function),
                    AllocationScheme::Global => (),
                }
            }
            // Compute size casts
            for dim in function.dimensions() {
                if !dim.kind().intersects(DimKind::UNROLL | DimKind::LOOP) { continue; }
                for level in dim.induction_levels() {
                    if let Some((_, ref incr)) = level.increment {
                        let name = name_map.declare_size_cast(incr, level.t());
                        if let Some(name) = name {
                            let old_name = name_map.name_size(incr, Type::I(32));
                            self.print_cast(Type::I(32), level.t(), op::Rounding::Exact,
                                            &name, &old_name);
                        }
                    }
                }
            }
            let ind_levels = function.init_induction_levels().into_iter()
                .chain(function.block_dims().iter().flat_map(|d| d.induction_levels()));
            for level in ind_levels {
                self.parallel_induction_level(level, name_map);
            }
            self.cfg(function, function.cfg(), name_map);
        }
        let var_decls = self.var_decls(&namer);
        let mut body = String::new();
        body.push_str(&var_decls);
        self.out_function.push_str(&"\n");
        body.push_str(&self.out_function);
        format!(include_str!("template/device.ptx"),
        sm_major = gpu.sm_major,
        sm_minor = gpu.sm_minor,
        addr_size = gpu.addr_size,
        name = function.name,
        params = param_decls,
        num_thread = function.num_threads(),
        body = body
        )
    }

    pub fn host_function(&mut self, fun: &Function, gpu: &Gpu, out: &mut Write) {
        let block_sizes = Self::host_3sizes(fun.block_dims().iter());
        let thread_sizes = Self::host_3sizes(fun.thread_dims().iter().rev());
        let extern_param_names =  fun.params.iter()
            .map(|x| &x.name as &str).collect_vec().join(", ");
        let mut next_extra_var_id = 0;
        let mut extra_def = vec![];
        let mut extra_cleanup = vec![];
        let params = fun.device_code_args().map(|p| match *p {
            ParamVal::External(p, _) => format!("&{}", p.name),
            ParamVal::Size(ref size) => {
                let extra_var = format!("_extra_{}", next_extra_var_id);
                next_extra_var_id += 1;
                extra_def.push(format!("int32_t {} = {};", extra_var, Self::host_size(size)));
                format!("&{}", extra_var)
            },
            ParamVal::GlobalMem(_, ref size, _) => {
                let extra_var = format!("_extra_{}", next_extra_var_id);
                next_extra_var_id += 1;
                let size = Self::host_size(size);
                extra_def.push(format!("CUDeviceptr {};", extra_var));
                extra_def.push(format!("CHECK_CUDA(cuMemAlloc(&{}, {}));", extra_var, size));
                extra_cleanup.push(format!("CHECK_CUDA(cuMemFree({}));", extra_var));
                format!("&{}", extra_var)
            },
        }).collect_vec().join(", ");
        let extern_params = fun.params.iter()
            .map(|p| format!("{} {}", Self::host_type(&p.t), p.name))
            .collect_vec().join(", ");
        let res = write!(out, include_str!("template/host.c"),
        name = fun.name,
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

impl Printer for CudaPrinter {

    /// Get a proper string representation of an integer in target language
    fn get_int(n: u32) -> String {
        format!("{}", n)
    }

    /// Get a proper string representation of an integer in target language
    fn get_float(f: f64) -> String {
        let binary = unsafe { std::mem::transmute::<f64, u64>(f) };
        format!("0D{:016X}", binary)
    }

    /// Print a type in the backend
    fn get_type(t: Type) -> String {
       match t {
        Type::I(1) => "pred".to_string(),
        Type::I(size) => format!("s{size}", size = size),
        Type::F(size) => format!("f{size}", size = size),
        _ => panic!()
    }
 }

    /// Print return_id = op1 op op2
    fn print_binop(&mut self, op: ir::BinOp,
                   return_type: Type,
                   rounding: op::Rounding,
                   return_id: &str, lhs: &str, rhs: &str) {
        let op = Self::binary_op(op);
        let rounding = Self::rounding(rounding);
        let ret_type = Self::get_type(return_type);
        unwrap!(writeln!(self.out_function, "{}{}.{} {}, {}, {};",
                         op, rounding, ret_type, return_id, lhs, rhs));
    }

    /// Print return_id = op1 * op2
    fn print_mul(&mut self, return_type: Type,
                 round: op::Rounding,
                 mul_mode: MulMode,
                 return_id: &str,
                 lhs: &str, rhs: &str) {
        let operator = if round == op::Rounding::Exact {
            format!("mul{}", Self::mul_mode(mul_mode))
        } else {
            format!("mul{}", Self::rounding(round))
        };
        let t = Self::get_type(Self::get_inst_type(mul_mode, return_type));
        unwrap!(writeln!(self.out_function, "{}.{} {}, {}, {};",
                         operator, t, return_id, lhs, rhs));
    }

    /// Print return_id = mlhs * mrhs + arhs
    fn print_mad(&mut self, return_type: Type,
                 round: op::Rounding,
                 mul_mode: MulMode,
                 return_id: &str,
                 mlhs: &str, mrhs: &str, arhs: &str) {
        let operator = if round == op::Rounding::Exact {
            format!("mad{}", Self::mul_mode(mul_mode))
        } else {
            format!("fma{}", Self::rounding(round))
        };
        let t = Self::get_type(Self::get_inst_type(mul_mode, return_type));
        unwrap!(writeln!(self.out_function, "{}.{} {}, {}, {}, {};",
                         operator, t, return_id, mlhs, mrhs, arhs));
    }

    /// Print return_id = op
    fn print_mov(&mut self, return_type: Type, return_id: &str, op: &str) {
        unwrap!(writeln!(self.out_function, "mov.{} {}, {};",
                         Self::get_type(return_type), return_id, op));
    }

    /// Print return_id = load [addr]
    fn print_ld(&mut self, return_type: Type, flag: InstFlag, return_id: &str,  addr: &str) {
        let operator = Self::ld_operator(flag);
        unwrap!(writeln!(self.out_function, "{}.{} {}, [{}];",
                         operator, Self::get_type(return_type), return_id,  addr));
    }

    /// Print store val [addr]
    fn print_st(&mut self, val_type: Type, mem_flag: InstFlag, addr: &str, val: &str) {
        let operator = Self::st_operator(mem_flag);
        unwrap!(writeln!(self.out_function, "{}.{} [{}], {};",
                         operator, Self::get_type(val_type), addr, val));
    }

    /// Print if (cond) store val [addr]
    fn print_cond_st(&mut self, val_type: Type,
                     mem_flag: InstFlag,
                     cond: &str, addr: &str, val: &str) {
        let operator = Self::st_operator(mem_flag);
        unwrap!(writeln!(self.out_function, "@{} {}.{} [{}], {};",
                         cond, operator, Self::get_type(val_type), addr, val));
    }

    /// Print return_id = (val_type) val
    fn print_cast(&mut self, from_t: Type, to_t: Type,
                  round: op::Rounding,
                  return_id: &str,
                  op1: &str) {
        let rounding = Self::rounding(round);
        let to_t = Self::get_type(to_t);
        let from_t = Self::get_type(from_t);
        let operator = format!("cvt{}.{}.{}", rounding, to_t, from_t);
        unwrap!(writeln!(self.out_function, "{} {}, {};",  operator, return_id, op1));
    }

    /// print a label where to jump
    fn print_label(&mut self, label_id: &str) {
        unwrap!(writeln!(self.out_function, "LOOP_{}:", label_id));
    }

    /// Print return_id = op1 && op2
    fn print_and(&mut self, return_id: &str, op1: &str, op2: &str) {
        unwrap!(writeln!(self.out_function, "and.pred {}, {}, {};", return_id, op1, op2));
    }

    /// Print return_id = op1 || op2
    fn print_or(&mut self, return_id: &str, op1: &str, op2: &str) {
        unwrap!(writeln!(self.out_function, "or.pred {}, {}, {};", return_id, op1, op2));
    }

    /// Print return_id = op1 == op2
    fn print_equal(&mut self, return_id: &str, op1: &str, op2: &str) {
        unwrap!(writeln!(self.out_function, "setp.eq.u32 {}, {}, {};", return_id, op1, op2));
    }

    /// Print return_id = op1 < op2
    fn print_lt(&mut self, return_id: &str, op1: &str, op2: &str) {
        unwrap!(writeln!(self.out_function, "setp.lt.u32 {}, {}, {};", return_id, op1, op2));
    }

    /// Print return_id = op1 > op2
    fn print_gt(&mut self, return_id: &str, op1: &str, op2: &str) {
        unwrap!(writeln!(self.out_function, "setp.gt.u32 {}, {}, {};", return_id, op1, op2));
    }

    /// Print if (cond) jump label(label_id)
    fn print_cond_jump(&mut self, label_id: &str, cond: &str) {
        unwrap!(writeln!(self.out_function, "@{} bra.uni LOOP_{};", cond, label_id));
    }

    /// Print wait on all threads
    fn print_sync(&mut self) {
        unwrap!(writeln!(self.out_function, "bar.sync 0;"));
    }

    fn print_vector_inst(&mut self, inst: &Instruction,
                         dim: &Dimension,
                         namer: &mut NameMap,
                         fun: &Function) {
        let size = unwrap!(dim.size().as_int());
        let flag = unwrap!(inst.mem_flag());
        match *inst.operator() {
            op::Ld(_, ref addr, _) => {
                let operator = format!("{}.v{}", Self::ld_operator(flag), size);
                let dst = (0..size).map(|i| {
                    namer.indexed_inst_name(inst, dim.id(), i).to_string()
                }).collect_vec().join(", ");
                let t = Self::get_type(unwrap!(inst.t()));
                unwrap!(writeln!(self.out_function, "{}.{} {{{}}}, [{}];",
                                 operator, t, dst, namer.name_op(addr)))
            },
            op::St(ref addr, ref val, _, _) => {
                let operator = format!("{}.v{}", Self::st_operator(flag), size);
                let guard = if inst.has_side_effects() {
                    if let Some(pred) = namer.side_effect_guard() {
                        format!("@{} ", pred)
                    } else { String::new() }
                } else { String::new() };
                let t = Self::get_type(Self::lower_type(val.t(), fun));
                let src = (0..size).map(|i| {
                    namer.indexed_op_name(val, dim.id(), i).to_string()
                }).collect_vec().join(", ");
                unwrap!(writeln!(self.out_function, "{}{}.{} [{}], {{{}}};",
                                 guard, operator, t, namer.name_op(addr), src));
            },
            _ => panic!("non-vectorizable instruction"),
        }
    }
}
