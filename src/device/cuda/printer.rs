//! Provides functions to print PTX code.
use codegen::*;
use device::cuda::{Gpu, Namer};
use ir::{self, op, Type};
use itertools::Itertools;
use search_space::{DimKind, Domain, InstFlag, MemSpace};
use std;
use std::borrow::Cow;
use std::fmt::Write as WriteFmt;
use std::io::Write;
use utils::*;

#[derive(Default)]
pub struct CudaPrinter {
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

    /// Prints the variables declared by the `Namer`.
    fn var_decls(&mut self, namer: &Namer) -> String {
        let print_decl = |(&t, n)| {
            let prefix = Namer::gen_prefix(&t);
            format!(".reg.{} %{}<{}>;", Self::get_type(t), prefix, n)
        };
        namer
            .num_var
            .iter()
            .map(print_decl)
            .collect_vec()
            .join("\n  ")
    }

    /// Declares block and thread indexes.
    fn decl_par_indexes(function: &Function, namer: &mut NameMap) -> String {
        let mut decls = vec![];
        // Load block indexes.
        for (dim, dir) in function.block_dims().iter().zip(&["x", "y", "z"]) {
            let index = namer.name_index(dim.id());
            decls.push(format!("mov.u32 {}, %ctaid.{};", index, dir));
        }
        // Compute thread indexes.
        for (dim, dir) in function.thread_dims().iter().rev().zip(&["x", "y", "z"]) {
            decls.push(format!(
                "mov.s32 {}, %tid.{};",
                namer.name_index(dim.id()),
                dir
            ));
        }
        decls.join("\n  ")
    }

    /// Declares a shared memory block.
    fn shared_mem_decl(&mut self, block: &InternalMemoryRegion, namer: &mut NameMap) {
        let ptr_type_name = Self::get_type(Type::I(32));
        let name = namer.name_addr(block.id());
        unwrap!(writeln!(
            self.buffer,
            ".shared.align 16 .u8 {vec_name}[{size}];\
             \n  mov.{t} {name}, {vec_name};\n",
            vec_name = &name[1..],
            name = name,
            t = ptr_type_name,
            size = unwrap!(block.alloc_size().as_int())
        ));
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
            ref t => panic!("invalid type for the host: {}", t),
        }
    }

    /// Returns the string representation of thread and block sizes on the host.
    fn host_3sizes<'a, IT>(dims: IT) -> [String; 3]
    where
        IT: Iterator<Item = &'a Dimension<'a>> + 'a,
    {
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
        }
    }

    /// Prints a parameter decalartion.
    fn param_decl(&mut self, param: &ParamVal, namer: &NameMap) -> String {
        format!(
            ".param .{t}{attr} {name}",
            t = Self::get_type(param.t()),
            attr = if param.is_pointer() {
                ".ptr.global.align 16"
            } else {
                ""
            },
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
            param_decls = function
                .device_code_args()
                .map(|v| self.param_decl(v, name_map))
                .collect_vec()
                .join(",\n  ");
            // LOAD PARAMETERS
            for val in function.device_code_args() {
                unwrap!(writeln!(
                    self.buffer,
                    "ld.param.{t} {var_name}, [{name}];",
                    t = Self::get_type(val.t()),
                    var_name = name_map.name_param_val(val.key()),
                    name = name_map.name_param(val.key())
                ));
            }
            // INDEX LOAD
            let idx_loads = Self::decl_par_indexes(function, name_map);
            self.buffer.push_str(&idx_loads);
            self.buffer.push_str(&"\n");
            //MEM DECL
            for block in function.mem_blocks() {
                match block.alloc_scheme() {
                    AllocationScheme::Shared => self.shared_mem_decl(block, name_map),
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
                                &[],
                                ir::UnaryOp::Cast(level.t()),
                                Type::I(32),
                                &name,
                                &old_name,
                            );
                        }
                    }
                }
            }
            let ind_levels = function.init_induction_levels().into_iter().chain(
                function
                    .block_dims()
                    .iter()
                    .flat_map(|d| d.induction_levels()),
            );
            for level in ind_levels {
                self.parallel_induction_level(level, name_map);
            }
            self.cfg(function, function.cfg(), name_map);
        }
        let var_decls = self.var_decls(&namer);
        let mut body = String::new();
        body.push_str(&var_decls);
        self.buffer.push_str(&"\n");
        body.push_str(&self.buffer);
        format!(
            include_str!("template/device.ptx"),
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
        let extern_param_names = fun
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
                ParamVal::External(p, _) => format!("&{}", p.name),
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
            }).collect_vec()
            .join(", ");
        let extern_params = fun
            .params
            .iter()
            .map(|p| format!("{} {}", Self::host_type(&p.t), p.name))
            .collect_vec()
            .join(", ");
        let res = write!(
            out,
            include_str!("template/host.c"),
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

impl Printer for CudaPrinter {
    /// Get a proper string representation of an integer in target language
    fn get_int(n: u32) -> String {
        format!("{}", n)
    }

    /// Print result = op1 op op2
    fn print_binop(
        &mut self,
        vector_factors: &[u32],
        op: ir::BinOp,
        operands_type: Type,
        rounding: op::Rounding,
        result: &str,
        lhs: &str,
        rhs: &str,
    ) {
        assert!(vector_factors.is_empty());
        let op = Self::binary_op(op);
        let rounding = Self::rounding(rounding);
        let operands_type = Self::get_type(operands_type);
        unwrap!(writeln!(
            self.buffer,
            "{}{}.{} {}, {}, {};",
            op, rounding, operands_type, result, lhs, rhs
        ));
    }

    /// Prints result = operator operand.
    fn print_unary_op(
        &mut self,
        vector_factors: &[u32],
        operator: ir::UnaryOp,
        operand_type: ir::Type,
        result: &str,
        operand: &str,
    ) {
        assert!(vector_factors.is_empty());
        let operator = match operator {
            ir::UnaryOp::Mov => std::borrow::Cow::from("mov"),
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
                let op = format!("cvt.{}.{}", rounding, Self::get_type(cast_type));
                std::borrow::Cow::from(op)
            }
        };
        let t = Self::get_type(operand_type);
        unwrap!(writeln!(
            self.buffer,
            "{}.{} {}, {};",
            operator, t, result, operand
        ));
    }

    /// Print result = op1 * op2
    fn print_mul(
        &mut self,
        vector_factors: &[u32],
        return_type: Type,
        round: op::Rounding,
        mul_mode: MulMode,
        result: &str,
        lhs: &str,
        rhs: &str,
    ) {
        assert!(vector_factors.is_empty());
        let operator = if round == op::Rounding::Exact {
            format!("mul{}", Self::mul_mode(mul_mode))
        } else {
            format!("mul{}", Self::rounding(round))
        };
        let t = Self::get_type(Self::get_inst_type(mul_mode, return_type));
        unwrap!(writeln!(
            self.buffer,
            "{}.{} {}, {}, {};",
            operator, t, result, lhs, rhs
        ));
    }

    /// Print result = mlhs * mrhs + arhs
    fn print_mad(
        &mut self,
        vector_factors: &[u32],
        return_type: Type,
        round: op::Rounding,
        mul_mode: MulMode,
        result: &str,
        mlhs: &str,
        mrhs: &str,
        arhs: &str,
    ) {
        assert!(vector_factors.is_empty());
        let operator = if round == op::Rounding::Exact {
            format!("mad{}", Self::mul_mode(mul_mode))
        } else {
            format!("fma{}", Self::rounding(round))
        };
        let t = Self::get_type(Self::get_inst_type(mul_mode, return_type));
        unwrap!(writeln!(
            self.buffer,
            "{}.{} {}, {}, {}, {};",
            operator, t, result, mlhs, mrhs, arhs
        ));
    }

    /// Print result = load [addr]
    fn print_ld(
        &mut self,
        vector_factors: &[u32],
        return_type: Type,
        mem_space: MemSpace,
        flag: InstFlag,
        result: &str,
        addr: &str,
    ) {
        let operator = Self::ld_operator(mem_space, flag);
        let vector = match vector_factors {
            [] => "",
            [2] => ".v2",
            [4] => ".v4",
            p => panic!("invalid vector pattern: {:?}", p),
        };
        unwrap!(writeln!(
            self.buffer,
            "{}{}.{} {}, [{}];",
            vector,
            operator,
            Self::get_type(return_type),
            result,
            addr
        ));
    }

    /// Print store val [addr]
    fn print_st(
        &mut self,
        vector_factors: &[u32],
        val_type: Type,
        mem_space: MemSpace,
        mem_flag: InstFlag,
        predicate: Option<&str>,
        addr: &str,
        val: &str,
    ) {
        let vector = match vector_factors {
            [] => "",
            [2] => ".v2",
            [4] => ".v4",
            p => panic!("invalid vector pattern: {:?}", p),
        };
        if let Some(predicate) = predicate {
            unwrap!(write!(self.buffer, "@{} ", predicate));
        }
        let operator = Self::st_operator(mem_space, mem_flag);
        unwrap!(writeln!(
            self.buffer,
            "{}{}.{} [{}], {};",
            vector,
            operator,
            Self::get_type(val_type),
            addr,
            val
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
            "@{} bra.uni LOOP_{};",
            cond, label_id
        ));
    }

    /// Print wait on all threads
    fn print_sync(&mut self) {
        unwrap!(writeln!(self.buffer, "bar.sync 0;"));
    }

    fn name_operand<'a>(
        vector_dims: &[&Dimension],
        op: &ir::Operand,
        namer: &'a NameMap,
    ) -> Cow<'a, str> {
        if vector_dims.is_empty() {
            namer.name_op(op)
        } else {
            let sizes = vector_dims
                .iter()
                .map(|d| unwrap!(d.size().as_int()))
                .collect_vec();
            let names = NDRange::new(&sizes)
                .map(|indexes| {
                    let indexes_map = vector_dims
                        .iter()
                        .zip_eq(indexes)
                        .map(|(d, idx)| (d.id(), idx))
                        .collect_vec();
                    namer.indexed_op_name(op, &indexes_map)
                }).format(", ");
            Cow::Owned(format!("{{{}}}", names))
        }
    }

    fn name_inst<'a>(
        vector_dims: &[&Dimension],
        inst: ir::InstId,
        namer: &'a NameMap,
    ) -> Cow<'a, str> {
        if vector_dims.is_empty() {
            Cow::Borrowed(namer.name_inst(inst))
        } else {
            let sizes = vector_dims
                .iter()
                .map(|d| unwrap!(d.size().as_int()))
                .collect_vec();
            let names = NDRange::new(&sizes)
                .map(|indexes| {
                    let indexes_map = vector_dims
                        .iter()
                        .zip_eq(indexes)
                        .map(|(d, idx)| (d.id(), idx))
                        .collect_vec();
                    namer.indexed_inst_name(inst, &indexes_map)
                }).format(", ");
            Cow::Owned(format!("{{{}}}", names))
        }
    }
}
