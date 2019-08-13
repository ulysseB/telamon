//! Provides functions to print PTX code.
use std::borrow::Cow;
use std::fmt::Write as WriteFmt;
use std::io::Write;

use fxhash::FxHashSet;
use itertools::Itertools;

use telamon::codegen::*;
use telamon::ir::{self, op, Type};
use telamon::search_space::{DimKind, Domain, InstFlag, MemSpace};
use utils::*;

use crate::{Gpu, ValuePrinter};

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

    /// Prints the variables declared by the `ValuePrinter`.
    fn var_decls(&mut self, name_map: &NameMap<ValuePrinter>) -> String {
        let value_printer = name_map.value_printer();
        let print_decl = |(&t, n)| {
            let prefix = ValuePrinter::gen_prefix(t);
            format!(".reg.{} %{}<{}>;", Self::get_type(t), prefix, n)
        };
        value_printer
            .num_var
            .iter()
            .map(print_decl)
            .collect_vec()
            .join("\n  ")
    }

    /// Declares block and thread indexes.
    fn decl_par_indexes(
        function: &Function,
        name_map: &mut NameMap<ValuePrinter>,
    ) -> String {
        let mut decls = vec![];
        // Load block indexes.
        for (dim, dir) in function.block_dims().iter().zip(&["x", "y", "z"]) {
            let index = name_map.name_index(dim.id());
            decls.push(format!("mov.u32 {}, %ctaid.{};", index, dir));
        }
        // Compute thread indexes.
        for (dim, dir) in function.thread_dims().iter().rev().zip(&["x", "y", "z"]) {
            decls.push(format!(
                "mov.s32 {}, %tid.{};",
                name_map.name_index(dim.id()),
                dir
            ));
        }
        decls.join("\n  ")
    }

    /// Declares a shared memory block.
    fn shared_mem_decl(
        &mut self,
        block: &MemoryRegion,
        name_map: &mut NameMap<ValuePrinter>,
    ) {
        let ptr_type_name = Self::get_type(Type::I(32));
        let name = name_map.name_addr(block.id());
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
    fn param_decl(
        &mut self,
        param: &ParamVal,
        name_map: &NameMap<ValuePrinter>,
    ) -> String {
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
        let mut value_printer = ValuePrinter::default();
        let name_map = &mut NameMap::new(function, &mut value_printer);
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
                var_name = name_map.name_param_val(val.key()),
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
        let ind_levels = function.init_induction_levels().iter().chain(
            function
                .block_dims()
                .iter()
                .flat_map(|d| d.induction_levels()),
        );
        for level in ind_levels {
            self.parallel_induction_level(level, name_map);
        }
        self.cfg(function, function.cfg(), name_map);
        let var_decls = self.var_decls(&name_map);
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
        let mut magics = FxHashSet::default();
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
                ParamVal::DivMagic(ref size, t) => {
                    magics.insert((size, t));
                    format!("&_{}_magic{}", t.ident_name(), size.ident_name())
                }
                ParamVal::DivShift(ref size, t) => {
                    magics.insert((size, t));
                    format!("&_{}_shift{}", t.ident_name(), size.ident_name())
                }
            })
            .collect_vec()
            .join(", ");
        for (size, t) in magics {
            let magic = format!("_{}_magic{}", t.ident_name(), size.ident_name());
            let shift = format!("_{}_shift{}", t.ident_name(), size.ident_name());
            extra_def.push(format!("{} {}, {};", Self::host_type(t), magic, shift));
            extra_def.push(format!(
                "{}_magic({}, &{}, &{});",
                Self::host_type(t),
                Self::host_size(size),
                magic,
                shift
            ));
        }
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
    type ValuePrinter = ValuePrinter;

    fn print_int_inst(
        &mut self,
        result: &Nameable<'_>,
        inst: &IntInst<'_>,
        namer: &NameMap<'_, '_, Self::ValuePrinter>,
    ) {
        use IntInst::*;

        let result = &result.name(namer);
        match *inst {
            Move(ref op, t) => self.print_move(t, result, &op.name(namer)),
            Cast(ref op, from_t, to_t) => self.print_unary_op(
                [1, 1],
                ir::UnaryOp::Cast(to_t),
                from_t,
                result,
                &op.name(namer),
            ),
            Add {
                arg_t,
                ref lhs,
                ref rhs,
            } => self.print_add_int(arg_t, result, &lhs.name(namer), &rhs.name(namer)),
            Shr {
                arg_t,
                ref lhs,
                ref rhs,
            } => unwrap!(writeln!(
                self.buffer,
                "  shr.{} {}, {}, {};",
                Self::get_type(arg_t),
                result,
                lhs.name(namer),
                rhs.name(namer),
            )),
            Sub {
                arg_t,
                ref lhs,
                ref rhs,
            } => unwrap!(writeln!(
                self.buffer,
                "  sub.{} {}, {}, {};",
                Self::get_type(arg_t),
                result,
                lhs.name(namer),
                rhs.name(namer)
            )),
            Div {
                arg_t,
                ref lhs,
                ref rhs,
            } => unwrap!(writeln!(
                self.buffer,
                "  div.{} {}, {}, {};",
                Self::get_type(arg_t),
                result,
                lhs.name(namer),
                rhs.name(namer)
            )),
            Min {
                arg_t,
                ref lhs,
                ref rhs,
            } => unwrap!(writeln!(
                self.buffer,
                "  min.{} {}, {}, {};",
                Self::get_type(arg_t),
                result,
                lhs.name(namer),
                rhs.name(namer)
            )),
            Mad {
                arg_t,
                mul_mode,
                ref mlhs,
                ref mrhs,
                ref arhs,
            } => self.print_mad(
                [1, 1],
                mul_mode.output_type(arg_t),
                op::Rounding::Exact,
                mul_mode,
                result,
                &mlhs.name(namer),
                &mrhs.name(namer),
                &arhs.name(namer),
            ),
            Mul {
                arg_t,
                mul_mode,
                ref lhs,
                ref rhs,
            } => self.print_mul(
                [1, 1],
                mul_mode.output_type(arg_t),
                op::Rounding::Exact,
                mul_mode,
                result,
                &lhs.name(namer),
                &rhs.name(namer),
            ),
        }
    }

    /// Print result = op1 op op2
    fn print_binop(
        &mut self,
        vector_factors: [u32; 2],
        op: ir::BinOp,
        operands_type: Type,
        rounding: op::Rounding,
        result: &str,
        lhs: &str,
        rhs: &str,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        let op = Self::binary_op(op);
        let rounding = Self::rounding(rounding);
        let operands_type = Self::get_type(operands_type);
        unwrap!(writeln!(
            self.buffer,
            "  {}{}.{} {}, {}, {};",
            op, rounding, operands_type, result, lhs, rhs
        ));
    }

    /// Prints result = operator operand.
    fn print_unary_op(
        &mut self,
        vector_factors: [u32; 2],
        operator: ir::UnaryOp,
        operand_type: ir::Type,
        result: &str,
        operand: &str,
    ) {
        assert_eq!(vector_factors, [1, 1]);
        let t_str = Self::get_type(operand_type);

        match operator {
            ir::UnaryOp::Mov => unwrap!(writeln!(
                self.buffer,
                "  mov.{} {}, {};",
                t_str, result, operand
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
                            reg = result,
                            operand = operand
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
                    result,
                    operand
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
        result: &str,
        lhs: &str,
        rhs: &str,
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
            operator, t, result, lhs, rhs
        ));
    }

    /// Print result = mlhs * mrhs + arhs
    fn print_mad(
        &mut self,
        vector_factors: [u32; 2],
        return_type: Type,
        round: op::Rounding,
        mul_mode: MulMode,
        result: &str,
        mlhs: &str,
        mrhs: &str,
        arhs: &str,
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
            operator, t, result, mlhs, mrhs, arhs
        ));
    }

    /// Print result = load [addr]
    fn print_ld(
        &mut self,
        vector_factors: [u32; 2],
        return_type: Type,
        mem_space: MemSpace,
        flag: InstFlag,
        result: &str,
        addr: &str,
        predicate: Option<(&str, Cow<'_, str>)>,
    ) {
        let p = if let Some((p, oob)) = predicate {
            match vector_factors {
                [1, 1] => {
                    unwrap!(writeln!(
                        self.buffer,
                        "  mov.{} {}, {};",
                        Self::get_type(return_type),
                        result,
                        oob
                    ));
                }
                // TODO
                _ => (),
            }

            format!("@{}  ", p)
        } else {
            "".to_string()
        };

        let operator = Self::ld_operator(mem_space, flag);
        let vector = match vector_factors {
            [1, 1] => "",
            [1, 2] => ".v2",
            [1, 4] => ".v4",
            p => panic!("invalid vector pattern: {:?}", p),
        };
        unwrap!(writeln!(
            self.buffer,
            "  {}{}{}.{} {}, [{}];",
            p,
            operator,
            vector,
            Self::get_type(return_type),
            result,
            addr
        ));
    }

    /// Print store val [addr]
    fn print_st(
        &mut self,
        vector_factors: [u32; 2],
        val_type: Type,
        mem_space: MemSpace,
        mem_flag: InstFlag,
        predicate: Option<&str>,
        addr: &str,
        val: &str,
    ) {
        let vector = match vector_factors {
            [1, 1] => "",
            [1, 2] => ".v2",
            [1, 4] => ".v4",
            p => panic!("invalid vector pattern: {:?}", p),
        };
        if let Some(predicate) = predicate {
            unwrap!(write!(self.buffer, "@{} ", predicate));
        }
        let operator = Self::st_operator(mem_space, mem_flag);
        unwrap!(writeln!(
            self.buffer,
            "  {}{}.{} [{}], {};",
            operator,
            vector,
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
        // TODO bra.uni
        unwrap!(writeln!(self.buffer, "  @{} bra LOOP_{};", cond, label_id));
    }

    /// Print wait on all threads
    fn print_sync(&mut self) {
        unwrap!(writeln!(self.buffer, "  bar.sync 0;"));
    }

    fn name_operand<'a>(
        vector_levels: &[Vec<Dimension>; 2],
        op: &'a ir::Operand,
        name_map: &'a NameMap<ValuePrinter>,
    ) -> Cow<'a, str> {
        assert!(vector_levels[0].is_empty());
        if vector_levels[1].is_empty() {
            name_map.name_op(op)
        } else {
            let sizes = vector_levels[1]
                .iter()
                .map(|d| unwrap!(d.size().as_int()))
                .collect_vec();
            let names = NDRange::new(&sizes)
                .map(|indexes| {
                    let indexes_map = vector_levels[1]
                        .iter()
                        .zip_eq(indexes)
                        .map(|(d, idx)| (d.id(), idx))
                        .collect_vec();
                    name_map.indexed_op_name(op, &indexes_map)
                })
                .format(", ");
            Cow::Owned(format!("{{{}}}", names))
        }
    }

    fn name_inst<'a>(
        vector_levels: &[Vec<Dimension>; 2],
        inst: ir::InstId,
        name_map: &'a NameMap<ValuePrinter>,
    ) -> Cow<'a, str> {
        assert!(vector_levels[0].is_empty());
        if vector_levels[1].is_empty() {
            Cow::Borrowed(name_map.name_inst(inst))
        } else {
            let sizes = vector_levels[1]
                .iter()
                .map(|d| unwrap!(d.size().as_int()))
                .collect_vec();
            let names = NDRange::new(&sizes)
                .map(|indexes| {
                    let indexes_map = vector_levels[1]
                        .iter()
                        .zip_eq(indexes)
                        .map(|(d, idx)| (d.id(), idx))
                        .collect_vec();
                    name_map.indexed_inst_name(inst, &indexes_map)
                })
                .format(", ");
            Cow::Owned(format!("{{{}}}", names))
        }
    }
}
