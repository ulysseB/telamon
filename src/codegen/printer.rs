use std::convert::TryFrom;

use itertools::Itertools;
use utils::*;

use crate::codegen::llir::IntoVector;
use crate::codegen::*;
use crate::ir::{self, op, Type};
use crate::search_space::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MulMode {
    Wide,
    Low,
    High,
    Empty,
}

impl MulMode {
    fn from_type(from: Type, to: Type) -> Self {
        match (from, to) {
            (Type::F(x), Type::F(y)) if x == y => MulMode::Empty,
            (Type::I(a), Type::I(b)) if b == 2 * a => MulMode::Wide,
            (_, _) => MulMode::Low,
        }
    }
}

// TODO(cleanup): remove this function once values are preprocessed by codegen. If values
// are preprocessed, types will be already lowered.
fn lower_type(t: ir::Type, fun: &Function) -> ir::Type {
    unwrap!(fun
        .space()
        .ir_instance()
        .device()
        .lower_type(t, fun.space()))
}

#[allow(clippy::too_many_arguments)]
pub trait InstPrinter {
    // FIXME: give the input type rather than the return type ?

    /// Print return_id = lhs op rhs
    fn print_binop(
        &mut self,
        vector_factors: [u32; 2],
        op: ir::BinOp,
        operands_type: Type,
        rounding: op::Rounding,
        return_id: llir::VRegister<'_>,
        lhs: llir::VOperand<'_>,
        rhs: llir::VOperand<'_>,
    );

    /// Print return_id = op
    fn print_unary_op(
        &mut self,
        vector_factors: [u32; 2],
        operator: ir::UnaryOp,
        operand_type: Type,
        return_id: llir::VRegister<'_>,
        operand: llir::VOperand<'_>,
    );

    /// Print return_id = op1 * op2
    // FIXME: can we encode it in BinOp
    fn print_mul(
        &mut self,
        vector_factors: [u32; 2],
        return_type: Type,
        round: op::Rounding,
        mul_mode: MulMode,
        return_id: llir::VRegister<'_>,
        op1: llir::VOperand<'_>,
        op2: llir::VOperand<'_>,
    );

    /// Print return_id = mlhs * mrhs + arhs
    fn print_mad(
        &mut self,
        vector_factors: [u32; 2],
        ret_type: Type,
        round: op::Rounding,
        mul_mode: MulMode,
        return_id: llir::VRegister<'_>,
        mlhs: llir::VOperand<'_>,
        mrhs: llir::VOperand<'_>,
        arhs: llir::VOperand<'_>,
    );

    /// Print return_id = load [addr]
    fn print_ld(
        &mut self,
        vector_factors: [u32; 2],
        return_type: Type,
        mem_space: MemSpace,
        flag: InstFlag,
        return_id: llir::VRegister<'_>,
        addr: llir::Operand<'_>,
    );

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
    );

    /// print a label where to jump
    fn print_label(&mut self, label_id: &str);

    /// Print if (cond) jump label(label_id)
    fn print_cond_jump(&mut self, label_id: &str, cond: &str);

    /// Print wait on all threads
    fn print_sync(&mut self);
}

/// Helper struct to provide useful methods wrapping an `InstPrinter` instance.
struct InstPrinterHelper<'a> {
    inst_printer: &'a mut dyn InstPrinter,
}

impl<'a> InstPrinterHelper<'a> {
    fn int<'b, T: ir::IntLiteral<'b>>(&self, lit: T) -> llir::Operand<'b> {
        let (value, bits) = lit.decompose();
        llir::Operand::IntLiteral(value, bits)
    }

    /// Prints a scalar addition on integers.
    fn print_add_int(
        &mut self,
        t: ir::Type,
        result: llir::Register<'_>,
        lhs: llir::Operand<'_>,
        rhs: llir::Operand<'_>,
    ) {
        let rounding = ir::op::Rounding::Exact;
        self.inst_printer.print_binop(
            [1, 1],
            ir::BinOp::Add,
            t,
            rounding,
            result.into_vector(),
            lhs.into_vector(),
            rhs.into_vector(),
        );
    }

    /// Prints a scalar less-than on integers.
    fn print_lt_int(
        &mut self,
        t: ir::Type,
        result: llir::Register<'_>,
        lhs: llir::Operand<'_>,
        rhs: llir::Operand<'_>,
    ) {
        let rounding = ir::op::Rounding::Exact;
        self.inst_printer.print_binop(
            [1, 1],
            ir::BinOp::Lt,
            t,
            rounding,
            result.into_vector(),
            lhs.into_vector(),
            rhs.into_vector(),
        );
    }

    /// Prints an AND operation.
    fn print_and(
        &mut self,
        t: ir::Type,
        result: llir::Register<'_>,
        lhs: llir::Operand<'_>,
        rhs: llir::Operand<'_>,
    ) {
        let rounding = ir::op::Rounding::Exact;
        self.inst_printer.print_binop(
            [1, 1],
            ir::BinOp::And,
            t,
            rounding,
            result.into_vector(),
            lhs.into_vector(),
            rhs.into_vector(),
        );
    }

    /// Prints a move instruction.
    fn print_move(
        &mut self,
        t: ir::Type,
        result: llir::Register<'_>,
        operand: llir::Operand<'_>,
    ) {
        self.inst_printer.print_unary_op(
            [1, 1],
            ir::UnaryOp::Mov,
            t,
            result.into_vector(),
            operand.into_vector(),
        );
    }

    /// Prints a scalar equals instruction.
    fn print_equals(
        &mut self,
        t: ir::Type,
        result: llir::Register<'_>,
        lhs: llir::Operand<'_>,
        rhs: llir::Operand<'_>,
    ) {
        let rounding = ir::op::Rounding::Exact;
        self.inst_printer.print_binop(
            [1, 1],
            ir::BinOp::Equals,
            t,
            rounding,
            result.into_vector(),
            lhs.into_vector(),
            rhs.into_vector(),
        );
    }
}

/// High-level printer struct delegating to an `InstPrinter` instance the role of printing actual
/// instructions.
///
/// The printer's task is to lower high(er) level construct into instructions, which get passed to
/// the underlying `InstPrinter`.
pub struct Printer<'a, 'b> {
    helper: InstPrinterHelper<'a>,
    namer: &'a mut NameMap<'b>,
}

impl<'a, 'b> Printer<'a, 'b> {
    pub fn new(
        inst_printer: &'a mut dyn InstPrinter,
        namer: &'a mut NameMap<'b>,
    ) -> Self {
        Printer {
            helper: InstPrinterHelper { inst_printer },
            namer,
        }
    }

    /// Change the side-effect guards so that the specified threads are disabled.
    fn disable_threads<'d, I>(&mut self, threads: I)
    where
        I: Iterator<Item = &'d Dimension<'d>>,
    {
        let mut guard: Option<llir::Register<'_>> = None;
        for dim in threads {
            let new_guard = self.namer.gen_name(ir::Type::I(1));
            let index = self.namer.name_index(dim.id());
            let zero = self.helper.int(0i32);
            self.helper
                .print_equals(ir::Type::I(32), new_guard, index.into(), zero);
            if let Some(guard) = guard {
                self.helper.print_and(
                    ir::Type::I(1),
                    guard,
                    guard.into(),
                    new_guard.into(),
                );
            } else {
                guard = Some(new_guard);
            };
        }
        self.namer.set_side_effect_guard(guard);
    }

    pub fn privatise_global_block(&mut self, block: &MemoryRegion, fun: &Function) {
        if fun.block_dims().is_empty() {
            return;
        }
        let addr = self.namer.name_addr(block.id());
        let addr_type = lower_type(Type::PtrTo(block.id()), fun);
        let size = self.namer.name_size(block.local_size(), Type::I(32));
        let d0 = self
            .namer
            .name_index(fun.block_dims()[0].id())
            .into_operand();
        let var = fun.block_dims()[1..].iter().fold(d0, |old_var, dim| {
            let var = self.namer.gen_name(Type::I(32));
            let size = self.namer.name_size(dim.size(), Type::I(32));
            let idx = self.namer.name_index(dim.id()).into_operand();
            self.helper.inst_printer.print_mad(
                [1, 1],
                Type::I(32),
                op::Rounding::Exact,
                MulMode::Low,
                var.into_vector(),
                old_var.into_vector(),
                size.into_vector(),
                idx.into_vector(),
            );
            var.into_operand()
        });
        let mode = MulMode::from_type(Type::I(32), addr_type);
        self.helper.inst_printer.print_mad(
            [1, 1],
            addr_type,
            op::Rounding::Exact,
            mode,
            addr.into_vector(),
            var.into_vector(),
            size.into_vector(),
            addr.into_operand().into_vector(),
        );
    }

    /// Prints a classic loop - that is, a sequential loop with an index and a jump to the beginning
    /// at the end of the block
    fn standard_loop(
        &mut self,
        fun: &Function,
        dim: &Dimension<'b>,
        cfgs: &'b [Cfg<'b>],
    ) {
        let idx = self.namer.name_index(dim.id());
        let zero = self.helper.int(0i32);
        self.helper.print_move(Type::I(32), idx, zero);
        let mut ind_var_vec = vec![];
        let loop_id = self.namer.gen_loop_id();
        let ind_levels = dim.induction_levels();
        for level in ind_levels.iter() {
            let dim_id = level.increment.as_ref().map(|&(dim, _)| dim);
            let ind_var = self
                .namer
                .name_induction_var(level.ind_var, dim_id)
                .to_register()
                .unwrap();
            let base_components = level.base.components().map({
                let namer = &self.namer;
                move |v| namer.name(v)
            });
            match base_components.collect_vec()[..] {
                [ref base] => self.helper.print_move(level.t(), ind_var, base.clone()),
                [ref lhs, ref rhs] => self.helper.print_add_int(
                    level.t(),
                    ind_var,
                    lhs.clone(),
                    rhs.clone(),
                ),
                _ => panic!(),
            };
            ind_var_vec.push(ind_var);
        }
        self.helper.inst_printer.print_label(&loop_id.to_string());
        self.cfg_vec(fun, cfgs);
        for (level, ind_var) in ind_levels.iter().zip_eq(ind_var_vec) {
            if let Some((_, ref increment)) = level.increment {
                let step = self.namer.name_size(increment, level.t());
                self.helper
                    .print_add_int(level.t(), ind_var, ind_var.into(), step);
            };
        }
        let one = self.helper.int(1i32);
        self.helper
            .print_add_int(ir::Type::I(32), idx, idx.into(), one);
        let lt_cond = self.namer.gen_name(ir::Type::I(1));
        let size = self.namer.name_size(dim.size(), Type::I(32));
        self.helper
            .print_lt_int(ir::Type::I(32), lt_cond, idx.into(), size);
        self.helper
            .inst_printer
            .print_cond_jump(&loop_id.to_string(), lt_cond.name());
    }

    /// Prints an unroll loop - loop without jumps
    fn unroll_loop(&mut self, fun: &Function, dim: &Dimension<'b>, cfgs: &'b [Cfg<'b>]) {
        let mut incr_levels = Vec::new();
        for level in dim.induction_levels() {
            let dim_id = level.increment.as_ref().map(|&(dim, _)| dim);
            let ind_var = self
                .namer
                .name_induction_var(level.ind_var, dim_id)
                .to_register()
                .unwrap();
            let base_components = level
                .base
                .components()
                .map(|v| self.namer.name(v))
                .collect_vec();
            let base = match base_components[..] {
                [ref base] => base.clone(),
                [ref lhs, ref rhs] => {
                    let tmp = self.namer.gen_name(level.t());
                    self.helper
                        .print_add_int(level.t(), tmp, lhs.clone(), rhs.clone());
                    tmp.into()
                }
                _ => panic!(),
            };
            if let Some((_, ref incr)) = level.increment {
                incr_levels.push((level, ind_var, incr, base.clone()));
            }
            self.helper.print_move(level.t(), ind_var, base);
        }
        for i in 0..unwrap!(dim.size().as_int()) {
            self.namer.set_current_index(dim, i);
            if i > 0 {
                for &(level, ind_var, ref incr, ref base) in &incr_levels {
                    if let Some(step) = incr.as_int() {
                        let stepxi = self.helper.int(i32::try_from(step * i).unwrap());
                        self.helper.print_add_int(
                            level.t(),
                            ind_var,
                            stepxi,
                            base.clone(),
                        );
                    } else {
                        let step = self.namer.name_size(incr, level.t());
                        self.helper.print_add_int(
                            level.t(),
                            ind_var,
                            step,
                            ind_var.into(),
                        );
                    };
                }
            }
            self.cfg_vec(fun, cfgs);
        }
        self.namer.unset_current_index(dim);
    }

    /// Prints a multiplicative induction var level.
    pub fn parallel_induction_level(&mut self, level: &InductionLevel<'b>) {
        let dim_id = level.increment.as_ref().map(|&(dim, _)| dim);
        let ind_var = self
            .namer
            .name_induction_var(level.ind_var, dim_id)
            .to_register()
            .unwrap();
        let base_components = level
            .base
            .components()
            .map({
                let namer = &self.namer;
                move |v| namer.name(v)
            })
            .collect_vec();
        if let Some((dim, ref increment)) = level.increment {
            let index = self.namer.name_index(dim).into_operand();
            let step = self.namer.name_size(increment, Type::I(32));
            let mode = MulMode::from_type(Type::I(32), level.t());
            match base_components[..] {
                [] => {
                    self.helper.inst_printer.print_mul(
                        [1, 1],
                        level.t(),
                        op::Rounding::Exact,
                        mode,
                        ind_var.into_vector(),
                        index.into_vector(),
                        step.into_vector(),
                    );
                }
                [ref base] => {
                    self.helper.inst_printer.print_mad(
                        [1, 1],
                        level.t(),
                        op::Rounding::Exact,
                        mode,
                        ind_var.into_vector(),
                        index.into_vector(),
                        step.into_vector(),
                        base.clone().into_vector(),
                    );
                }
                _ => panic!(),
            }
        } else {
            match base_components[..] {
                [] => {
                    let zero = self.helper.int(0i32);
                    self.helper.print_move(level.t(), ind_var, zero);
                }
                [ref base] => {
                    self.helper.print_move(level.t(), ind_var, base.clone());
                }
                [ref lhs, ref rhs] => {
                    self.helper.print_add_int(
                        level.t(),
                        ind_var,
                        lhs.clone(),
                        rhs.clone(),
                    );
                }
                _ => panic!(),
            }
        }
    }

    /// Prints a Loop
    fn gen_loop(&mut self, fun: &Function, dim: &Dimension<'b>, cfgs: &'b [Cfg<'b>]) {
        match dim.kind() {
            DimKind::LOOP => self.standard_loop(fun, dim, cfgs),
            DimKind::UNROLL => self.unroll_loop(fun, dim, cfgs),
            _ => panic!("{:?} loop should not be printed here !", dim.kind()),
        }
    }

    fn cfg_vec(&mut self, fun: &Function, cfgs: &'b [Cfg<'b>]) {
        for c in cfgs.iter() {
            self.cfg(fun, c);
        }
    }

    /// Prints a cfg.
    pub fn cfg(&mut self, fun: &Function, c: &'b Cfg<'b>) {
        match c {
            Cfg::Root(cfgs) => self.cfg_vec(fun, cfgs),
            Cfg::Loop(dim, cfgs) => self.gen_loop(fun, dim, cfgs),
            Cfg::Threads(dims, ind_levels, inner) => {
                // Disable inactive threads
                self.disable_threads(
                    dims.iter().zip_eq(fun.thread_dims().iter()).filter_map(
                        |(&active_dim_id, dim)| {
                            if active_dim_id.is_none() {
                                Some(dim)
                            } else {
                                None
                            }
                        },
                    ),
                );
                for level in ind_levels {
                    self.parallel_induction_level(level);
                }
                self.cfg_vec(fun, inner);
                self.helper.inst_printer.print_sync();
            }
            Cfg::Instruction(vec_dims, inst) => self.inst(vec_dims, inst, fun),
        }
    }

    /// Prints an instruction.
    fn inst(
        &mut self,
        vector_levels: &[Vec<Dimension>; 2],
        inst: &'b Instruction<'b>,
        fun: &Function,
    ) {
        // Multiple dimension can be mapped to the same vectorization level so we combine
        // them when computing the vectorization factor.
        let vector_factors = [
            vector_levels[0]
                .iter()
                .map(|d| unwrap!(d.size().as_int()))
                .product(),
            vector_levels[1]
                .iter()
                .map(|d| unwrap!(d.size().as_int()))
                .product(),
        ];
        let helper = &mut self.helper;
        match inst.operator() {
            op::BinOp(op, lhs, rhs, round) => {
                let t = lower_type(lhs.t(), fun);
                let inst = self.namer.vector_inst(vector_levels, inst.id());
                let lhs = self.namer.vector_operand(vector_levels, lhs);
                let rhs = self.namer.vector_operand(vector_levels, rhs);
                helper.inst_printer.print_binop(
                    vector_factors,
                    *op,
                    t,
                    *round,
                    inst,
                    lhs,
                    rhs,
                )
            }
            op::Mul(lhs, rhs, round, return_type) => {
                let low_lhs_type = lower_type(lhs.t(), fun);
                let low_ret_type = lower_type(*return_type, fun);
                let mode = MulMode::from_type(low_lhs_type, low_ret_type);
                let inst = self.namer.vector_inst(vector_levels, inst.id());
                let lhs = self.namer.vector_operand(vector_levels, lhs);
                let rhs = self.namer.vector_operand(vector_levels, rhs);
                helper.inst_printer.print_mul(
                    vector_factors,
                    low_ret_type,
                    *round,
                    mode,
                    inst,
                    lhs,
                    rhs,
                )
            }
            op::Mad(mul_lhs, mul_rhs, add_rhs, round) => {
                let low_mlhs_type = lower_type(mul_lhs.t(), fun);
                let low_arhs_type = lower_type(add_rhs.t(), fun);
                let mode = MulMode::from_type(low_mlhs_type, low_arhs_type);
                let t = unwrap!(inst.t());
                let inst = self.namer.vector_inst(vector_levels, inst.id());
                let mul_lhs = self.namer.vector_operand(vector_levels, mul_lhs);
                let mul_rhs = self.namer.vector_operand(vector_levels, mul_rhs);
                let add_rhs = self.namer.vector_operand(vector_levels, add_rhs);
                helper.inst_printer.print_mad(
                    vector_factors,
                    t,
                    *round,
                    mode,
                    inst,
                    mul_lhs,
                    mul_rhs,
                    add_rhs,
                )
            }
            op::UnaryOp(operator, operand) => {
                let operator = match *operator {
                    ir::UnaryOp::Cast(t) => ir::UnaryOp::Cast(lower_type(t, fun)),
                    op => op,
                };
                let t = lower_type(operand.t(), fun);
                let name = self.namer.vector_inst(vector_levels, inst.id());
                let operand = self.namer.vector_operand(vector_levels, operand);
                helper.inst_printer.print_unary_op(
                    vector_factors,
                    operator,
                    t,
                    name,
                    operand,
                )
            }
            op::Ld(ld_type, addr, pattern) => {
                let mem_flag = unwrap!(inst.mem_flag());
                let inst = self.namer.vector_inst(vector_levels, inst.id());
                let addr = self.namer.name_op(addr);
                helper.inst_printer.print_ld(
                    vector_factors,
                    lower_type(*ld_type, fun),
                    access_pattern_space(pattern, fun.space()),
                    mem_flag,
                    inst,
                    addr,
                )
            }
            op::St(addr, val, _, pattern) => {
                let guard = if inst.has_side_effects() {
                    self.namer.side_effect_guard()
                } else {
                    None
                };
                let t = lower_type(val.t(), fun);
                let addr = self.namer.name_op(addr);
                let val = self.namer.vector_operand(vector_levels, val);
                helper.inst_printer.print_st(
                    vector_factors,
                    t,
                    access_pattern_space(pattern, fun.space()),
                    unwrap!(inst.mem_flag()),
                    guard,
                    addr,
                    val,
                );
            }
            op @ op::TmpLd(..) | op @ op::TmpSt(..) => {
                panic!("non-printable instruction {:?}", op)
            }
        }
    }
}
