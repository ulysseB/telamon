use crate::codegen::*;
use crate::search_space::*;
use itertools::Itertools;
use std::borrow::Cow;
use utils::*;

use crate::ir::{self, op, Type};
use crate::search_space::DimKind;

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

#[allow(clippy::too_many_arguments)]
pub trait InstPrinter {
    type VP: ValuePrinter;

    // FIXME: give the input type rather than the return type ?

    /// Print return_id = lhs op rhs
    fn print_binop(
        &mut self,
        vector_factors: [u32; 2],
        op: ir::BinOp,
        operands_type: Type,
        rounding: op::Rounding,
        return_id: &str,
        lhs: &str,
        rhs: &str,
    );

    /// Print return_id = op
    fn print_unary_op(
        &mut self,
        vector_factors: [u32; 2],
        operator: ir::UnaryOp,
        operand_type: Type,
        return_id: &str,
        operand: &str,
    );

    /// Print return_id = op1 * op2
    // FIXME: can we encode it in BinOp
    fn print_mul(
        &mut self,
        vector_factors: [u32; 2],
        return_type: Type,
        round: op::Rounding,
        mul_mode: MulMode,
        return_id: &str,
        op1: &str,
        op2: &str,
    );

    /// Print return_id = mlhs * mrhs + arhs
    fn print_mad(
        &mut self,
        vector_factors: [u32; 2],
        ret_type: Type,
        round: op::Rounding,
        mul_mode: MulMode,
        return_id: &str,
        mlhs: &str,
        mrhs: &str,
        arhs: &str,
    );

    /// Print return_id = load [addr]
    fn print_ld(
        &mut self,
        vector_factors: [u32; 2],
        return_type: Type,
        mem_space: MemSpace,
        flag: InstFlag,
        return_id: &str,
        addr: &str,
    );

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
    );

    /// print a label where to jump
    fn print_label(&mut self, label_id: &str);

    /// Print if (cond) jump label(label_id)
    fn print_cond_jump(&mut self, label_id: &str, cond: &str);

    /// Print wait on all threads
    fn print_sync(&mut self);

    /// Name an operand, vectorized on the given dimensions.
    fn name_operand<'a>(
        vector_levels: &[Vec<Dimension>; 2],
        op: &ir::Operand,
        namer: &'a NameMap<Self::VP>,
    ) -> Cow<'a, str>;

    /// Names an instruction, vectorized on the given dimensions.
    fn name_inst<'a>(
        vector_levels: &[Vec<Dimension>; 2],
        inst: ir::InstId,
        namer: &'a NameMap<Self::VP>,
    ) -> Cow<'a, str>;

    /// Prints a scalar less-than on integers.
    fn print_lt_int(&mut self, t: ir::Type, result: &str, lhs: &str, rhs: &str) {
        let rounding = ir::op::Rounding::Exact;
        self.print_binop([1, 1], ir::BinOp::Lt, t, rounding, result, lhs, rhs);
    }

    /// Prints a scalar equals instruction.
    fn print_equals(&mut self, t: ir::Type, result: &str, lhs: &str, rhs: &str) {
        let rounding = ir::op::Rounding::Exact;
        self.print_binop([1, 1], ir::BinOp::Equals, t, rounding, result, lhs, rhs);
    }

    /// Prints a scalar addition on integers.
    fn print_add_int(&mut self, t: ir::Type, result: &str, lhs: &str, rhs: &str) {
        let rounding = ir::op::Rounding::Exact;
        self.print_binop([1, 1], ir::BinOp::Add, t, rounding, result, lhs, rhs);
    }

    /// Prints an AND operation.
    fn print_and(&mut self, t: ir::Type, result: &str, lhs: &str, rhs: &str) {
        let rounding = ir::op::Rounding::Exact;
        self.print_binop([1, 1], ir::BinOp::And, t, rounding, result, lhs, rhs);
    }

    /// Prints a move instruction.
    fn print_move(&mut self, t: ir::Type, result: &str, operand: &str) {
        self.print_unary_op([1, 1], ir::UnaryOp::Mov, t, result, operand);
    }

    fn cfg_vec(&mut self, fun: &Function, cfgs: &[Cfg], namer: &mut NameMap<Self::VP>) {
        for c in cfgs.iter() {
            self.cfg(fun, c, namer);
        }
    }

    /// Prints a cfg.
    fn cfg<'a>(&mut self, fun: &Function, c: &Cfg<'a>, namer: &mut NameMap<Self::VP>) {
        match c {
            Cfg::Root(cfgs) => self.cfg_vec(fun, cfgs, namer),
            Cfg::Loop(dim, cfgs) => self.gen_loop(fun, dim, cfgs, namer),
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
                    namer,
                );
                for level in ind_levels {
                    self.parallel_induction_level(level, namer);
                }
                self.cfg_vec(fun, inner, namer);
                self.print_sync();
            }
            Cfg::Instruction(vec_dims, inst) => self.inst(vec_dims, inst, namer, fun),
        }
    }

    /// Prints a multiplicative induction var level.
    fn parallel_induction_level(
        &mut self,
        level: &InductionLevel,
        namer: &NameMap<Self::VP>,
    ) {
        let dim_id = level.increment.as_ref().map(|&(dim, _)| dim);
        let ind_var = namer.name_induction_var(level.ind_var, dim_id);
        let base_components =
            level.base.components().map(|v| namer.name(v)).collect_vec();
        if let Some((dim, ref increment)) = level.increment {
            let index = namer.name_index(dim);
            let step = namer.name_size(increment, Type::I(32));
            let mode = MulMode::from_type(Type::I(32), level.t());
            match base_components[..] {
                [] => self.print_mul(
                    [1, 1],
                    level.t(),
                    op::Rounding::Exact,
                    mode,
                    &ind_var,
                    &index,
                    &step,
                ),
                [ref base] => self.print_mad(
                    [1, 1],
                    level.t(),
                    op::Rounding::Exact,
                    mode,
                    &ind_var,
                    &index,
                    &step,
                    &base,
                ),
                _ => panic!(),
            }
        } else {
            match base_components[..] {
                [] => {
                    let zero = Self::VP::get_const_int(&0.into(), 32);
                    self.print_move(level.t(), &ind_var, &zero);
                }
                [ref base] => self.print_move(level.t(), &ind_var, &base),
                [ref lhs, ref rhs] => self.print_add_int(level.t(), &ind_var, &lhs, &rhs),
                _ => panic!(),
            }
        }
    }

    /// Change the side-effect guards so that the specified threads are disabled.
    fn disable_threads<'a, I>(&mut self, threads: I, namer: &mut NameMap<Self::VP>)
    where
        I: Iterator<Item = &'a Dimension<'a>>,
    {
        let mut guard: Option<String> = None;
        for dim in threads {
            let new_guard = namer.gen_name(ir::Type::I(1));
            let index = namer.name_index(dim.id());
            self.print_equals(
                ir::Type::I(32),
                &new_guard,
                index,
                &Self::VP::get_const_int(&0.into(), 32),
            );
            if let Some(ref guard) = guard {
                self.print_and(ir::Type::I(1), guard, guard, &new_guard);
            } else {
                guard = Some(new_guard);
            };
        }
        namer.set_side_effect_guard(guard.map(RcStr::new));
    }

    /// Prints a Loop
    fn gen_loop(
        &mut self,
        fun: &Function,
        dim: &Dimension,
        cfgs: &[Cfg],
        namer: &mut NameMap<Self::VP>,
    ) {
        match dim.kind() {
            DimKind::LOOP => self.standard_loop(fun, dim, cfgs, namer),
            DimKind::UNROLL => self.unroll_loop(fun, dim, cfgs, namer),
            _ => panic!("{:?} loop should not be printed here !", dim.kind()),
        }
    }

    /// Prints a classic loop - that is, a sequential loop with an index and a jump to the beginning
    /// at the end of the block
    fn standard_loop(
        &mut self,
        fun: &Function,
        dim: &Dimension,
        cfgs: &[Cfg],
        namer: &mut NameMap<Self::VP>,
    ) {
        let idx = namer.name_index(dim.id()).to_string();
        let zero = Self::VP::get_const_int(&0.into(), 32);
        self.print_move(Type::I(32), &idx, &zero);
        let mut ind_var_vec = vec![];
        let loop_id = namer.gen_loop_id();
        let ind_levels = dim.induction_levels();
        for level in ind_levels.iter() {
            let dim_id = level.increment.as_ref().map(|&(dim, _)| dim);
            let ind_var = namer.name_induction_var(level.ind_var, dim_id);
            let base_components = level.base.components().map(|v| namer.name(v));
            match base_components.collect_vec()[..] {
                [ref base] => self.print_move(level.t(), &ind_var, &base),
                [ref lhs, ref rhs] => self.print_add_int(level.t(), &ind_var, lhs, rhs),
                _ => panic!(),
            };
            ind_var_vec.push(ind_var.into_owned());
        }
        self.print_label(&loop_id.to_string());
        self.cfg_vec(fun, cfgs, namer);
        for (level, ind_var) in ind_levels.iter().zip_eq(ind_var_vec) {
            if let Some((_, ref increment)) = level.increment {
                let step = namer.name_size(increment, level.t());
                self.print_add_int(level.t(), &ind_var, &ind_var, &step);
            };
        }
        let one = Self::VP::get_const_int(&1.into(), 32);
        self.print_add_int(ir::Type::I(32), &idx, &idx, &one);
        let lt_cond = namer.gen_name(ir::Type::I(1));
        let size = namer.name_size(dim.size(), Type::I(32));
        self.print_lt_int(ir::Type::I(32), &lt_cond, &idx, &size);
        self.print_cond_jump(&loop_id.to_string(), &lt_cond);
    }

    /// Prints an unroll loop - loop without jumps
    fn unroll_loop(
        &mut self,
        fun: &Function,
        dim: &Dimension,
        cfgs: &[Cfg],
        namer: &mut NameMap<Self::VP>,
    ) {
        let mut incr_levels = Vec::new();
        for level in dim.induction_levels() {
            let dim_id = level.increment.as_ref().map(|&(dim, _)| dim);
            let ind_var = namer.name_induction_var(level.ind_var, dim_id).to_string();
            let base_components = level
                .base
                .components()
                .map(|v| namer.name(v).into_owned())
                .collect_vec();
            let base = match base_components[..] {
                [ref base] => base.clone(),
                [ref lhs, ref rhs] => {
                    let tmp = namer.gen_name(level.t());
                    self.print_add_int(level.t(), &tmp, lhs, rhs);
                    tmp
                }
                _ => panic!(),
            };
            self.print_unary_op([1, 1], ir::UnaryOp::Mov, level.t(), &ind_var, &base);
            if let Some((_, ref incr)) = level.increment {
                incr_levels.push((level, ind_var, incr, base));
            }
        }
        for i in 0..unwrap!(dim.size().as_int()) {
            namer.set_current_index(dim, i);
            if i > 0 {
                for &(level, ref ind_var, ref incr, ref base) in &incr_levels {
                    if let Some(step) = incr.as_int() {
                        let stepxi = Self::VP::get_const_int(&(step * i).into(), 32);
                        self.print_add_int(level.t(), ind_var, &stepxi, base);
                    } else {
                        let step = namer.name_size(incr, level.t());
                        self.print_add_int(level.t(), ind_var, &step, ind_var);
                    };
                }
            }
            self.cfg_vec(fun, cfgs, namer);
        }
        namer.unset_current_index(dim);
    }

    fn privatise_global_block(
        &mut self,
        block: &MemoryRegion,
        namer: &mut NameMap<Self::VP>,
        fun: &Function,
    ) {
        if fun.block_dims().is_empty() {
            return;
        }
        let addr = namer.name_addr(block.id()).into_owned();
        let addr_type = Self::lower_type(Type::PtrTo(block.id()), fun);
        let size = namer
            .name_size(block.local_size(), Type::I(32))
            .into_owned();
        let d0 = namer.name_index(fun.block_dims()[0].id()).to_string();
        let var = fun.block_dims()[1..].iter().fold(d0, |old_var, dim| {
            let var = namer.gen_name(Type::I(32));
            let size = namer.name_size(dim.size(), Type::I(32));
            let idx = namer.name_index(dim.id()).to_string();
            self.print_mad(
                [1, 1],
                Type::I(32),
                op::Rounding::Exact,
                MulMode::Low,
                &var,
                &old_var,
                &size,
                &idx,
            );
            var
        });
        let mode = MulMode::from_type(Type::I(32), addr_type);
        self.print_mad(
            [1, 1],
            addr_type,
            op::Rounding::Exact,
            mode,
            &addr,
            &var,
            &size,
            &addr,
        );
    }

    /// Prints an instruction.
    fn inst(
        &mut self,
        vector_levels: &[Vec<Dimension>; 2],
        inst: &Instruction,
        namer: &mut NameMap<Self::VP>,
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
        match inst.operator() {
            op::BinOp(op, lhs, rhs, round) => {
                let t = Self::lower_type(lhs.t(), fun);
                self.print_binop(
                    vector_factors,
                    *op,
                    t,
                    *round,
                    &Self::name_inst(vector_levels, inst.id(), namer),
                    &Self::name_operand(vector_levels, lhs, namer),
                    &Self::name_operand(vector_levels, rhs, namer),
                )
            }
            op::Mul(lhs, rhs, round, return_type) => {
                let low_lhs_type = Self::lower_type(lhs.t(), fun);
                let low_ret_type = Self::lower_type(*return_type, fun);
                let mode = MulMode::from_type(low_lhs_type, low_ret_type);
                self.print_mul(
                    vector_factors,
                    low_ret_type,
                    *round,
                    mode,
                    &Self::name_inst(vector_levels, inst.id(), namer),
                    &Self::name_operand(vector_levels, lhs, namer),
                    &Self::name_operand(vector_levels, rhs, namer),
                )
            }
            op::Mad(mul_lhs, mul_rhs, add_rhs, round) => {
                let low_mlhs_type = Self::lower_type(mul_lhs.t(), fun);
                let low_arhs_type = Self::lower_type(add_rhs.t(), fun);
                let mode = MulMode::from_type(low_mlhs_type, low_arhs_type);
                self.print_mad(
                    vector_factors,
                    unwrap!(inst.t()),
                    *round,
                    mode,
                    &Self::name_inst(vector_levels, inst.id(), namer),
                    &Self::name_operand(vector_levels, mul_lhs, namer),
                    &Self::name_operand(vector_levels, mul_rhs, namer),
                    &Self::name_operand(vector_levels, add_rhs, namer),
                )
            }
            op::UnaryOp(operator, operand) => {
                let operator = match *operator {
                    ir::UnaryOp::Cast(t) => ir::UnaryOp::Cast(Self::lower_type(t, fun)),
                    op => op,
                };
                let t = Self::lower_type(operand.t(), fun);
                let name = Self::name_inst(vector_levels, inst.id(), namer);
                let operand = Self::name_operand(vector_levels, operand, namer);
                self.print_unary_op(vector_factors, operator, t, &name, &operand)
            }
            op::Ld(ld_type, addr, pattern) => self.print_ld(
                vector_factors,
                Self::lower_type(*ld_type, fun),
                access_pattern_space(pattern, fun.space()),
                unwrap!(inst.mem_flag()),
                &Self::name_inst(vector_levels, inst.id(), namer),
                &Self::name_operand(&[vec![], vec![]], addr, namer),
            ),
            op::St(addr, val, _, pattern) => {
                let guard = if inst.has_side_effects() {
                    namer.side_effect_guard()
                } else {
                    None
                };
                self.print_st(
                    vector_factors,
                    Self::lower_type(val.t(), fun),
                    access_pattern_space(pattern, fun.space()),
                    unwrap!(inst.mem_flag()),
                    guard.as_ref().map(|x| x as _),
                    &Self::name_operand(&[vec![], vec![]], addr, namer),
                    &Self::name_operand(vector_levels, val, namer),
                );
            }
            op @ op::TmpLd(..) | op @ op::TmpSt(..) => {
                panic!("non-printable instruction {:?}", op)
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

    fn mul_mode(from: Type, to: Type) -> MulMode {
        match (from, to) {
            (Type::I(32), Type::I(64)) => MulMode::Wide,
            (ref x, ref y) if x == y => MulMode::Low,
            _ => panic!(),
        }
    }
}
