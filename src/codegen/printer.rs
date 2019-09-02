use std::convert::{TryFrom, TryInto};
use std::fmt;

use itertools::Itertools;

use crate::codegen::llir::IntLiteral as _;
use crate::codegen::*;
use crate::ir::{self, op, Type};
use crate::search_space::*;

pub trait IdentDisplay {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result;

    fn ident(&self) -> DisplayIdent<'_, Self> {
        DisplayIdent { inner: self }
    }
}

pub struct DisplayIdent<'a, T: ?Sized> {
    inner: &'a T,
}

impl<T: fmt::Debug + ?Sized> fmt::Debug for DisplayIdent<'_, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.inner, fmt)
    }
}

impl<T: IdentDisplay + ?Sized> fmt::Display for DisplayIdent<'_, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        IdentDisplay::fmt(self.inner, fmt)
    }
}

impl IdentDisplay for Size {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.factor() != 1 {
            write!(fmt, "_{}", self.factor())?;
        }

        for dividend in self.dividend() {
            write!(fmt, "_{}", dividend)?;
        }

        if self.divisor() != 1 {
            write!(fmt, "_{}", self.divisor())?;
        }

        Ok(())
    }
}

impl IdentDisplay for Type {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::I(s) => write!(fmt, "i{}", s),
            Type::F(s) => write!(fmt, "f{}", s),
            Type::PtrTo(mem) => write!(fmt, "memptr{}", mem.0),
        }
    }
}

// TODO(cleanup): remove this function once values are preprocessed by codegen. If values
// are preprocessed, types will be already lowered.
fn lower_type(t: ir::Type, fun: &Function) -> ir::Type {
    fun.space()
        .ir_instance()
        .device()
        .lower_type(t, fun.space())
        .unwrap()
}

pub trait InstPrinter {
    /// print a label where to jump
    fn print_label(&mut self, label: llir::Label<'_>);

    fn print_inst(&mut self, inst: llir::PredicatedInstruction<'_>);
}

/// Helper struct to provide useful methods wrapping an `InstPrinter` instance.
struct InstPrinterHelper<'a> {
    inst_printer: &'a mut dyn InstPrinter,
}

impl<'a> InstPrinterHelper<'a> {
    /// Prints a scalar addition on integers.
    fn print_add_int(
        &mut self,
        result: llir::Register<'_>,
        lhs: llir::Operand<'_>,
        rhs: llir::Operand<'_>,
    ) {
        self.inst_printer
            .print_inst(llir::Instruction::iadd(result, lhs, rhs).unwrap().into())
    }

    /// Prints a scalar less-than on integers.
    fn print_lt_int(
        &mut self,
        result: llir::Register<'_>,
        lhs: llir::Operand<'_>,
        rhs: llir::Operand<'_>,
    ) {
        self.inst_printer
            .print_inst(llir::Instruction::set_lt(result, lhs, rhs).unwrap().into())
    }

    /// Prints an AND operation.
    fn print_and(
        &mut self,
        result: llir::Register<'_>,
        lhs: llir::Operand<'_>,
        rhs: llir::Operand<'_>,
    ) {
        self.inst_printer
            .print_inst(llir::Instruction::and(result, lhs, rhs).unwrap().into())
    }

    /// Prints a move instruction.
    fn print_move(&mut self, result: llir::Register<'_>, operand: llir::Operand<'_>) {
        self.inst_printer
            .print_inst(llir::Instruction::mov(result, operand).unwrap().into())
    }

    /// Prints a scalar equals instruction.
    fn print_equals(
        &mut self,
        result: llir::Register<'_>,
        lhs: llir::Operand<'_>,
        rhs: llir::Operand<'_>,
    ) {
        self.inst_printer
            .print_inst(llir::Instruction::set_eq(result, lhs, rhs).unwrap().into())
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
            self.helper
                .print_equals(new_guard, index.into(), 0i32.int_literal());
            if let Some(guard) = guard {
                self.helper.print_and(guard, guard.into(), new_guard.into());
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
        let size = self.namer.name_size(block.local_size(), Type::I(32));
        let d0 = self
            .namer
            .name_index(fun.block_dims()[0].id())
            .into_operand();
        let var = fun.block_dims()[1..].iter().fold(d0, |old_var, dim| {
            let var = self.namer.gen_name(Type::I(32));
            let size = self.namer.name_size(dim.size(), Type::I(32));
            let idx = self.namer.name_index(dim.id()).into_operand();
            self.helper.inst_printer.print_inst(
                llir::Instruction::imad_low(var, old_var, size, idx)
                    .unwrap()
                    .into(),
            );
            var.into_operand()
        });
        self.helper.inst_printer.print_inst(
            llir::Instruction::imad(addr, var, size, addr.into_operand())
                .unwrap()
                .into(),
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
        self.helper.print_move(idx, 0i32.int_literal());
        let mut ind_var_vec = vec![];
        let loop_label = self.namer.gen_label("LOOP");
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
                [ref base] => self.helper.print_move(ind_var, base.clone()),
                [ref lhs, ref rhs] => {
                    self.helper.print_add_int(ind_var, lhs.clone(), rhs.clone())
                }
                _ => panic!(),
            };
            ind_var_vec.push(ind_var);
        }
        self.helper.inst_printer.print_label(loop_label);
        self.cfg_vec(fun, cfgs);
        for (level, ind_var) in ind_levels.iter().zip_eq(ind_var_vec) {
            if let Some((_, ref increment)) = level.increment {
                let step = self.namer.name_size(increment, level.t());
                self.helper.print_add_int(ind_var, ind_var.into(), step);
            };
        }
        self.helper
            .print_add_int(idx, idx.into(), 1i32.int_literal());
        let lt_cond = self.namer.gen_name(ir::Type::I(1));
        let size = self.namer.name_size(dim.size(), Type::I(32));
        self.helper.print_lt_int(lt_cond, idx.into(), size);
        self.helper
            .inst_printer
            .print_inst(llir::Instruction::jump(loop_label).predicated(lt_cond));
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
                    self.helper.print_add_int(tmp, lhs.clone(), rhs.clone());
                    tmp.into()
                }
                _ => panic!(),
            };
            if let Some((_, ref incr)) = level.increment {
                incr_levels.push((level, ind_var, incr, base.clone()));
            }
            self.helper.print_move(ind_var, base);
        }
        for i in 0..dim.size().as_int().unwrap() {
            self.namer.set_current_index(dim, i);
            if i > 0 {
                for &(level, ind_var, ref incr, ref base) in &incr_levels {
                    if let Some(step) = incr.as_int() {
                        let stepxi = i32::try_from(step * i)
                            .unwrap()
                            .typed_int_literal(level.t())
                            .unwrap();
                        self.helper.print_add_int(ind_var, stepxi, base.clone());
                    } else {
                        let step = self.namer.name_size(incr, level.t());
                        self.helper.print_add_int(ind_var, step, ind_var.into());
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
            match base_components[..] {
                [] => self.helper.inst_printer.print_inst(
                    llir::Instruction::imul(ind_var, index, step)
                        .unwrap()
                        .into(),
                ),
                [ref base] => self.helper.inst_printer.print_inst(
                    llir::Instruction::imad(ind_var, index, step, base.clone())
                        .unwrap()
                        .into(),
                ),
                _ => panic!(),
            }
        } else {
            match base_components[..] {
                [] => self
                    .helper
                    .print_move(ind_var, 0i32.typed_int_literal(ind_var.t()).unwrap()),
                [ref base] => self.helper.print_move(ind_var, base.clone()),
                [ref lhs, ref rhs] => {
                    self.helper.print_add_int(ind_var, lhs.clone(), rhs.clone())
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
                self.helper
                    .inst_printer
                    .print_inst(llir::Instruction::sync().into());
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
                .map(|d| d.size().as_int().unwrap())
                .product(),
            vector_levels[1]
                .iter()
                .map(|d| d.size().as_int().unwrap())
                .product(),
        ];
        let helper = &mut self.helper;
        match inst.operator() {
            &op::BinOp(op, ref lhs, ref rhs, round) => helper.inst_printer.print_inst(
                llir::Instruction::binary(
                    llir::BinOp::from_ir(
                        op,
                        round,
                        lower_type(lhs.t(), fun),
                        lower_type(rhs.t(), fun),
                    )
                    .unwrap(),
                    self.namer.vector_inst(vector_levels, inst.id()),
                    self.namer.vector_operand(vector_levels, lhs),
                    self.namer.vector_operand(vector_levels, rhs),
                )
                .unwrap()
                .into(),
            ),
            &op::Mul(ref lhs, ref rhs, round, return_type) => {
                helper.inst_printer.print_inst(
                    llir::Instruction::binary(
                        llir::BinOp::from_ir_mul(
                            round,
                            lower_type(lhs.t(), fun),
                            lower_type(rhs.t(), fun),
                            lower_type(return_type, fun),
                        )
                        .unwrap(),
                        self.namer.vector_inst(vector_levels, inst.id()),
                        self.namer.vector_operand(vector_levels, lhs),
                        self.namer.vector_operand(vector_levels, rhs),
                    )
                    .unwrap()
                    .into(),
                )
            }
            &op::Mad(ref mul_lhs, ref mul_rhs, ref add_rhs, round) => {
                helper.inst_printer.print_inst(
                    llir::Instruction::ternary(
                        llir::TernOp::from_ir_mad(
                            round,
                            lower_type(mul_lhs.t(), fun),
                            lower_type(mul_rhs.t(), fun),
                            lower_type(add_rhs.t(), fun),
                        )
                        .unwrap(),
                        self.namer.vector_inst(vector_levels, inst.id()),
                        self.namer.vector_operand(vector_levels, mul_lhs),
                        self.namer.vector_operand(vector_levels, mul_rhs),
                        self.namer.vector_operand(vector_levels, add_rhs),
                    )
                    .unwrap()
                    .into(),
                )
            }
            &op::UnaryOp(operator, ref operand) => {
                // Need to lower inner types
                let operator = match operator {
                    ir::UnaryOp::Cast(t) => ir::UnaryOp::Cast(lower_type(t, fun)),
                    ir::UnaryOp::Exp(t) => ir::UnaryOp::Exp(lower_type(t, fun)),
                    _ => operator,
                };
                helper.inst_printer.print_inst(
                    llir::Instruction::unary(
                        llir::UnOp::from_ir(operator, lower_type(operand.t(), fun))
                            .unwrap(),
                        self.namer.vector_inst(vector_levels, inst.id()),
                        self.namer.vector_operand(vector_levels, operand),
                    )
                    .unwrap()
                    .into(),
                )
            }
            &op::Ld(ld_type, ref addr, ref pattern) => helper.inst_printer.print_inst(
                llir::Instruction::load(
                    llir::LoadSpec::from_ir(
                        vector_factors,
                        lower_type(ld_type, fun),
                        access_pattern_space(pattern, fun.space()),
                        inst.mem_flag().unwrap(),
                    )
                    .unwrap(),
                    self.namer.vector_inst(vector_levels, inst.id()),
                    self.namer.name_op(addr).try_into().unwrap(),
                )
                .unwrap()
                .into(),
            ),
            op::St(addr, val, _, pattern) => {
                let guard = if inst.has_side_effects() {
                    self.namer.side_effect_guard()
                } else {
                    None
                };
                helper.inst_printer.print_inst(
                    llir::Instruction::store(
                        llir::StoreSpec::from_ir(
                            vector_factors,
                            lower_type(val.t(), fun),
                            access_pattern_space(pattern, fun.space()),
                            inst.mem_flag().unwrap(),
                        )
                        .unwrap(),
                        self.namer.name_op(addr).try_into().unwrap(),
                        self.namer.vector_operand(vector_levels, val),
                    )
                    .unwrap()
                    .predicated(guard),
                )
            }
            op @ op::TmpLd(..) | op @ op::TmpSt(..) => {
                panic!("non-printable instruction {:?}", op)
            }
        }
    }
}
