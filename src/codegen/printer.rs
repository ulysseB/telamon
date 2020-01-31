use std::convert::{TryFrom, TryInto};
use std::{fmt, ops};

use itertools::Itertools;

use crate::ir::{self, op, Type};
use crate::search_space::*;

use super::llir::{FloatLiteral as _, IntLiteral as _};
use super::*;

fn ndrange<K, Idx, II>(into_iter: II) -> impl Iterator<Item = Vec<(K, Idx)>>
where
    K: Clone,
    Idx: Default + Clone,
    ops::Range<Idx>: Iterator<Item = Idx>,
    II: IntoIterator<Item = (K, Idx)>,
{
    into_iter
        .into_iter()
        .map(|(key, size)| (Idx::default()..size).map(move |pos| (key.clone(), pos)))
        .multi_cartesian_product()
        .pad_using(1, |_| Vec::new())
}

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
    fn print_inst<T: Into<llir::PredicatedInstruction<'a>>>(&mut self, inst: T) {
        self.inst_printer.print_inst(inst.into())
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
        I: Iterator<Item = &'d Dimension>,
    {
        let mut guard: Option<llir::Register<'_>> = None;
        for dim in threads {
            let new_guard = self.namer.gen_name(ir::Type::I(1));
            let index = self.namer.name_index_as_operand(dim.id());
            self.helper
                .print_equals(new_guard, index, 0i32.int_literal());
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
        let d0 = self.namer.name_index_as_operand(fun.block_dims()[0].id());
        let var = fun.block_dims()[1..].iter().fold(d0, |old_var, dim| {
            let var = self.namer.gen_name(Type::I(32));
            let size = self.namer.name_size(dim.size(), Type::I(32));
            let idx = self.namer.name_index_as_operand(dim.id());
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
    fn standard_loop(&mut self, fun: &Function, dim: &Dimension, cfgs: &'b [Cfg<'b>]) {
        let idx = self.namer.name_index(dim.id());
        self.helper.print_move(idx, 0i32.int_literal());

        for instruction in self.namer.expr_to_operand().init_exprs(dim.id()) {
            self.helper.print_inst(instruction.clone());
        }

        let loop_label = self.namer.gen_label("LOOP");
        self.helper.inst_printer.print_label(loop_label);

        for instruction in self.namer.expr_to_operand().compute_exprs(Some(dim.id())) {
            self.helper.print_inst(instruction.clone());
        }

        self.cfg_vec(fun, cfgs);

        for instruction in self.namer.expr_to_operand().update_exprs(dim.id()) {
            self.helper.print_inst(instruction.clone());
        }

        self.helper.print_inst(
            llir::Instruction::iadd(idx, idx.into_operand(), 1i32.int_literal()).unwrap(),
        );

        let lt_cond = self.namer.gen_name(ir::Type::I(1));
        let size = self.namer.name_size(dim.size(), Type::I(32));
        self.helper.print_lt_int(lt_cond, idx.into(), size);
        self.helper
            .inst_printer
            .print_inst(llir::Instruction::jump(loop_label).predicated(lt_cond));

        for instruction in self.namer.expr_to_operand().reset_exprs(dim.id()) {
            self.helper.print_inst(instruction.clone());
        }
    }

    /// Prints an unroll loop - loop without jumps
    fn unroll_loop(&mut self, fun: &Function, dim: &Dimension, cfgs: &'b [Cfg<'b>]) {
        for instruction in self.namer.expr_to_operand().init_exprs(dim.id()) {
            self.helper.print_inst(instruction.clone());
        }

        for i in 0..dim.size().as_int().unwrap() {
            self.namer.set_current_index(dim, i);

            for instruction in self.namer.expr_to_operand().compute_exprs(Some(dim.id()))
            {
                self.helper.print_inst(instruction.clone());
            }

            self.cfg_vec(fun, cfgs);

            for instruction in self.namer.expr_to_operand().update_exprs(dim.id()) {
                self.helper.print_inst(instruction.clone());
            }
        }
        self.namer.unset_current_index(dim);

        for instruction in self.namer.expr_to_operand().reset_exprs(dim.id()) {
            self.helper.print_inst(instruction.clone());
        }
    }

    /// Prints a Loop
    fn gen_loop(&mut self, fun: &Function, dim: &Dimension, cfgs: &'b [Cfg<'b>]) {
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
            Cfg::Root(cfgs) => {
                for instruction in self.namer.expr_to_operand().compute_exprs(None) {
                    self.helper.print_inst(instruction.clone());
                }

                self.cfg_vec(fun, cfgs)
            }
            Cfg::Loop { dimension, body } => self.gen_loop(fun, dimension, body),
            Cfg::Threads(dims, inner) => {
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
        let llinst = match inst.operator() {
            &op::BinOp(op, ref lhs, ref rhs, round) => llir::Instruction::binary(
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
            &op::Mul(ref lhs, ref rhs, round, return_type) => llir::Instruction::binary(
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
            &op::Mad(ref mul_lhs, ref mul_rhs, ref add_rhs, round) => {
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
                .into()
            }
            &op::UnaryOp(operator, ref operand) => {
                // Need to lower inner types
                let operator = match operator {
                    ir::UnaryOp::Cast(t) => ir::UnaryOp::Cast(lower_type(t, fun)),
                    ir::UnaryOp::Exp(t) => ir::UnaryOp::Exp(lower_type(t, fun)),
                    _ => operator,
                };
                llir::Instruction::unary(
                    llir::UnOp::from_ir(operator, lower_type(operand.t(), fun)).unwrap(),
                    self.namer.vector_inst(vector_levels, inst.id()),
                    self.namer.vector_operand(vector_levels, operand),
                )
                .unwrap()
                .into()
            }
            &op::Ld(ld_type, ref addr, ref pattern) => {
                let spec = llir::LoadSpec::from_ir(
                    vector_factors,
                    lower_type(ld_type, fun),
                    access_pattern_space(pattern, fun.space()),
                    inst.mem_flag().unwrap(),
                )
                .unwrap();

                let dst = self.namer.vector_inst(vector_levels, inst.id());

                let (addr, predicate) = self.namer.name_op(addr);

                if predicate.is_some() {
                    let zero = 0f32.float_literal();
                    match &dst {
                        &llir::ScalarOrVector::Scalar(dst) => {
                            self.helper
                                .print_inst(llir::Instruction::mov(dst, zero).unwrap());
                        }
                        llir::ScalarOrVector::Vector(dst) => {
                            for &dst in dst {
                                self.helper.print_inst(
                                    llir::Instruction::mov(dst, zero.clone()).unwrap(),
                                );
                            }
                        }
                    }
                }

                llir::Instruction::load(spec, dst, addr.try_into().unwrap())
                    .unwrap()
                    .predicated(predicate)
            }
            op::St(addr, val, _, pattern) => {
                let (addr, predicate) = self.namer.name_op(addr);
                assert!(predicate.is_none(), "predicated store");

                let guard = if inst.has_side_effects() {
                    self.namer.side_effect_guard()
                } else {
                    None
                };
                llir::Instruction::store(
                    llir::StoreSpec::from_ir(
                        vector_factors,
                        lower_type(val.t(), fun),
                        access_pattern_space(pattern, fun.space()),
                        inst.mem_flag().unwrap(),
                    )
                    .unwrap(),
                    addr.try_into().unwrap(),
                    self.namer.vector_operand(vector_levels, val),
                )
                .unwrap()
                .predicated(guard)
            }
            op @ op::TmpLd(..) | op @ op::TmpSt(..) => {
                panic!("non-printable instruction {:?}", op)
            }
        };
        self.helper.inst_printer.print_inst(llinst);
    }
}
