use std::borrow::{Borrow, Cow};
use std::ops;
use std::rc::Rc;

use fxhash::{FxHashMap, FxHashSet};
use itertools::Itertools;
use utils::*;

use crate::codegen::*;
use crate::ir::{self, op, Type};
use crate::search_space::*;

use super::iteration::IterationVarId;
use super::predicates::PredicateId;

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

    fn output_type(self, arg_t: Type) -> Type {
        match (self, arg_t) {
            (MulMode::Wide, Type::I(i)) => Type::I(2 * i),
            (MulMode::Low, Type::I(i)) | (MulMode::High, Type::I(i)) => Type::I(i),
            (MulMode::Empty, Type::F(f)) => Type::F(f),
            _ => panic!("Invalid mul mode type combination"),
        }
    }
}

use super::name_map::{IntoNameable, Nameable};

#[derive(Debug, Copy, Clone)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    fn inverse(self) -> Self {
        match self {
            CmpOp::Eq => CmpOp::Ne,
            CmpOp::Ne => CmpOp::Eq,
            CmpOp::Lt => CmpOp::Ge,
            CmpOp::Le => CmpOp::Gt,
            CmpOp::Gt => CmpOp::Le,
            CmpOp::Ge => CmpOp::Lt,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum BoolOp {
    And,
    Or,
}

/// A predicate expression
#[derive(Clone)]
pub enum PredExpr<'a> {
    /// Typed comparison between two registers
    ///
    /// TODO: ftz for float
    CmpOp {
        op: CmpOp,
        t: ir::Type,
        lhs: Nameable<'a>,
        rhs: Nameable<'a>,
    },
    /// Boolean operation
    BoolOp(BoolOp, Vec<Cow<'a, PredExpr<'a>>>),
    /// Negation
    Not(Box<Cow<'a, PredExpr<'a>>>),
    /// Variable
    Variable(Nameable<'a>),
}

impl<'a> ops::Not for PredExpr<'a> {
    type Output = PredExpr<'a>;

    fn not(self) -> PredExpr<'a> {
        match self {
            PredExpr::Not(inner) => inner.into_owned(),
            _ => PredExpr::Not(Box::new(Cow::Owned(self))),
        }
    }
}

impl<'a> PredExpr<'a> {
    fn new_cmp<L, R>(op: CmpOp, t: ir::Type, lhs: L, rhs: R) -> Self
    where
        L: IntoNameable<'a>,
        R: IntoNameable<'a>,
    {
        PredExpr::CmpOp {
            op,
            t,
            lhs: lhs.into_nameable(),
            rhs: rhs.into_nameable(),
        }
    }

    fn new_lt<L, R>(t: ir::Type, lhs: L, rhs: R) -> Self
    where
        L: IntoNameable<'a>,
        R: IntoNameable<'a>,
    {
        Self::new_cmp(CmpOp::Lt, t, lhs, rhs)
    }

    fn to_bool(&self) -> Option<bool> {
        match self {
            PredExpr::BoolOp(op, args) if args.is_empty() => Some(match op {
                BoolOp::And => true,
                BoolOp::Or => false,
            }),
            PredExpr::Not(arg) => arg.to_bool().map(|x| !x),
            _ => None,
        }
    }
}

pub fn predicates_from_range_and_guard<'a, B, RI>(
    space: &SearchSpace,
    ranges: RI,
    guard: Option<Nameable<'a>>,
) -> PredExpr<'a>
where
    B: Borrow<ir::RangePredicate>,
    RI: IntoIterator<Item = B>,
{
    PredExpr::BoolOp(
        BoolOp::And,
        ranges
            .into_iter()
            .map(|range| {
                let range = range.borrow();
                Cow::Owned(PredExpr::new_lt(
                    ir::Type::I(32),
                    range.induction_variable(),
                    Size::from_ir(&ir::PartialSize::from(range.bound().clone()), space),
                ))
            })
            .chain(guard.map(|guard| Cow::Owned(PredExpr::Variable(guard))))
            .collect(),
    )
}

#[derive(Debug, Clone)]
pub enum IntExpr<'a> {
    Named(Nameable<'a>, ir::Type),
    Add {
        arg_t: ir::Type,
        lhs: IntExprPtr<'a>,
        rhs: IntExprPtr<'a>,
    },
    Mad {
        arg_t: ir::Type,
        mul_mode: MulMode,
        mlhs: IntExprPtr<'a>,
        mrhs: IntExprPtr<'a>,
        arhs: IntExprPtr<'a>,
    },
    Mul {
        arg_t: ir::Type,
        mul_mode: MulMode,
        lhs: IntExprPtr<'a>,
        rhs: IntExprPtr<'a>,
    },
}

pub type IntExprPtr<'a> = Rc<IntExpr<'a>>;

impl<'a> IntExpr<'a> {
    // NB: the result may or may not be in `result`.  The caller needs to issue a `move` itself to
    // put it there if needed.
    fn compile<VP: ValuePrinter>(
        &self,
        result: Option<Nameable<'a>>,
        t: ir::Type,
        namer: &mut NameMap<'_, '_, VP>,
        flow: &mut Vec<(Nameable<'a>, IntInst<'a>)>,
    ) -> Nameable<'a> {
        let (out, out_t) = match self {
            IntExpr::Named(arg, arg_t) => {
                if *arg_t == t {
                    if let Some(result) = result {
                        flow.push((
                            result.clone(),
                            IntInst::new_move(*arg_t, arg.clone()),
                        ));
                        return result;
                    }
                }

                (arg.clone(), *arg_t)
            }
            IntExpr::Add { arg_t, lhs, rhs } => {
                let lhs = lhs.compile(None, *arg_t, namer, flow);
                let rhs = rhs.compile(
                    result.as_ref().filter(|_| t == *arg_t).cloned(),
                    *arg_t,
                    namer,
                    flow,
                );

                let out = result
                    .as_ref()
                    .filter(|_| t == *arg_t)
                    .cloned()
                    .unwrap_or_else(|| namer.gen_name(*arg_t).into_nameable());
                flow.push((out.clone(), IntInst::new_add(*arg_t, lhs, rhs)));
                (out, *arg_t)
            }
            IntExpr::Mul {
                arg_t,
                mul_mode,
                lhs,
                rhs,
            } => {
                let out_t = mul_mode.output_type(*arg_t);
                let lhs = lhs.compile(None, *arg_t, namer, flow);
                let rhs = rhs.compile(
                    result.as_ref().filter(|_| t == *arg_t).cloned(),
                    *arg_t,
                    namer,
                    flow,
                );

                let out = result
                    .as_ref()
                    .filter(|_| t == out_t)
                    .cloned()
                    .unwrap_or_else(|| namer.gen_name(out_t).into_nameable());
                flow.push((out.clone(), IntInst::new_mul(*arg_t, *mul_mode, lhs, rhs)));
                (out, out_t)
            }
            IntExpr::Mad {
                arg_t,
                mul_mode,
                mlhs,
                mrhs,
                arhs,
            } => {
                let out_t = mul_mode.output_type(*arg_t);
                let mlhs = mlhs.compile(None, *arg_t, namer, flow);
                let mrhs = mrhs.compile(None, *arg_t, namer, flow);
                let arhs = arhs.compile(
                    result.as_ref().filter(|_| t == *arg_t).cloned(),
                    *arg_t,
                    namer,
                    flow,
                );

                let out = result
                    .as_ref()
                    .filter(|_| t == out_t)
                    .cloned()
                    .unwrap_or_else(|| namer.gen_name(out_t).into_nameable());
                flow.push((
                    out.clone(),
                    IntInst::new_mad(*arg_t, *mul_mode, mlhs, mrhs, arhs),
                ));
                (out, out_t)
            }
            _ => panic!("invalid types"),
        };

        if out_t != t {
            let result = result.unwrap_or_else(|| namer.gen_name(t).into_nameable());
            flow.push((result.clone(), IntInst::new_cast(out, out_t, t)));
            result
        } else {
            out
        }
    }
}

#[derive(Debug, Clone)]
pub enum IntInst<'a> {
    Move(Nameable<'a>, ir::Type),
    Cast(Nameable<'a>, ir::Type, ir::Type),
    Add {
        arg_t: ir::Type,
        lhs: Nameable<'a>,
        rhs: Nameable<'a>,
    },
    Sub {
        arg_t: ir::Type,
        lhs: Nameable<'a>,
        rhs: Nameable<'a>,
    },
    Div {
        arg_t: ir::Type,
        lhs: Nameable<'a>,
        rhs: Nameable<'a>,
    },
    Min {
        arg_t: ir::Type,
        lhs: Nameable<'a>,
        rhs: Nameable<'a>,
    },
    Mad {
        arg_t: ir::Type,
        mul_mode: MulMode,
        mlhs: Nameable<'a>,
        mrhs: Nameable<'a>,
        arhs: Nameable<'a>,
    },
    Mul {
        arg_t: ir::Type,
        mul_mode: MulMode,
        lhs: Nameable<'a>,
        rhs: Nameable<'a>,
    },
}

impl<'a> IntInst<'a> {
    fn new_move<T: IntoNameable<'a>>(t: ir::Type, arg: T) -> Self {
        IntInst::Move(arg.into_nameable(), t)
    }

    fn new_cast<T: IntoNameable<'a>>(arg: T, from_t: ir::Type, to_t: ir::Type) -> Self {
        IntInst::Cast(arg.into_nameable(), from_t, to_t)
    }

    fn new_add<A: IntoNameable<'a>, B: IntoNameable<'a>>(
        arg_t: ir::Type,
        lhs: A,
        rhs: B,
    ) -> Self {
        IntInst::Add {
            arg_t,
            lhs: lhs.into_nameable(),
            rhs: rhs.into_nameable(),
        }
    }

    fn new_sub<A: IntoNameable<'a>, B: IntoNameable<'a>>(
        arg_t: ir::Type,
        lhs: A,
        rhs: B,
    ) -> Self {
        IntInst::Sub {
            arg_t,
            lhs: lhs.into_nameable(),
            rhs: rhs.into_nameable(),
        }
    }

    fn new_div<A: IntoNameable<'a>, B: IntoNameable<'a>>(
        arg_t: ir::Type,
        lhs: A,
        rhs: B,
    ) -> Self {
        IntInst::Div {
            arg_t,
            lhs: lhs.into_nameable(),
            rhs: rhs.into_nameable(),
        }
    }

    fn new_min<A: IntoNameable<'a>, B: IntoNameable<'a>>(
        arg_t: ir::Type,
        lhs: A,
        rhs: B,
    ) -> Self {
        IntInst::Min {
            arg_t,
            lhs: lhs.into_nameable(),
            rhs: rhs.into_nameable(),
        }
    }

    fn new_mad<A: IntoNameable<'a>, B: IntoNameable<'a>, C: IntoNameable<'a>>(
        arg_t: ir::Type,
        mul_mode: MulMode,
        mlhs: A,
        mrhs: B,
        arhs: C,
    ) -> Self {
        IntInst::Mad {
            arg_t,
            mul_mode,
            mlhs: mlhs.into_nameable(),
            mrhs: mrhs.into_nameable(),
            arhs: arhs.into_nameable(),
        }
    }

    fn new_mul<A: IntoNameable<'a>, B: IntoNameable<'a>>(
        arg_t: ir::Type,
        mul_mode: MulMode,
        lhs: A,
        rhs: B,
    ) -> Self {
        IntInst::Mul {
            arg_t,
            mul_mode,
            lhs: lhs.into_nameable(),
            rhs: rhs.into_nameable(),
        }
    }
}

pub struct Loop<'a> {
    pub dim: ir::DimId,
    pub label: u32,
    // for (lhs, rhs) in init: lhs = rhs
    pub inits: Vec<(Nameable<'a>, IntInst<'a>)>,
    // condition is `idx < bound`
    pub index: Nameable<'a>,
    pub bound: Nameable<'a>,
    // for (lhs, rhs) in update: lhs += rhs
    pub increments: Vec<(Nameable<'a>, IntInst<'a>)>,
}

fn iteration_var_def<'a, VP: ValuePrinter>(
    fun: &'a Function,
    iter_id: IterationVarId,
    namer: &mut NameMap<'_, '_, VP>,
) -> Vec<(Nameable<'a>, IntInst<'a>)> {
    let mut instructions = Vec::new();
    let mut expr = None;
    for (dim, size) in fun.iteration_variables()[iter_id].outer_dims() {
        if let Some(expr) = expr.as_mut() {
            let tmp = namer.gen_name(ir::Type::I(32));
            instructions.push((
                tmp.clone().into_nameable(),
                std::mem::replace(
                    expr,
                    IntInst::new_mad(ir::Type::I(32), MulMode::Low, dim, size, tmp),
                ),
            ));
        } else {
            expr = Some(IntInst::new_mul(ir::Type::I(32), MulMode::Low, dim, size));
        }
    }

    instructions.push((
        iter_id.into_nameable(),
        expr.unwrap_or_else(|| IntInst::new_move(ir::Type::I(32), 0i32)),
    ));
    instructions
}

#[allow(clippy::too_many_arguments)]
pub trait InstPrinter {
    type ValuePrinter: ValuePrinter;

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
        predicate: Option<(&str, Cow<'_, str>)>,
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
        op: &'a ir::Operand,
        namer: &'a NameMap<Self::ValuePrinter>,
    ) -> Cow<'a, str>;

    /// Names an instruction, vectorized on the given dimensions.
    fn name_inst<'a>(
        vector_levels: &[Vec<Dimension>; 2],
        inst: ir::InstId,
        namer: &'a NameMap<Self::ValuePrinter>,
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

    fn cfg_vec(
        &mut self,
        fun: &Function,
        cfgs: &[Cfg],
        namer: &mut NameMap<Self::ValuePrinter>,
    ) {
        for c in cfgs.iter() {
            self.cfg(fun, c, namer);
        }
    }

    fn init_inst_predicates(
        &mut self,
        fun: &Function,
        inst_id: ir::InstId,
        namer: &mut NameMap<Self::ValuePrinter>,
    ) {
        let const_true = true.into_nameable();

        // Those are all the dimensions the predicates iterate over.  Note that this can be a
        // strict subset of the instruction's instantiation dimension.
        for indices in fun.instruction_predicates()[&inst_id]
            .iter()
            .map(|&(dim, size)| (0..size).map(move |pos| (dim, pos)))
            .multi_cartesian_product()
            .pad_using(1, |_| vec![])
        {
            let pred_name =
                namer.name_instruction_predicate(inst_id, &indices.into_iter().collect());
            self.print_move(ir::Type::I(1), &pred_name, &const_true.name(namer));
        }
    }

    // Print updating the inst predicates, assuming that they were all set to true initially.
    fn update_inst_predicates(
        &mut self,
        fun: &Function,
        inst_id: ir::InstId,
        predicates: &[PredicateId],
        namer: &mut NameMap<Self::ValuePrinter>,
    ) {
        let const_false = false.into_nameable();
        let instantiation_dims: FxHashMap<_, _> = fun.instruction_predicates()[&inst_id]
            .iter()
            .cloned()
            .collect();

        // Stores the current index.  Reused across unrolls.
        let idx = namer.gen_name(ir::Type::I(32));
        // Stores the final condition.  Reused across unrolls.
        let lt_cond = namer.gen_name(ir::Type::I(1));

        for &pred_id in predicates {
            let predicate = &fun.predicates()[pred_id];
            // Those are the dimensions this range predicate iterates over and we need to fix for
            // the others (if there are multiple predicates which share an iteration).
            let my_dims: FxHashSet<_> =
                predicate.instantiation_dims().map(|(dim, _)| dim).collect();

            // The instantiation dims are sorted in decreasing stride order, so that the smallest
            // stride is fastest varying.  Since we also iterate in reverse size order, the
            // condition `idx < range` is monotonous, and we can jump to the end as soon as it
            // becomes true.
            for pred_indices in predicate
                .instantiation_dims()
                .map(|&(dim, stride)| {
                    (0..instantiation_dims[&dim])
                        .rev()
                        .map(move |pos| (dim, pos, stride))
                })
                .multi_cartesian_product()
                .pad_using(1, |_| vec![])
            {
                // Compute the index for this iteration based on the iteration variable
                let var = namer.name_iteration_var(predicate.iteration_var());
                self.print_move(ir::Type::I(32), &idx, &var);

                let constant = pred_indices
                    .iter()
                    .map(|&(_, pos, stride)| (pos as u32) * stride)
                    .sum::<u32>();

                if constant != 0 {
                    self.print_int_inst(
                        &(&idx).into_nameable(),
                        &IntInst::new_add(ir::Type::I(32), &idx, constant as i32),
                        namer,
                    );
                }

                // Check if we are in bound...
                self.print_lt_int(
                    ir::Type::I(32),
                    &lt_cond,
                    &idx,
                    &namer.name_size(predicate.bound(), ir::Type::I(32)),
                );

                // ... and update the predicates
                for other_dims in instantiation_dims
                    .iter()
                    .filter(|(dim, _)| !my_dims.contains(dim))
                    .map(|(&dim, &size)| (0..size).map(move |pos| (dim, pos)))
                    .multi_cartesian_product()
                    .pad_using(1, |_| vec![])
                {
                    let pred_name = namer.name_instruction_predicate(
                        inst_id,
                        &other_dims
                            .into_iter()
                            .chain(pred_indices.iter().map(|&(dim, pos, _)| (dim, pos)))
                            .collect(),
                    );

                    self.print_and(ir::Type::I(1), pred_name, pred_name, &lt_cond);
                }
            }
        }
    }

    /// Prints a cfg.
    fn cfg<'a>(
        &mut self,
        fun: &Function,
        c: &Cfg<'a>,
        namer: &mut NameMap<Self::ValuePrinter>,
    ) {
        match c {
            Cfg::Root(cfgs) => {
                // Iteration variables
                let iteration_vars = fun.iteration_variables();
                for iter_id in iteration_vars.global_defs() {
                    for (target, inst) in iteration_var_def(fun, iter_id, namer) {
                        self.print_int_inst(&target, &inst, namer);
                    }
                }

                // Predicates
                if fun.predicate_accesses() {
                    for &(inst_id, ref predicates) in fun.global_predicates() {
                        self.init_inst_predicates(fun, inst_id, namer);
                        self.update_inst_predicates(fun, inst_id, &predicates[..], namer);
                    }
                }

                self.cfg_vec(fun, cfgs, namer)
            }
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
        namer: &NameMap<Self::ValuePrinter>,
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
                    let zero = namer.value_printer().get_const_int(&0.into(), 32);
                    self.print_move(level.t(), &ind_var, &zero);
                }
                [ref base] => self.print_move(level.t(), &ind_var, &base),
                [ref lhs, ref rhs] => self.print_add_int(level.t(), &ind_var, &lhs, &rhs),
                _ => panic!(),
            }
        }
    }

    /// Change the side-effect guards so that the specified threads are disabled.
    fn disable_threads<'a, I>(
        &mut self,
        threads: I,
        namer: &mut NameMap<Self::ValuePrinter>,
    ) where
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
                &namer.value_printer().get_const_int(&0.into(), 32),
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
        namer: &mut NameMap<Self::ValuePrinter>,
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
        namer: &mut NameMap<Self::ValuePrinter>,
    ) {
        let idx = dim.id().into_nameable();

        let mut init_vec: Vec<(Nameable<'_>, IntInst<'_>)> =
            vec![(idx.clone(), IntInst::new_move(ir::Type::I(32), 0i32))];
        let mut update_vec: Vec<(Nameable<'_>, IntInst<'_>)> = vec![(
            idx.clone(),
            IntInst::new_add(ir::Type::I(32), idx.clone(), 1i32),
        )];
        let mut reset_vec: Vec<(Nameable<'_>, IntInst<'_>)> = Vec::new();

        let ind_levels = dim.induction_levels();
        for level in ind_levels.iter() {
            let dim_id = level.increment.as_ref().map(|&(dim, _)| dim);
            let ind_var = (level.ind_var, dim_id);
            match level.base.components().collect_vec()[..] {
                [base] => init_vec
                    .push((ind_var.into_nameable(), IntInst::new_move(level.t(), base))),
                [lhs, rhs] => init_vec.push((
                    ind_var.into_nameable(),
                    IntInst::new_add(level.t(), lhs, rhs),
                )),
                _ => panic!(),
            };

            if let Some((_, increment)) = &level.increment {
                update_vec.push((
                    ind_var.into_nameable(),
                    IntInst::new_add(level.t(), ind_var, (increment, level.t())),
                ));
            }
        }

        let iteration_vars = fun.iteration_variables();
        for iter_id in iteration_vars.loop_defs(dim.id()) {
            init_vec.extend(iteration_var_def(fun, iter_id, namer));
        }

        for (iter_id, increment) in iteration_vars.loop_updates(dim.id()) {
            update_vec.push((
                iter_id.into_nameable(),
                IntInst::new_add(ir::Type::I(32), iter_id, (increment, ir::Type::I(32))),
            ));

            // Reset the variable after the loop.  This should usually end up fused with the
            // increment of the outer loop if there is one.
            let reset_inst = if let Some(increment) = increment.as_int() {
                IntInst::new_mad(
                    ir::Type::I(32),
                    MulMode::Low,
                    -(increment as i32),
                    dim.size(),
                    iter_id,
                )
            } else if let Some(size) = dim.size().as_int() {
                IntInst::new_mad(
                    ir::Type::I(32),
                    MulMode::Low,
                    (increment, ir::Type::I(32)),
                    -(size as i32),
                    iter_id,
                )
            } else {
                panic!("dynamic increment and size")
            };

            reset_vec.push((iter_id.into_nameable(), reset_inst));
        }

        // TODO Predicates
        // TODO: must count the number of iterations that we can do.
        let mut min_size: Option<String> = None;
        if fun.predicate_accesses() {
            // Print init here
            for (target, inst) in init_vec.drain(..) {
                self.print_int_inst(&target, &inst, namer);
            }

            for &(inst_id, ref predicates) in fun.loop_predicates(dim.id()) {
                let max_sizes: FxHashMap<_, _> = fun.instruction_predicates()[&inst_id]
                    .iter()
                    .map(|&(dim, size)| (dim, size - 1))
                    .collect();

                for &pid in predicates {
                    let cur_size = namer.gen_name(ir::Type::I(32));
                    let pred = &fun.predicates()[pid];
                    let var = namer.name_iteration_var(pred.iteration_var());
                    let max_offset = pred
                        .instantiation_dims()
                        .map(|&(dim, stride)| (max_sizes[&dim] as u32 - 1) * stride)
                        .sum::<u32>() as i32;
                    let bound = namer.name_size(pred.bound(), ir::Type::I(32));
                    self.print_add_int(
                        ir::Type::I(32),
                        &cur_size,
                        &var,
                        &max_offset.into_nameable().name(namer),
                    );
                    let this_stride = fun.iteration_variables()[pred.iteration_var()]
                        .stride_for(dim.id())
                        .map(|s| s.as_int().unwrap() as i32);

                    if let Some(this_stride) = this_stride {
                        // TODO: Use this predicate value when initializing; it is independant of the
                        // loop.
                        self.print_int_inst(
                            &(&cur_size).into_nameable(),
                            &IntInst::new_sub(ir::Type::I(32), &*bound, &cur_size),
                            namer,
                        );
                        self.print_int_inst(
                            &(&cur_size).into_nameable(),
                            &IntInst::new_div(ir::Type::I(32), &cur_size, this_stride),
                            namer,
                        );
                        if let Some(min_size) = &min_size {
                            self.print_int_inst(
                                &min_size.into_nameable(),
                                &IntInst::new_min(ir::Type::I(32), min_size, &cur_size),
                                namer,
                            );
                        } else {
                            self.print_int_inst(
                                &(&cur_size).into_nameable(),
                                &IntInst::new_min(ir::Type::I(32), &cur_size, dim.size()),
                                namer,
                            );
                            min_size = Some(cur_size);
                        }
                    }
                }

                self.init_inst_predicates(fun, inst_id, namer);
            }
        }

        let iter_label = namer.gen_loop_id();

        self.print_loop(
            fun,
            &Loop {
                dim: dim.id(),
                label: iter_label,
                inits: init_vec,
                index: idx.clone(),
                bound: min_size
                    .as_ref()
                    .map(IntoNameable::into_nameable)
                    .unwrap_or_else(|| dim.size().into_nameable()),
                increments: update_vec,
            },
            cfgs,
            namer,
        );

        let mut reset_label = None;
        if min_size.is_some() {
            let ge_cond = namer.gen_name(ir::Type::I(1));
            self.print_predicate_expr(
                &ge_cond,
                &PredExpr::new_cmp(CmpOp::Ge, ir::Type::I(32), idx, dim.size()),
                namer,
            );
            let label = namer.gen_loop_id();
            self.print_cond_jump(&label.to_string(), &ge_cond);
            reset_label = Some(label);

            // Here we are in the remainder part of the loop; update the predicates.
            for &(inst_id, ref predicates) in fun.loop_predicates(dim.id()) {
                self.update_inst_predicates(fun, inst_id, predicates, namer);
            }

            // TODO: self.print_jump
            self.print_move(ir::Type::I(1), &ge_cond, &true.into_nameable().name(namer));
            self.print_cond_jump(&iter_label.to_string(), &ge_cond);
        }

        if let Some(label) = reset_label {
            self.print_label(&label.to_string());
        }

        // Iteration variables need to be reset after the loop.
        //
        // We don't bundle this in the `Loop` structure because most languages don't have a proper
        // way of handling this.
        for (target, inst) in &reset_vec {
            self.print_int_inst(target, inst, namer);
        }
    }

    /// Prints an unroll loop - loop without jumps
    fn unroll_loop(
        &mut self,
        fun: &Function,
        dim: &Dimension,
        cfgs: &[Cfg],
        namer: &mut NameMap<Self::ValuePrinter>,
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
                        let stepxi =
                            namer.value_printer().get_const_int(&(step * i).into(), 32);
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
        namer: &mut NameMap<Self::ValuePrinter>,
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
        namer: &mut NameMap<Self::ValuePrinter>,
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
            op::Ld {
                t: ld_type,
                operands: [addr],
                access_pattern: pattern,
                predicate,
            } => {
                let predicate = if fun.predicate_accesses() {
                    // TODO: Handle vector
                    predicate.as_ref().map(|p| {
                        (
                            namer
                                .name_instruction_predicate(
                                    inst.id(),
                                    namer.current_indices(),
                                )
                                .to_string(),
                            p.default_value().into_nameable(),
                        )
                    })
                /*
                predicate.as_ref().map(|p| {
                    let expr = predicates_from_range_and_guard(
                        fun.space(),
                        p.ranges(),
                        None,
                    );
                    let name = namer.gen_name(ir::Type::I(1));
                    self.print_predicate_expr(&name, &expr, namer);
                    (name, p.default_value().into_nameable())
                })
                */
                } else {
                    None
                };

                self.print_ld(
                    vector_factors,
                    Self::lower_type(*ld_type, fun),
                    access_pattern_space(pattern, fun.space()),
                    unwrap!(inst.mem_flag()),
                    &Self::name_inst(vector_levels, inst.id(), namer),
                    &Self::name_operand(&[vec![], vec![]], addr, namer),
                    predicate.as_ref().map(|(a, b)| (a as &str, b.name(namer))),
                );
            }
            op::St {
                operands: [addr, val],
                access_pattern: pattern,
                has_side_effects,
                predicate,
            } => {
                let guard = if *has_side_effects {
                    namer.side_effect_guard()
                } else {
                    None
                };

                /*
                let predicate = predicates_from_range_and_guard(
                    fun.space(),
                    predicate
                        .as_ref()
                        .and_then(|p| {
                            if fun.predicate_accesses() {
                                Some(p)
                            } else {
                                None
                            }
                        })
                        .into_iter()
                        .flat_map(|p| p.ranges()),
                    guard.as_ref().map(|s| (s as &str).into_nameable()),
                );

                let guard = if let Some(p) = predicate.to_bool() {
                    assert!(p, "Instruction is always skipped");

                    None
                } else {
                    let predicate_name = namer.gen_name(ir::Type::I(1));
                    self.print_predicate_expr(&predicate_name, &predicate, namer);
                    Some(predicate_name)
                };
                */

                let predicate = if fun.predicate_accesses() {
                    // TODO: Handle vector
                    predicate.as_ref().map(|p| {
                        namer
                            .name_instruction_predicate(
                                inst.id(),
                                namer.current_indices(),
                            )
                            .to_string()
                    })
                } else {
                    None
                };

                self.print_st(
                    vector_factors,
                    Self::lower_type(val.t(), fun),
                    access_pattern_space(pattern, fun.space()),
                    unwrap!(inst.mem_flag()),
                    predicate.as_ref().map(|s| s as &str),
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
            Cast(ref op, from_t, to_t) => unimplemented!("CAST"),
            Add {
                arg_t,
                ref lhs,
                ref rhs,
            } => self.print_add_int(arg_t, result, &lhs.name(namer), &rhs.name(namer)),
            Sub {
                arg_t,
                ref lhs,
                ref rhs,
            } => unimplemented!("generic sub"),
            Div {
                arg_t,
                ref lhs,
                ref rhs,
            } => unimplemented!("generic div"),
            Min {
                arg_t,
                ref lhs,
                ref rhs,
            } => unimplemented!("generic min"),
            Mad {
                arg_t,
                mul_mode,
                ref mlhs,
                ref mrhs,
                ref arhs,
            } => self.print_mad(
                [1, 1],
                arg_t,
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
                arg_t,
                op::Rounding::Exact,
                mul_mode,
                result,
                &lhs.name(namer),
                &rhs.name(namer),
            ),
        }
    }

    fn print_loop(
        &mut self,
        fun: &Function,
        loop_: &Loop<'_>,
        body: &[Cfg],
        namer: &mut NameMap<'_, '_, Self::ValuePrinter>,
    ) {
        // Initialization
        for (target, inst) in &loop_.inits {
            self.print_int_inst(target, inst, namer);
        }

        // Loop label
        self.print_label(&loop_.label.to_string());

        // XXX temp hack
        //for &(inst_id, ref predicates) in fun.loop_predicates(loop_.dim) {
        //self.init_inst_predicates(fun, inst_id, namer);
        //self.update_inst_predicates(fun, inst_id, &predicates[..], namer);
        //}

        // Loop body
        self.cfg_vec(fun, body, namer);

        // Update
        for (target, inst) in &loop_.increments {
            self.print_int_inst(target, inst, namer);
        }

        // Loop condition
        let lt_cond = namer.gen_name(ir::Type::I(1));
        self.print_lt_int(
            ir::Type::I(32),
            &lt_cond,
            &loop_.index.name(namer),
            &loop_.bound.name(namer),
        );
        self.print_cond_jump(&loop_.label.to_string(), &lt_cond);
    }

    fn print_predicate_expr(
        &mut self,
        result: &str,
        pexpr: &PredExpr<'_>,
        namer: &mut NameMap<'_, '_, Self::ValuePrinter>,
    ) {
        match pexpr {
            PredExpr::CmpOp { op, t, lhs, rhs } => {
                let (ir_op, lhs, rhs) = match op {
                    CmpOp::Eq => (ir::BinOp::Equals, lhs, rhs),
                    CmpOp::Ne => unimplemented!("!="),
                    CmpOp::Lt => (ir::BinOp::Lt, lhs, rhs),
                    CmpOp::Gt => (ir::BinOp::Lt, rhs, lhs),
                    CmpOp::Le => (ir::BinOp::Leq, lhs, rhs),
                    CmpOp::Ge => (ir::BinOp::Leq, rhs, lhs),
                };

                self.print_binop(
                    [1, 1],
                    ir_op,
                    *t,
                    ir::op::Rounding::Exact,
                    result,
                    &lhs.name(namer),
                    &rhs.name(namer),
                );
            }
            PredExpr::BoolOp(op, args) => {
                let mut iter = args.iter();
                if let Some(first) = iter.next() {
                    self.print_predicate_expr(result, first, namer);

                    for arg in iter {
                        let tmp = namer.gen_name(ir::Type::I(1));
                        self.print_predicate_expr(&tmp, arg, namer);
                        self.print_bool_op(*op, &result, &result, &tmp);
                    }
                } else {
                    let init_val = match op {
                        BoolOp::And => true.into_nameable(),
                        BoolOp::Or => false.into_nameable(),
                    };

                    self.print_move(ir::Type::I(1), result, &init_val.name(namer));
                }
            }
            PredExpr::Not(arg) => unimplemented!("nope"),
            PredExpr::Variable(var) => {
                self.print_move(ir::Type::I(1), result, &var.name(namer))
            }
        }
    }

    fn print_bool_op(&mut self, op: BoolOp, result: &str, lhs: &str, rhs: &str) {
        let ir_op = match op {
            BoolOp::And => ir::BinOp::And,
            BoolOp::Or => ir::BinOp::Or,
        };

        self.print_binop(
            [1, 1],
            ir_op,
            ir::Type::I(1),
            ir::op::Rounding::Exact,
            result,
            lhs,
            rhs,
        );
    }
}
