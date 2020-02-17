use std::cmp;
use std::collections::hash_map::Entry;
use std::convert::TryFrom;
use std::hash;
use std::sync::Arc;

use fxhash::{FxHashMap, FxHashSet};
use log::{debug, trace};

use crate::ir;
use crate::search_space::{AssertOrd, DimKind, Domain, SearchSpace};

use super::dimension::MergedDimensions;
use super::llir::{self, IntLiteral};
use super::{Function, ParamVal, ParamValKey, Size};

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    // A kernel parameter.  This is a runtime constant.
    Parameter(Arc<ir::Parameter>),
    // Address of a memory block
    MemBlock(ir::MemId, ir::AccessType),
    // A dimension size.  This is a runtime constant.
    Size(Size),
    // A signed constant value.
    Constant(i32),
    // A dimension index.
    Dimension(ir::DimId),
    // A strided sum.  This is `base + sum(exprptr * size)`
    //
    // The components of the strided sum must be sorted (??)
    StridedSum {
        base: ExprPtr,
        elems: Vec<(ExprPtr, Size)>,
    },
    // A rolling sum.  This is initialized with `base` then `+= size * size` at each iteration of
    // `dimid`.
    //
    // The components of the rolling sum must be ordered in nesting order (outermost dimension
    // first), and the same dimension must not appear multiple times.
    //
    // All the dimensions in a rolling sum must be sequential dimensions, i.e. LOOP or UNROLL
    // dimensions.
    RollingSum {
        base: ExprPtr,
        elems: Vec<(ir::DimId, Size)>,
    },
    And(Vec<ExprPtr>),
    InRange {
        base: ExprPtr,
        start: ExprPtr,
        end: ExprPtr,
    },
    Proj(TuplePtr, usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Tuple {
    Delinearize { base: ExprPtr, sizes: Vec<Size> },
}

impl Tuple {
    fn len(&self) -> usize {
        match self {
            Tuple::Delinearize { sizes, .. } => sizes.len(),
        }
    }

    fn collect_host_values(
        &self,
        space: &SearchSpace,
        merged_dimensions: &MergedDimensions<'_>,
        host_values: &mut Vec<ParamVal>,
    ) {
        match self {
            Tuple::Delinearize { base, sizes } => {
                base.collect_host_values(space, merged_dimensions, host_values);

                let mut div_by = Size::from(1);
                for size in sizes.iter().skip(1).rev() {
                    div_by *= size;
                    host_values.extend(ParamVal::from_size(size));
                    host_values.extend(ParamVal::from_div_size(&div_by));
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TuplePtr {
    inner: Arc<Tuple>,
    defined_at: Option<ir::DimId>,
}

impl TuplePtr {
    fn id(&self) -> usize {
        &*self.inner as *const Tuple as usize
    }

    pub fn new(inner: Arc<Tuple>, defined_at: Option<ir::DimId>) -> Self {
        TuplePtr { inner, defined_at }
    }
}

impl cmp::PartialEq for TuplePtr {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl cmp::Eq for TuplePtr {}

impl hash::Hash for TuplePtr {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

impl cmp::Ord for TuplePtr {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.id().cmp(&other.id())
    }
}

impl cmp::PartialOrd for TuplePtr {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::ops::Deref for TuplePtr {
    type Target = Tuple;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub struct ExprId(usize);

impl Expr {
    /// Convert an expression to its constant value.
    ///
    /// # Failure
    ///
    /// Fails when the expression is not constant.
    pub fn try_to_i32(&self) -> Option<i32> {
        match self {
            Expr::Parameter(_)
            | Expr::StridedSum { .. }
            | Expr::RollingSum { .. }
            | Expr::MemBlock(..)
            | Expr::Dimension(_)
            | Expr::And(_)
            | Expr::InRange { .. }
            | Expr::Proj(..) => None,
            &Expr::Constant(c) => Some(c),
            Expr::Size(size) => size.as_int().and_then(|u| i32::try_from(u).ok()),
        }
    }

    /// Returns whether this expression is constant and equal to `0`
    pub fn is_zero(&self) -> bool {
        self.try_to_i32() == Some(0)
    }

    /// Returns  whether this expression is constant and equal to `1`
    pub fn is_one(&self) -> bool {
        self.try_to_i32() == Some(1)
    }

    /// Returns whether this expression is a constant known at runtime.
    ///
    /// A runtime-constant expression is an expression which is independent of the position across
    /// all problem dimension.
    ///
    /// Note that memory block addresses are not runtime constant because they are in shared memory
    /// and need not be at the same position across blocks (?).
    pub fn is_runtime_constant(&self) -> bool {
        match self {
            Expr::Parameter(_) | Expr::Size(_) | Expr::Constant(_) => true,
            Expr::Dimension(_)
            | Expr::RollingSum { .. }
            | Expr::MemBlock(..)
            | Expr::Proj(..) => false,
            Expr::StridedSum { base, elems, .. } => {
                base.is_runtime_constant()
                    && elems.iter().all(|(expr, _)| expr.is_runtime_constant())
            }
            Expr::And(elems) => elems.iter().all(|expr| expr.is_runtime_constant()),
            Expr::InRange { base, start, end } => {
                base.is_runtime_constant()
                    && start.is_runtime_constant()
                    && end.is_runtime_constant()
            }
        }
    }

    fn bitwidth(&self) -> u16 {
        match self {
            Expr::Parameter(param) => param.t.bitwidth().unwrap() as u16,
            Expr::Size(_) => 32,
            Expr::Constant(_) => 32,
            Expr::Dimension(_) => 32,
            Expr::MemBlock(..) => 32,
            Expr::StridedSum { base, .. } => base.bitwidth(),
            Expr::RollingSum { base, .. } => base.bitwidth(),
            // Predicates
            Expr::InRange { .. } => 1,
            Expr::And(_) => 1,
            // Projection
            Expr::Proj(tuple, _) => match &**tuple {
                Tuple::Delinearize { base, .. } => base.bitwidth(),
            },
        }
    }

    fn collect_host_values(
        &self,
        space: &SearchSpace,
        merged_dimensions: &MergedDimensions<'_>,
        host_values: &mut Vec<ParamVal>,
    ) {
        match self {
            Expr::Parameter(param) => host_values.extend(ParamVal::from_operand(
                &crate::ir::Operand::Param(param.clone()),
                space,
            )),

            Expr::MemBlock(..) => (),

            Expr::Size(size) => host_values.extend(ParamVal::from_size(size)),

            Expr::Constant(_) => (),

            Expr::Dimension(_) => (),

            Expr::StridedSum { base, elems } => {
                base.collect_host_values(space, merged_dimensions, host_values);

                for (expr, stride) in elems {
                    expr.collect_host_values(space, merged_dimensions, host_values);
                    host_values.extend(ParamVal::from_size(stride));
                }
            }

            Expr::RollingSum { base, elems } => {
                base.collect_host_values(space, merged_dimensions, host_values);

                for &(dim, ref stride) in elems {
                    assert!(space.innermost(base.defined_at, Some(dim)) == Some(dim));

                    host_values.extend(ParamVal::from_size(stride));

                    let dim_size = merged_dimensions[dim].size();
                    match (dim_size.as_int(), stride.as_int()) {
                        (Some(_), _) => (),
                        (None, Some(_stride)) => {
                            host_values.extend(ParamVal::from_size(dim_size))
                        }
                        (None, None) => {
                            host_values.extend(ParamVal::from_size(&(dim_size * stride)))
                        }
                    }
                }
            }

            Expr::InRange { base, start, end } => {
                base.collect_host_values(space, merged_dimensions, host_values);
                start.collect_host_values(space, merged_dimensions, host_values);
                end.collect_host_values(space, merged_dimensions, host_values);
            }

            Expr::And(elems) => {
                for elem in elems {
                    elem.collect_host_values(space, merged_dimensions, host_values);
                }
            }

            Expr::Proj(tuple, _index) => {
                tuple.collect_host_values(space, merged_dimensions, host_values)
            }
        }
    }

    pub fn host_values(
        &self,
        space: &SearchSpace,
        merged_dimensions: &MergedDimensions<'_>,
    ) -> impl Iterator<Item = ParamVal> {
        let mut host_values = Vec::new();
        self.collect_host_values(space, merged_dimensions, &mut host_values);
        host_values.into_iter()
    }
}

#[derive(Debug, Default, Clone)]
pub struct ExprToOperand<'a> {
    init_exprs: FxHashMap<ir::DimId, Vec<llir::Instruction<'a>>>,
    compute_exprs: FxHashMap<Option<ir::DimId>, Vec<llir::PredicatedInstruction<'a>>>,
    update_exprs: FxHashMap<ir::DimId, Vec<llir::Instruction<'a>>>,
    reset_exprs: FxHashMap<ir::DimId, Vec<llir::Instruction<'a>>>,
    cache: FxHashMap<ExprPtr, llir::Operand<'a>>,
}

impl<'a> ExprToOperand<'a> {
    pub fn init_exprs(&self, dim: ir::DimId) -> &[llir::Instruction<'a>] {
        self.init_exprs
            .get(&dim)
            .map(|x| &x[..])
            .unwrap_or_else(|| &[])
    }

    pub fn compute_exprs(
        &self,
        dim: Option<ir::DimId>,
    ) -> &[llir::PredicatedInstruction<'a>] {
        self.compute_exprs
            .get(&dim)
            .map(|x| &x[..])
            .unwrap_or_else(|| &[])
    }

    pub fn update_exprs(&self, dim: ir::DimId) -> &[llir::Instruction<'a>] {
        self.update_exprs
            .get(&dim)
            .map(|x| &x[..])
            .unwrap_or_else(|| &[])
    }

    pub fn reset_exprs(&self, dim: ir::DimId) -> &[llir::Instruction<'a>] {
        self.reset_exprs
            .get(&dim)
            .map(|x| &x[..])
            .unwrap_or_else(|| &[])
    }

    pub(super) fn add_reset(
        &mut self,
        dim: ir::DimId,
        instruction: llir::Instruction<'a>,
    ) {
        self.reset_exprs
            .entry(dim)
            .or_insert_with(Vec::new)
            .push(instruction);
    }

    pub fn to_operand(&self, expr: &ExprPtr) -> llir::Operand<'a> {
        self.cache[expr].clone()
    }
}

#[derive(Debug, Default)]
pub struct ExprDispatch<'a> {
    init_exprs: FxHashMap<ir::DimId, Vec<llir::Instruction<'a>>>,
    compute_exprs: FxHashMap<Option<ir::DimId>, Vec<llir::PredicatedInstruction<'a>>>,
    update_exprs: FxHashMap<ir::DimId, Vec<llir::Instruction<'a>>>,
    reset_exprs: FxHashMap<ir::DimId, Vec<llir::Instruction<'a>>>,
}

impl<'a> ExprDispatch<'a> {
    fn add_init(&mut self, dim: ir::DimId, inst: llir::Instruction<'a>) {
        trace!("init({}): {}", dim, inst);

        self.init_exprs
            .entry(dim)
            .or_insert_with(Vec::new)
            .push(inst);
    }

    fn add_compute(
        &mut self,
        dim: Option<ir::DimId>,
        inst: impl Into<llir::PredicatedInstruction<'a>>,
    ) {
        let inst = inst.into();

        trace!(
            "compute({}): {} ",
            if let Some(dim) = dim {
                format!("{}", dim)
            } else {
                "None".to_string()
            },
            inst
        );

        self.compute_exprs
            .entry(dim)
            .or_insert_with(Vec::new)
            .push(inst);
    }

    fn add_update(&mut self, dim: ir::DimId, inst: llir::Instruction<'a>) {
        trace!("update({}): {}", dim, inst);

        self.update_exprs
            .entry(dim)
            .or_insert_with(Vec::new)
            .push(inst);
    }

    fn add_reset(&mut self, dim: ir::DimId, inst: llir::Instruction<'a>) {
        trace!("reset({}): {}", dim, inst);

        self.reset_exprs
            .entry(dim)
            .or_insert_with(Vec::new)
            .push(inst);
    }
}

pub struct ExprToOperandBuilder<'a, 'b> {
    function: &'a Function<'a>,
    merged_dimensions: &'a MergedDimensions<'a>,
    namer: &'a mut super::NameMap<'b>,

    dispatch: ExprDispatch<'b>,
    cache: FxHashMap<ExprPtr, llir::Operand<'b>>,
    tuple_cache: FxHashMap<TuplePtr, Vec<llir::Operand<'b>>>,
}

impl<'a, 'b> ExprToOperandBuilder<'a, 'b> {
    pub fn new(
        function: &'a Function<'a>,
        merged_dimensions: &'a MergedDimensions<'a>,
        namer: &'a mut super::NameMap<'b>,
    ) -> Self {
        ExprToOperandBuilder {
            function,
            merged_dimensions,
            namer,

            dispatch: Default::default(),
            cache: Default::default(),
            tuple_cache: Default::default(),
        }
    }

    pub fn finish(self) -> ExprToOperand<'b> {
        ExprToOperand {
            init_exprs: self.dispatch.init_exprs,
            compute_exprs: self.dispatch.compute_exprs,
            update_exprs: self.dispatch.update_exprs,
            reset_exprs: self.dispatch.reset_exprs,
            cache: self.cache,
        }
    }

    fn tuple_to_operand(&mut self, tuple: &TuplePtr, index: usize) -> llir::Operand<'b> {
        match self.tuple_cache.get(tuple) {
            Some(operands) => operands[index].clone(),
            None => {
                let operands = match &**tuple {
                    Tuple::Delinearize { base, sizes } => {
                        let compute_at = base.defined_at;
                        let base = self.to_operand(base);

                        let mut registers = Vec::with_capacity(sizes.len());

                        let mut div_by = Size::from(1);
                        let mut prev = None;
                        // Skip the first (outermost) size: we never need to actually use it in
                        // division/modulos.  We get it back via `prev` at the end.
                        for size in sizes.iter().skip(1).rev() {
                            let mod_by = self.namer.name_size(&size, base.t());
                            let neg_mod_by = self.namer.gen_name(base.t());
                            self.dispatch.add_compute(
                                None,
                                llir::Instruction::neg(neg_mod_by, mod_by).unwrap(),
                            );
                            let neg_mod_by = neg_mod_by.into_operand();

                            div_by *= size;

                            let div_magic = self.namer.name_div_magic(&div_by, base.t());
                            let div_shift = self.namer.name_div_shift(&div_by, base.t());

                            let dst =
                                prev.unwrap_or_else(|| self.namer.gen_name(base.t()));
                            let next = self.namer.gen_name(base.t());
                            self.dispatch.add_compute(
                                compute_at,
                                llir::Instruction::imad_high(
                                    next,
                                    base.clone(),
                                    div_magic,
                                    base.clone(),
                                )
                                .unwrap(),
                            );
                            self.dispatch.add_compute(
                                compute_at,
                                llir::Instruction::shr(
                                    next,
                                    next.into_operand(),
                                    div_shift,
                                )
                                .unwrap(),
                            );
                            self.dispatch.add_compute(
                                compute_at,
                                llir::Instruction::imad_low(
                                    dst,
                                    next.into_operand(),
                                    neg_mod_by,
                                    prev.map(|prev| prev.into_operand())
                                        .unwrap_or_else(|| base.clone()),
                                )
                                .unwrap(),
                            );

                            registers.push(dst);
                            prev = Some(next)
                        }

                        registers.push(prev.unwrap_or_else(|| {
                            let single = self.namer.gen_name(base.t());
                            self.dispatch.add_compute(
                                compute_at,
                                llir::Instruction::mov(single, base.clone()).unwrap(),
                            );
                            single
                        }));

                        // We computed the last projection first so we need to flip it
                        registers.reverse();
                        assert_eq!(registers.len(), sizes.len());

                        /*
                        let p = self.namer.gen_name(ir::Type::I(1));
                        let mut predicate = None;
                        for (size, &cur) in sizes.iter().zip(&registers).skip(1).rev() {
                            let size = self.namer.name_size(&size, base.t());

                            self.dispatch.add_compute(
                                compute_at,
                                llir::Instruction::iadd(
                                    cur,
                                    cur.into_operand(),
                                    1i32.int_literal(),
                                )
                                .unwrap()
                                .predicated(predicate),
                            );

                            self.dispatch.add_compute(
                                compute_at,
                                llir::Instruction::set_ge(p, cur.into_operand(), size)
                                    .unwrap(),
                            );
                            self.dispatch.add_compute(
                                compute_at,
                                llir::Instruction::mov(cur, 0i32.int_literal())
                                    .unwrap()
                                    .predicated(p),
                            );

                            predicate = Some(p);
                        }

                        self.dispatch.add_compute(
                            compute_at,
                            llir::Instruction::iadd(
                                registers[0],
                                registers[0].into_operand(),
                                1i32.int_literal(),
                            )
                            .unwrap()
                            .predicated(predicate),
                        ); */

                        registers
                            .into_iter()
                            .map(llir::Register::into_operand)
                            .collect::<Vec<_>>()
                    }
                };

                let operand = operands[index].clone();
                self.tuple_cache.insert(tuple.clone(), operands);
                operand
            }
        }
    }

    pub fn to_operand(&mut self, expr: &ExprPtr) -> llir::Operand<'b> {
        match self.cache.get(expr) {
            Some(operand) => operand.clone(),
            None => {
                let operand = match &**expr {
                    Expr::Parameter(param) => self
                        .namer
                        .name_param_val(ParamValKey::External(&*param))
                        .into_operand(),

                    &Expr::MemBlock(mem_block, access_type) => {
                        self.namer.name_addr(mem_block, access_type).into_operand()
                    }

                    Expr::Size(size) => {
                        self.namer.name_size(&size, ir::Type::I(expr.bitwidth()))
                    }

                    &Expr::Constant(c) => c.int_literal(),

                    &Expr::Dimension(id) => self.namer.name_index_as_operand(id),

                    Expr::Proj(tuple, index) => self.tuple_to_operand(&*tuple, *index),

                    Expr::InRange { base, start, end } => {
                        let dst = self.namer.gen_name(ir::Type::I(1));
                        let base = self.to_operand(base);
                        let start = self.to_operand(start);
                        let end = self.to_operand(end);

                        self.dispatch.add_compute(
                            expr.defined_at,
                            llir::Instruction::set_ge(dst, base.clone(), start).unwrap(),
                        );
                        self.dispatch.add_compute(
                            expr.defined_at,
                            llir::Instruction::set_lt_and(
                                dst,
                                base,
                                end,
                                dst.into_operand(),
                            )
                            .unwrap(),
                        );

                        dst.into_operand()
                    }

                    Expr::And(preds) => {
                        let dst = self.namer.gen_name(ir::Type::I(1));
                        assert!(preds.len() >= 2);

                        let first = self.to_operand(&preds[0]);
                        let second = self.to_operand(&preds[1]);
                        self.dispatch.add_compute(
                            expr.defined_at,
                            llir::Instruction::and(dst, first, second).unwrap(),
                        );

                        for pred in &preds[2..] {
                            let pred = self.to_operand(pred);
                            self.dispatch.add_compute(
                                expr.defined_at,
                                llir::Instruction::and(dst, dst.into_operand(), pred)
                                    .unwrap(),
                            );
                        }

                        dst.into_operand()
                    }

                    Expr::StridedSum { base, elems } => {
                        assert!(!elems.is_empty());

                        let reg = self.namer.gen_name(ir::Type::I(expr.bitwidth()));

                        let compute_at = expr.defined_at;
                        let mut prev = self.to_operand(base);
                        for (expr, stride) in elems {
                            let expr_op = self.to_operand(expr);

                            self.dispatch.add_compute(
                                compute_at,
                                llir::Instruction::imad(
                                    reg,
                                    expr_op,
                                    self.namer.name_size(stride, ir::Type::I(32)),
                                    prev,
                                )
                                .unwrap(),
                            );

                            prev = reg.into_operand();
                        }

                        prev
                    }

                    Expr::RollingSum { base, elems } => {
                        let reg = self.namer.gen_name(ir::Type::I(expr.bitwidth()));
                        let tmp = self.namer.gen_name(ir::Type::I(32));
                        let base_op = self.to_operand(base);
                        self.dispatch.add_compute(
                            base.defined_at,
                            llir::Instruction::mov(reg, base_op).unwrap(),
                        );

                        for &(dim, ref stride) in elems {
                            self.dispatch.add_update(
                                dim,
                                llir::Instruction::iadd_auto(
                                    reg,
                                    self.namer.name_size(stride, ir::Type::I(32)),
                                    reg.into_operand(),
                                )
                                .unwrap(),
                            );

                            let dim_size = self.merged_dimensions[dim].size();
                            let reset_insts = match (dim_size.as_int(), stride.as_int()) {
                                (Some(dim_size), Some(stride)) => {
                                    vec![llir::Instruction::isub(
                                        reg,
                                        reg.into_operand(),
                                        ((dim_size * stride) as i32)
                                            .typed_int_literal(reg.t())
                                            .unwrap(),
                                    )]
                                }
                                (Some(dim_size), None) => vec![
                                    llir::Instruction::neg(
                                        tmp,
                                        self.namer.name_size(stride, ir::Type::I(32)),
                                    ),
                                    llir::Instruction::imad(
                                        reg,
                                        (dim_size as i32).int_literal(),
                                        tmp.into_operand(),
                                        reg.into_operand(),
                                    ),
                                ],
                                (None, Some(stride)) => vec![
                                    llir::Instruction::neg(
                                        tmp,
                                        self.namer.name_size(dim_size, ir::Type::I(32)),
                                    ),
                                    llir::Instruction::imad(
                                        reg,
                                        (stride as i32).int_literal(),
                                        tmp.into_operand(),
                                        reg.into_operand(),
                                    ),
                                ],
                                (None, None) => vec![llir::Instruction::isub_auto(
                                    reg,
                                    reg.into_operand(),
                                    self.namer
                                        .name_size(&(dim_size * stride), ir::Type::I(32)),
                                )],
                            };

                            for reset_inst in reset_insts {
                                self.dispatch.add_reset(dim, reset_inst.unwrap());
                            }
                        }

                        reg.into_operand()
                    }
                };

                self.cache.insert(expr.clone(), operand.clone());
                operand
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExprPtr {
    expr: Arc<Expr>,
    defined_at: Option<ir::DimId>,
}

impl cmp::PartialEq for ExprPtr {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl cmp::Eq for ExprPtr {}

impl cmp::Ord for ExprPtr {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.id().cmp(&other.id())
    }
}

impl cmp::PartialOrd for ExprPtr {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl hash::Hash for ExprPtr {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state)
    }
}

impl std::ops::Deref for ExprPtr {
    type Target = Expr;

    fn deref(&self) -> &Self::Target {
        &*self.expr
    }
}

impl ExprPtr {
    fn id(&self) -> usize {
        &*self.expr as *const Expr as usize
    }

    pub fn new(expr: Arc<Expr>, defined_at: Option<ir::DimId>) -> Self {
        ExprPtr { expr, defined_at }
    }
}

pub struct ExprBuilder<'a> {
    space: &'a SearchSpace,
    cache: FxHashMap<Arc<Expr>, ExprPtr>,
    tuple_cache: FxHashMap<Arc<Tuple>, TuplePtr>,
}

impl<'a> ExprBuilder<'a> {
    pub fn new(space: &'a SearchSpace) -> Self {
        ExprBuilder {
            space,
            cache: Default::default(),
            tuple_cache: Default::default(),
        }
    }

    fn create_tuple(&mut self, tuple: Tuple) -> TuplePtr {
        match self.tuple_cache.entry(Arc::new(tuple)) {
            Entry::Occupied(occupied) => TuplePtr::clone(occupied.get()),
            Entry::Vacant(vacant) => {
                trace!("Creating tuple {:?}", vacant.key());

                let defined_at = match &**vacant.key() {
                    Tuple::Delinearize { base, .. } => base.defined_at,
                };
                let key = vacant.key().clone();
                TuplePtr::clone(vacant.insert(TuplePtr::new(key, defined_at)))
            }
        }
    }

    pub fn proj(&mut self, tuple: TuplePtr, index: usize) -> ExprPtr {
        self.create(Expr::Proj(tuple, index))
    }

    fn create(&mut self, expr: Expr) -> ExprPtr {
        let space = &self.space;
        match self.cache.entry(Arc::new(expr)) {
            Entry::Occupied(occupied) => ExprPtr::clone(occupied.get()),
            Entry::Vacant(vacant) => {
                trace!("Creating {:?}", vacant.key());

                let defined_at = match &**vacant.key() {
                    Expr::Parameter(_) | Expr::Size(_) | Expr::Constant(_) => None,
                    Expr::MemBlock(_, _) => {
                        // The memory block, depending on load/store access type, can only be used
                        // in the corresponding loop nest, where the double buffering will happen
                        // (if applicable).  This ensures we don't compute slices at an outer level
                        // compared to the double buffering, which would be incorrect.
                        // XXX: TODO
                        None
                    }
                    &Expr::Dimension(id) => {
                        if space
                            .domain()
                            .get_dim_kind(id)
                            .intersects(DimKind::PARALLEL)
                        {
                            None
                        } else {
                            Some(id)
                        }
                    }
                    Expr::StridedSum { base, elems } => {
                        let mut defined_at = base.defined_at;

                        for (expr, _) in elems {
                            defined_at =
                                self.space.innermost(defined_at, expr.defined_at);
                        }

                        defined_at
                    }
                    Expr::RollingSum { base, elems } => {
                        let mut defined_at = base.defined_at;

                        for &(dim, _) in elems {
                            assert_eq!(
                                self.space.innermost(base.defined_at, Some(dim)),
                                Some(dim),
                                "Rolling dimension `{:?}` is outside base `{:?}`",
                                dim,
                                base.defined_at
                            );

                            defined_at = self.space.innermost(defined_at, Some(dim));
                        }

                        defined_at
                    }

                    Expr::And(elems) => {
                        let mut defined_at = None;

                        for elem in elems {
                            defined_at =
                                self.space.innermost(defined_at, elem.defined_at);
                        }

                        defined_at
                    }

                    Expr::InRange { base, start, end } => self.space.innermost(
                        base.defined_at,
                        self.space.innermost(start.defined_at, end.defined_at),
                    ),

                    Expr::Proj(tuple, _) => tuple.defined_at,
                };

                let key = vacant.key().clone();
                ExprPtr::clone(vacant.insert(ExprPtr::new(key, defined_at)))
            }
        }
    }

    fn extract_thread_constant(&self, expr: &ExprPtr) -> Option<ExprPtr> {
        match &**expr {
            Expr::Parameter(_)
            | Expr::Size(_)
            | Expr::Constant(_)
            | Expr::MemBlock(..) => Some(expr.clone()),
            Expr::Dimension(dim) => {
                if self
                    .space
                    .domain()
                    .get_dim_kind(*dim)
                    .intersects(DimKind::BLOCK | DimKind::THREAD)
                {
                    Some(expr.clone())
                } else {
                    None
                }
            }
            Expr::StridedSum { base, elems } => {
                if elems.iter().all(|(expr, _)| self.is_thread_constant(expr))
                    && self.is_thread_constant(base)
                {
                    Some(expr.clone())
                } else {
                    self.extract_thread_constant(base)
                }
            }
            Expr::RollingSum { base, .. } => {
                if self.is_thread_constant(base) {
                    Some(expr.clone())
                } else {
                    self.extract_thread_constant(base)
                }
            }
            Expr::And(..) => None,
            Expr::InRange { .. } => None,
            Expr::Proj(tuple, index) => {
                if self.is_tuple_thread_constant(&**tuple, *index) {
                    Some(expr.clone())
                } else {
                    None
                }
            }
        }
    }

    fn is_thread_constant(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Parameter(_)
            | Expr::Size(_)
            | Expr::Constant(_)
            | Expr::MemBlock(..) => true,
            &Expr::Dimension(dim) => self
                .space
                .domain()
                .get_dim_kind(dim)
                .intersects(DimKind::BLOCK | DimKind::THREAD),
            Expr::StridedSum { base, elems, .. } => {
                self.is_thread_constant(&**base)
                    && elems
                        .iter()
                        .all(|(expr, _)| self.is_thread_constant(&**expr))
            }
            Expr::RollingSum { base, .. } => self.is_thread_constant(&**base),
            Expr::And(elems) => elems.iter().all(|expr| self.is_thread_constant(&**expr)),
            Expr::InRange { base, start, end } => {
                self.is_thread_constant(&**base)
                    && self.is_thread_constant(&**start)
                    && self.is_thread_constant(&**end)
            }
            Expr::Proj(tuple, index) => self.is_tuple_thread_constant(&**tuple, *index),
        }
    }

    fn is_tuple_thread_constant(&self, tuple: &Tuple, _index: usize) -> bool {
        match tuple {
            Tuple::Delinearize { base, .. } => self.is_thread_constant(&**base),
        }
    }

    pub fn parameter(&mut self, param: Arc<ir::Parameter>) -> ExprPtr {
        self.create(Expr::Parameter(param))
    }

    pub fn mem_block(
        &mut self,
        mem_id: ir::MemId,
        access_type: ir::AccessType,
    ) -> ExprPtr {
        self.create(Expr::MemBlock(mem_id, access_type))
    }

    pub fn size(&mut self, size: Size) -> ExprPtr {
        self.create(Expr::Size(size))
    }

    pub fn constant(&mut self, constant: i32) -> ExprPtr {
        self.create(Expr::Constant(constant))
    }

    pub fn dimension(&mut self, dim: ir::DimId) -> ExprPtr {
        if self
            .space
            .domain()
            .get_dim_kind(dim)
            .intersects(DimKind::VECTOR)
        {
            self.size(0u32.into())
        } else {
            self.create(Expr::Dimension(dim))
        }
    }

    pub fn in_range(&mut self, base: ExprPtr, start: ExprPtr, end: ExprPtr) -> ExprPtr {
        self.create(Expr::InRange { base, start, end })
    }

    pub fn delinearize(&mut self, base: ExprPtr, sizes: Vec<Size>) -> TuplePtr {
        self.create_tuple(Tuple::Delinearize { base, sizes })
    }

    pub fn rolling_sum<II>(&mut self, base: ExprPtr, elems: II) -> ExprPtr
    where
        II: IntoIterator<Item = (ir::DimId, Size)>,
    {
        // We need to split the dimensions into those which can actually be used in a rolling sum,
        // and those which can't.
        //
        // Only sequential dimensions (LOOP and UNROLL) can be used in a rolling sum; others must
        // be computed at the beginning of the code.  In addition, if the base of the sum is not a
        // thread constant, any dimension which is nested inside the dimension where the base is
        // defined can't be iterated either.
        let mut outer = Vec::new();
        let mut rolling = Vec::new();

        let base_compute_at = base.defined_at.map(|dim| self.space.nesting_order(dim));
        for (dim, stride) in elems.into_iter() {
            if self
                .space
                .domain()
                .get_dim_kind(dim)
                .intersects(DimKind::VECTOR)
            {
                // VECTOR dimensions are always `0` in computations; the broadcasting is handled at
                // a lower level -- we skip them here.
            } else if self
                .space
                .domain()
                .get_dim_kind(dim)
                .intersects(DimKind::PARALLEL)
                || Some(self.space.nesting_order(dim)) <= base_compute_at
            {
                outer.push((self.dimension(dim), stride));
            } else {
                rolling.push((dim, stride));
            }
        }

        enum Part {
            Strided(Vec<(ExprPtr, Size)>),
            Rolling(Vec<(ir::DimId, Size)>),
        }

        let mut parts = Vec::new();

        // Sort by outermost first
        rolling.sort_by_key(|&(dim, _)| AssertOrd(self.space.nesting_order(dim)));
        for (dim, stride) in rolling {
            if self.space.domain().get_dim_kind(dim) == DimKind::UNROLL {
                match parts.last_mut() {
                    Some(Part::Strided(strided)) => {
                        strided.push((self.dimension(dim), stride))
                    }
                    _ => parts.push(Part::Strided(vec![(self.dimension(dim), stride)])),
                }
            } else {
                match parts.last_mut() {
                    Some(Part::Rolling(rolling)) => rolling.push((dim, stride)),
                    _ => parts.push(Part::Rolling(vec![(dim, stride)])),
                }
            }
        }

        let mut base = base;

        if !outer.is_empty() {
            base = self.strided_sum(base, outer);
        }

        for part in parts {
            match part {
                Part::Strided(strided) => base = self.strided_sum(base, strided),
                Part::Rolling(mut rolling) => {
                    // This gets added by innermost first so we need to reverse it again here.
                    rolling.reverse();
                    base = self.create(Expr::RollingSum {
                        base,
                        elems: rolling,
                    })
                }
            }
        }

        base
    }

    pub fn strided_sum<II>(&mut self, base: ExprPtr, elems: II) -> ExprPtr
    where
        II: IntoIterator<Item = (ExprPtr, Size)>,
    {
        // We split between strided and rolling parts.Arc
        //
        // Rolling part are dimensions and inner rolling sums that include at least one dimension
        // inside the dimension where the base is defined at.
        let mut outer = Vec::new();
        let mut strided = Vec::new();

        enum Elem {
            Rolling(ir::DimId, Size),
            Strided(ExprPtr, Size),
        }

        let base_compute_at = base.defined_at.map(|dim| self.space.nesting_order(dim));
        for (expr, stride) in elems {
            if expr.is_zero() {
                continue;
            }

            if expr.defined_at.map(|dim| self.space.nesting_order(dim)) <= base_compute_at
            {
                outer.push((expr, stride));
            } else {
                match &*expr {
                    Expr::RollingSum { base, elems } => {
                        if elems.iter().any(|(dim, _)| {
                            Some(self.space.nesting_order(*dim)) > base_compute_at
                        }) {
                            for (dim, dim_stride) in elems {
                                assert_ne!(
                                    self.space.domain().get_dim_kind(*dim),
                                    DimKind::UNROLL
                                );

                                strided.push(Elem::Rolling(*dim, dim_stride * &stride));
                            }

                            if !base.is_zero() {
                                strided.push(Elem::Strided(base.clone(), stride));
                            }
                        } else {
                            strided.push(Elem::Strided(expr, stride))
                        }
                    }
                    Expr::Dimension(dim) => strided.push(Elem::Rolling(*dim, stride)),
                    _ => strided.push(Elem::Strided(expr, stride)),
                }
            }
        }

        let mut base = base;

        if !outer.is_empty() {
            outer.sort_by_key(|(expr, _)| {
                expr.defined_at
                    .map(|dim| AssertOrd(self.space.nesting_order(dim)))
            });

            base = self.create(Expr::StridedSum { base, elems: outer });
        }

        if !strided.is_empty() {
            // Sort by outermost first
            strided.sort_by_key(|elem| match elem {
                Elem::Rolling(dim, _) => Some(AssertOrd(self.space.nesting_order(*dim))),
                Elem::Strided(expr, _) => expr
                    .defined_at
                    .map(|dim| AssertOrd(self.space.nesting_order(dim))),
            });

            enum Part {
                Strided(Vec<(ExprPtr, Size)>, bool),
                Rolling(Vec<(ir::DimId, Size)>),
            }

            let mut current_dim = base.defined_at;
            let mut parts = Vec::new();
            for elem in strided {
                match elem {
                    Elem::Strided(expr, stride) => match parts.last_mut() {
                        Some(Part::Strided(strided, false))
                            if current_dim == expr.defined_at =>
                        {
                            strided.push((expr, stride));
                        }
                        _ => {
                            current_dim = expr.defined_at;
                            parts.push(Part::Strided(vec![(expr, stride)], false));
                        }
                    },
                    Elem::Rolling(dim, stride) => {
                        if self.space.domain().get_dim_kind(dim) == DimKind::UNROLL {
                            match parts.last_mut() {
                                Some(Part::Strided(strided, true)) => {
                                    strided.push((self.dimension(dim), stride))
                                }
                                _ => {
                                    current_dim = Some(dim);
                                    parts.push(Part::Strided(
                                        vec![(self.dimension(dim), stride)],
                                        true,
                                    ));
                                }
                            }
                        } else {
                            match parts.last_mut() {
                                Some(Part::Rolling(rolling)) => {
                                    rolling.push((dim, stride))
                                }
                                _ => parts.push(Part::Rolling(vec![(dim, stride)])),
                            }
                        }
                    }
                }
            }

            for part in parts {
                match part {
                    Part::Strided(strided, _unroll) => {
                        base = self.create(Expr::StridedSum {
                            base,
                            elems: strided,
                        });
                    }
                    Part::Rolling(mut rolling) => {
                        // This gets added by innermost first so we need to reverse it again here.
                        rolling.reverse();
                        base = self.create(Expr::RollingSum {
                            base,
                            elems: rolling,
                        });
                    }
                }
            }
        }

        base
    }

    pub fn and<II>(&mut self, elems: II) -> ExprPtr
    where
        II: IntoIterator<Item = ExprPtr>,
    {
        let mut elems = elems.into_iter().collect::<Vec<_>>();
        elems.sort_by_key(|expr| {
            expr.defined_at
                .map(|dim| AssertOrd(self.space.nesting_order(dim)))
        });

        let mut current_dim = None;
        let mut parts: Vec<Vec<_>> = Vec::new();
        for expr in elems {
            if current_dim == expr.defined_at {
                if let Some(part) = parts.last_mut() {
                    part.push(expr);
                    continue;
                }
            }

            current_dim = expr.defined_at;
            parts.push(vec![expr]);
        }

        let mut base = None;
        for mut part in parts {
            if let Some(base) = base.take() {
                part.push(base);
            }

            if part.len() <= 1 {
                base = part.into_iter().next();
            } else {
                base = Some(self.create(Expr::And(part)));
            }
        }

        base.unwrap()
    }
}

pub struct ExprLowering<'a> {
    builder: ExprBuilder<'a>,
    space: &'a SearchSpace,
    merged_dims: &'a MergedDimensions<'a>,
    device_code_args: &'a mut FxHashSet<ParamVal>,
    logical_dim_cache: FxHashMap<ir::LogicalDimId, ExprPtr>,
    parameter_cache: FxHashMap<&'a Arc<ir::Parameter>, ExprPtr>,
}

impl<'a> ExprLowering<'a> {
    pub fn new(
        space: &'a SearchSpace,
        merged_dims: &'a MergedDimensions<'a>,
        device_code_args: &'a mut FxHashSet<ParamVal>,
    ) -> Self {
        ExprLowering {
            builder: ExprBuilder::new(space),
            space,
            merged_dims,
            device_code_args,
            // Caches
            logical_dim_cache: Default::default(),
            parameter_cache: Default::default(),
        }
    }

    /// Returns a pair (expr, thread_constant_expr) of the actual value for the variable and the
    /// thread-constant base.  The thread-constant base is flipped when double-buffering is
    /// enabled.
    pub fn induction_var(
        &mut self,
        ind_var: &'a ir::InductionVar,
    ) -> (ExprPtr, Option<ExprPtr>) {
        let base = match ind_var.base() {
            ir::Operand::Param(param) => self.parameter(param),
            &ir::Operand::Addr(mem_id, access_type) => {
                self.builder.mem_block(mem_id, access_type)
            }
            _ => panic!("Unsupported operand for induction variable base"),
        };

        let merged_dims = &self.merged_dims;
        let space = &self.space;
        let ptr = self.builder.rolling_sum(
            base,
            ind_var.dims().iter().map(|&(dim, ref stride)| {
                (merged_dims[dim].id(), Size::from_ir(stride, space))
            }),
        );

        // Determine the dimension where we can compute the induction var.
        //
        // This is needed for the double-buffering because if there are outer loops etc. we need to
        // reset the buffers at that point.
        let mut dims = ind_var
            .dims()
            .iter()
            .map(|&(dim, _)| merged_dims[dim].id())
            .collect::<Vec<_>>();
        // dims.retain(|&dim| space.domain().get_dim_kind(dim).intersects(DimKind::THREAD));
        dims.sort_by_key(|&dim| AssertOrd(space.nesting_order(dim)));
        let compute_dim = if let Some(&dim) = dims.first() {
            let nesting_order = space.nesting_order(dim);
            // Only keep dims that are outermost to the thread dim
            let mut outer_dims = space
                .ir_instance()
                .dims()
                .filter(|other_dim| {
                    merged_dims[other_dim.id()]
                        .kind()
                        .intersects(DimKind::SEQUENTIAL)
                })
                .filter(|other_dim| nesting_order > other_dim.id())
                .collect::<Vec<_>>();
            // Sort by outermost first
            outer_dims.sort_by_key(|dim| AssertOrd(space.nesting_order(dim.id())));
            debug!("outer_dims: {:?}", outer_dims);
            outer_dims.last().map(|dim| merged_dims[dim.id()].id())
        } else {
            None
        };

        debug!(
            "induction var parallel dims: {:?} (compute at: {:?})",
            dims, compute_dim
        );

        let thread_constant = self.builder.extract_thread_constant(&ptr);
        (ptr, thread_constant)
    }

    pub fn logical_dim(&mut self, lid: ir::LogicalDimId) -> ExprPtr {
        match self.logical_dim_cache.entry(lid) {
            Entry::Occupied(occupied) => ExprPtr::clone(occupied.get()),
            Entry::Vacant(vacant) => {
                let mut rolling = Vec::new();
                let mut stride = Size::from(1u32);
                for dim in self.space.ir_instance().logical_dim(lid).dimensions() {
                    assert!(stride.as_int().is_some());

                    let dim = &self.merged_dims[dim];
                    rolling.push((dim.id(), stride.clone()));

                    stride *= dim.size();
                }

                let base = self.builder.size(0u32.into());
                vacant
                    .insert(self.builder.rolling_sum(base, rolling))
                    .clone()
            }
        }
    }

    pub fn parameter(&mut self, p: &'a Arc<ir::Parameter>) -> ExprPtr {
        match self.parameter_cache.entry(p) {
            Entry::Occupied(occupied) => ExprPtr::clone(occupied.get()),
            Entry::Vacant(vacant) => {
                self.device_code_args.insert(ParamVal::External(
                    p.clone(),
                    self.space
                        .ir_instance()
                        .device()
                        .lower_type(p.t, self.space)
                        .unwrap(),
                ));

                vacant.insert(self.builder.parameter(Arc::clone(p))).clone()
            }
        }
    }

    pub fn expr(&mut self, expr: &'a ir::IndexExpr) -> ExprPtr {
        match *expr {
            ir::IndexExpr::LogicalDim(lid) => self.logical_dim(lid),
            ir::IndexExpr::Parameter(ref p) => self.parameter(p),
            ir::IndexExpr::Size(ref size) => self.builder.size(size.into()),
            ir::IndexExpr::Sum(cst, ref args) => {
                let args = args
                    .iter()
                    .map(|expr| (self.expr(expr), 1u32.into()))
                    .collect::<Vec<_>>();
                let base = self.builder.constant(cst);
                self.builder.strided_sum(base, args)
            }
            ir::IndexExpr::Proj(ref tuple, idx) => {
                let tuple = self.tuple(tuple);
                self.builder.proj(tuple, idx)
            }
        }
    }

    pub fn tuple(&mut self, tuple: &'a ir::TupleExpr) -> TuplePtr {
        match tuple {
            ir::TupleExpr::Delinearize(expr, sizes) => {
                let expr = self.expr(expr);
                let sizes = sizes.iter().map(Size::from).collect::<Vec<_>>();
                self.builder.delinearize(expr, sizes)
            }
        }
    }

    pub fn predicate(&mut self, predicate: &'a ir::IndexPredicate) -> ExprPtr {
        match predicate {
            ir::IndexPredicate::InRange(expr, range) => {
                let expr = self.expr(expr);
                let start = self.expr(&range.start);
                let end = self.expr(&range.end);
                self.builder.in_range(expr, start, end)
            }
            ir::IndexPredicate::And(preds) => {
                let preds = preds
                    .iter()
                    .map(|pred| self.predicate(pred))
                    .collect::<Vec<_>>();
                self.builder.and(preds)
            }
        }
    }

    pub fn access(&mut self, access: &'a ir::Access) -> (ExprPtr, Option<ExprPtr>) {
        let len_bytes = access.base().elem_t.unwrap().len_byte().unwrap();
        let base = self.builder.parameter(Arc::clone(access.base()));
        let elems = access
            .strides()
            .iter()
            .map(|(expr, stride)| {
                (
                    self.expr(expr),
                    Size::from_ir(&ir::PartialSize::from(stride), self.space) * len_bytes,
                )
            })
            .collect::<Vec<_>>();
        let predicate = access.predicate().map(|p| self.predicate(p));
        (self.builder.strided_sum(base, elems), predicate)
    }
}
