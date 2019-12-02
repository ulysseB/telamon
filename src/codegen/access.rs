use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use fxhash::{FxHashMap, FxHashSet};
use itertools::Itertools;

use crate::ir;
use crate::search_space::{AssertOrd, DimKind, Domain, SearchSpace};

use super::dimension::MergedDimensions;
use super::{ParamVal, Size};

/// An induction variable which gets updated in lockstep with loops.
///
/// TODO: Should we use an integer stride instead?  The stride must always be integer anyways.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IterationVar {
    outer_dims: Vec<(ir::DimId, Size)>,
    loop_dims: Vec<(ir::DimId, Size)>,
}

impl IterationVar {
    /// The type of the iteration variable.
    ///
    /// Implementation note: currently all iteration variable are signed 32 bit values.  The code
    /// depends on this assumption in various places.
    pub fn t(&self) -> ir::Type {
        ir::Type::I(32)
    }

    /// Outer dimensions of the iteration variable along with the corresponding stride.
    ///
    /// Outer dimensions are parallel dimensions (typically thread or block dimensions) upon which
    /// the teration variable depends.
    ///
    /// Implementation note: currently all strides are actually integers.
    pub fn outer_dims(&self) -> &[(ir::DimId, Size)] {
        &self.outer_dims
    }

    /// Loop dimensions of the iteration variable along with the corresponding stride.
    ///
    /// Loop dimensions are non-instanciated sequential dimensions upon which the variable depends.
    /// In particular this include plain loop dimensions but excludes unrolled and vectorized
    /// dimensions.
    ///
    /// Implementation note: currently all strides are actually integers.
    pub fn loop_dims(&self) -> &[(ir::DimId, Size)] {
        &self.loop_dims
    }
}

/// Shared pointer to an iteration variable.
pub type IterationVarPtr = Arc<IterationVar>;

/// An instantiation of an induction variable
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Instantiation {
    induction_var: IterationVarPtr,
    instantiation_dims: Vec<(ir::DimId, u32, Size)>,
}

impl Instantiation {
    /// The iteration variable which is instanciated.
    pub fn iteration_var(&self) -> &IterationVarPtr {
        &self.induction_var
    }

    /// Iterator over the instantiating dimensions.
    ///
    /// For each tuple `(dim, size, stride)` it holdes that the dimension `dim` is instantiated
    /// (that is, either unrolled or vectorized) with size `size`, and contributes with stride
    /// `stride` to final value.
    pub fn strides(&self) -> impl Iterator<Item = &(ir::DimId, u32, Size)> {
        self.instantiation_dims.iter()
    }

    /// Instantiation dimensions.
    ///
    /// This maps instantiation dimension to their size, i.e. the number of times the variable
    /// should be repeated.
    pub fn instantiation_dims(&self) -> FxHashMap<ir::DimId, usize> {
        self.instantiation_dims
            .iter()
            .map(|&(id, size, ref _stride)| (id, size as usize))
            .collect()
    }
}

/// Shared pointer to an instantiation.
pub type InstantiationPtr = Arc<Instantiation>;

// START

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnpackStep {
    pub div_by: Size,
    pub mod_by: Size,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pack {
    expr: ExprPtr,
    steps: Vec<UnpackStep>,
}

impl Pack {
    pub fn expr(&self) -> &ExprPtr {
        &self.expr
    }

    pub fn instantiation_dims(&self) -> FxHashMap<ir::DimId, usize> {
        self.expr.instantiation_dims()
    }

    pub fn steps(&self) -> &[UnpackStep] {
        &self.steps
    }
}

pub type PackPtr = Arc<Pack>;

#[derive(Debug, PartialEq, Eq, Hash)]
struct PackKey<'a> {
    expr: ExprPtr,
    sizes: &'a [ir::Size],
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum MulSpec {
    Low,
    High,
    Wide,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    // A kernel parameter.  This is a runtime constant.
    Parameter(Arc<ir::Parameter>),
    // A dimension size.  This is a runtime constant.
    Size(Size),
    // Position across a dimension.
    Dimension(ir::DimId),
    // Unpacking of a packed dimension.  This is conceptually similar to a tuple projection.
    Unpack(PackPtr, usize),
    // A strided sum.  This is `base + sum(exprptr * size)`
    StridedSum {
        base: ExprPtr,
        elems: Vec<(ExprPtr, Size, MulSpec)>,
    },
    // A rolling sum.  This is initialized with `base` then `+= size * size` at each iteration of
    // `dimid`.
    RollingSum {
        base: Option<ExprPtr>,
        elems: Vec<(ir::DimId, Vec<Size>)>,
        wide: bool,
    },
}

impl Expr {
    fn is_global_constant(&self) -> bool {
        match self {
            Expr::Parameter(_) | Expr::Size(_) => true,
            Expr::Dimension(_) | Expr::RollingSum { .. } => false,
            Expr::Unpack(pack, _) => pack.expr.is_global_constant(),
            Expr::StridedSum { base, elems, .. } => {
                base.is_global_constant()
                    && elems.iter().all(|(expr, _)| expr.is_global_constant())
            }
        }
    }

    fn bitwidth(&self) -> u32 {
        match self {
            Expr::Parameter(param) => param.t.bitwidth().unwrap(),
            Expr::Size(_) => 32,
            Expr::Dimension(_) => 32,
            Expr::Unpack(_, _) => 32,
            Expr::StridedSum { base, wide, .. } => {
                if *wide {
                    match base.bitwidth() {
                        32 => 64,
                        _ => panic!("Invalid wide"),
                    }
                } else {
                    base.bitwidth()
                }
            }
        }
    }
}

pub struct ExprBuilder<'a> {
    space: &'a SearchSpace,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ExprPtr {
    expr: Arc<Expr>,
    defined_at: Option<ir::DimId>,
}

impl std::ops::Deref for ExprPtr {
    type Target = Expr;

    fn deref(&self) -> &Self::Target {
        &*self.expr
    }
}

impl ExprPtr {
    pub fn new(expr: Expr, defined_at: Option<ir::DimId>) -> Self {
        ExprPtr {
            expr: Arc::new(expr),
            defined_at,
        }
    }
}

impl<'a> ExprBuilder<'a> {
    fn is_thread_constant(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Parameter(_) => true,
            Expr::Size(_) => true,
            &Expr::Dimension(dim) => !self
                .space
                .domain()
                .get_dim_kind(dim)
                .intersects(DimKind::BLOCK | DimKind::THREAD),
            Expr::Unpack(pack, _) => self.is_thread_constant(&pack.expr),
            Expr::StridedSum { base, elems, .. } => {
                self.is_thread_constant(&**base)
                    && elems
                        .iter()
                        .all(|(expr, _)| self.is_thread_constant(&**expr))
            }
        }
    }

    pub fn parameter(param: Arc<ir::Parameter>) -> ExprPtr {
        ExprPtr::new(Expr::Parameter(param), None)
    }

    pub fn size(&self, size: Size) -> ExprPtr {
        ExprPtr::new(Expr::Size(size), None)
    }

    pub fn dimension(&self, dim: ir::DimId) -> ExprPtr {
        ExprPtr::new(
            Expr::Dimension(dim),
            if self
                .space
                .domain()
                .get_dim_kind(dim)
                .intersects(DimKind::PARALLEL)
            {
                None
            } else {
                Some(dim)
            },
        )
    }

    pub fn unpack(&self, pack: PackPtr, index: usize) -> ExprPtr {
        ExprPtr::new(Expr::Unpack(pack, index), pack.expr.defined_at)
    }

    pub fn sum(
        &self,
        mut base: Option<ExprPtr>,
        elems: impl IntoIterator<Item = (ExprPtr, Size)>,
        wide: bool,
    ) -> ExprPtr {
        // We split the sum into parts:
        //
        //  - A constant component, computed at the initialization and which could be offloaded to
        //    the host CPU
        //  - A thread-constant component, computed at the initialization
        //  - Strided components which can be computed at each iteration of sequential dimensions
        //  - Rolling components which gets updated at each iteration of sequential dimensions
        //
        // Note that there should be no component for vector dimensions

        let (gconstant, gvariables): (Vec<_>, Vec<_>) = elems
            .into_iter()
            .partition(|(expr, _)| expr.is_global_constant());
        let (tconstant, tvariables): (Vec<_>, Vec<_>) = gvariables
            .into_iter()
            .partition(|(expr, _)| self.is_thread_constant(&**expr));

        let mut elems = Vec::new();

        if !gconstant.is_empty()
            && base.map(|expr| expr.is_global_constant()).unwrap_or(true)
        {
            base = Some(ExprPtr::new(
                Expr::StridedSum {
                    base,
                    elems: gconstant,
                    wide,
                },
                None,
            ));
        } else {
            elems.extend(gconstant);
        }

        if !tconstant.is_empty()
            && base
                .as_ref()
                .map(|expr| self.is_thread_constant(expr))
                .unwrap_or(true)
        {
            base = Some(ExprPtr::new(
                Expr::StridedSum {
                    base,
                    elems: tconstant,
                    wide,
                },
                None,
            ));
        } else {
            elems.extend(tconstant);
        }

        elems.extend(tvariables);
        // TODO: first check that nothing is outside the base.
        // TODO: then sort the remaining according to outermost first.
        // TODO: then iterate.  alternate strided/rolling etc.

        if !elems.is_empty() {
            let (mut rolling, mut strided) = (Vec::new(), Vec::new());

            for (expr, stride) in elems {
                match &*expr {
                    &Expr::Dimension(dim) if expr.defined_at.is_some() => {
                        rolling.push((dim, vec![stride]))
                    }
                    Expr::RollingSum {
                        base,
                        wide: false,
                        elems,
                    } => {
                        assert!(expr.defined_at.is_some());

                        if let Some(base) = base {
                            strided.push((ExprPtr::clone(base), stride));
                        }

                        rolling.extend(elems.iter().map(|&(dim, ref strides)| {
                            (
                                dim,
                                strides
                                    .iter()
                                    .cloned()
                                    .chain(std::iter::once(stride))
                                    .collect(),
                            )
                        }));
                    }
                    _ => strided.push((expr, stride)),
                }
            }

            // Sort strided by outermost definition dimension first.
            // NB: Asumes that `base` is outermost of all strides.
            strided.sort_by_key(|(expr, _)| {
                expr.defined_at
                    .map(|dim| AssertOrd(self.space.nesting_order(dim)))
            });

            for (dim, group) in &strided.into_iter().group_by(|(expr, _)| expr.defined_at)
            {
                base = Some(ExprPtr::new(
                    Expr::StridedSum {
                        base,
                        elems: group.into_iter().collect(),
                        wide,
                    },
                    dim,
                ));
            }

            // Now remain the rolling
            if !rolling.is_empty() {
                let mut defined_at =
                    base.as_ref().map(|expr| expr.defined_at).unwrap_or(None);
                for &(dim, _) in &rolling {
                    if Some(self.space.nesting_order(dim))
                        > defined_at.map(|dim| self.space.nesting_order(dim))
                    {
                        defined_at = Some(dim);
                    }
                }

                base = Some(ExprPtr::new(
                    Expr::RollingSum {
                        base,
                        elems: rolling,
                        wide,
                    },
                    defined_at,
                ));
            }
        }

        base.unwrap_or_else(|| panic!("Empty sum"))
    }
}

pub enum Def {
    Pack(PackPtr),
    Expr(ExprPtr),
}

impl From<PackPtr> for Def {
    fn from(pack: PackPtr) -> Self {
        Def::Pack(pack)
    }
}

impl From<ExprPtr> for Def {
    fn from(expr: ExprPtr) -> Self {
        Def::Expr(expr)
    }
}

// Structure to keep track of where each iteration variable and expression should be defined
pub struct ExprVarDef<'a> {
    space: &'a SearchSpace,

    // Expressions defined at the global level
    outer_def: Vec<Def>,

    // Variables defined at a loop
    loop_defs: FxHashMap<ir::DimId, Vec<Def>>,

    // Induction variables defined at the global level
    outer_init: Vec<IterationVarPtr>,

    // Where induction var are initialized
    loop_inits: FxHashMap<ir::DimId, Vec<IterationVarPtr>>,

    // Induction variables updated at each iteration of a loop
    loop_updates: FxHashMap<ir::DimId, Vec<(IterationVarPtr, Size)>>,

    // Cache for the definition location of expressions
    expr_cache: FxHashMap<ExprPtr, Option<ir::DimId>>,

    // Cache for the definition location of packed dimensions
    pack_cache: FxHashMap<PackPtr, Option<ir::DimId>>,

    // Cache for the definition location of iteration variables
    induction_cache: FxHashMap<IterationVarPtr, Option<ir::DimId>>,
}

impl<'a> ExprVarDef<'a> {
    pub fn new(space: &'a SearchSpace) -> Self {
        ExprVarDef {
            space,
            outer_def: Default::default(),
            loop_defs: Default::default(),
            outer_init: Default::default(),
            loop_inits: Default::default(),
            loop_updates: Default::default(),
            expr_cache: Default::default(),
            pack_cache: Default::default(),
            induction_cache: Default::default(),
        }
    }

    pub fn exprs(&self) -> impl Iterator<Item = &ExprPtr> {
        self.expr_cache.keys()
    }

    pub fn packs(&self) -> impl Iterator<Item = &PackPtr> {
        self.pack_cache.keys()
    }

    pub fn induction_vars(&self) -> impl Iterator<Item = &IterationVarPtr> {
        self.induction_cache.keys()
    }

    // Returns the iteration variables to initialize at the beginning of the given dimension.
    //
    // If no dimension is provided, returns the iteration variables to initialize at the beginning
    // of the code.
    pub fn inits_at(
        &self,
        id: Option<ir::DimId>,
    ) -> impl Iterator<Item = &IterationVarPtr> {
        match id {
            None => self.outer_init.iter(),
            Some(id) => self
                .loop_inits
                .get(&id)
                .map(|x| x as &[_])
                .unwrap_or_else(|| &[])
                .iter(),
        }
    }

    /// Returns the definition to process at each iteration of a given dimension.
    ///
    /// If no dimension is provided, returns the definitions to process at the beginning of the
    /// code.
    pub fn defs_at(&self, id: Option<ir::DimId>) -> impl Iterator<Item = &Def> {
        match id {
            None => self.outer_def.iter(),
            Some(id) => self
                .loop_defs
                .get(&id)
                .map(|x| x as &[_])
                .unwrap_or_else(|| &[])
                .iter(),
        }
    }

    /// Returns the iteration variables to update after each iteration of the given (sequential)
    /// dimension.
    pub fn updates_at(
        &self,
        id: ir::DimId,
    ) -> impl Iterator<Item = (&IterationVarPtr, &Size)> {
        self.loop_updates
            .get(&id)
            .map(|x| x as &[_])
            .unwrap_or_else(|| &[])
            .iter()
            .map(|&(ref ivar, ref size)| (ivar, size))
    }

    /// Process the given expression and determines where each of its components need to be
    /// computed.
    pub fn expr(&mut self, expr: &ExprPtr) {
        self.process_expr(expr);
    }

    fn process_expr(&mut self, expr: &ExprPtr) -> Option<ir::DimId> {
        if let Some(&level) = self.expr_cache.get(expr) {
            level
        } else {
            let level = match **expr {
                Expr::Instantiation(ref inst) => {
                    let level = self.process_induction_var(&inst.induction_var);
                    // TODO: We might want to def at the instruction level instead?
                    // (= the innermost instantiation dim)
                    self.add_def_at(expr.clone(), level)
                }
                Expr::Parameter(_) => self.add_def_at(expr.clone(), None),
                Expr::Size(_) => self.add_def_at(expr.clone(), None),
                Expr::Sum(_, ref exprs) | Expr::Product(_, ref exprs) => {
                    // TODO: We could split the sum if we wanted
                    // TODO: also might want to put the sum near the instruction sometimes to avoid
                    // cross-product explosion?
                    let space = &*self.space;
                    let level = exprs
                        .iter()
                        .filter_map(|expr| self.process_expr(expr))
                        .fold(None, {
                            move |lhs, rhs| {
                                if let Some(lhs) = lhs {
                                    // Keep the innermost
                                    if space.nesting_order(lhs) < rhs {
                                        Some(rhs)
                                    } else {
                                        Some(lhs)
                                    }
                                } else {
                                    Some(rhs)
                                }
                            }
                        });

                    self.add_def_at(expr.clone(), level)
                }
                Expr::Unpack(ref pack, _idx) => self.process_pack(pack),
            };
            self.expr_cache.insert(expr.clone(), level);
            level
        }
    }

    fn add_def_at<T: Into<Def>>(
        &mut self,
        def: T,
        level: Option<ir::DimId>,
    ) -> Option<ir::DimId> {
        if let Some(level) = level {
            self.loop_defs
                .entry(level)
                .or_insert_with(Vec::new)
                .push(def.into());
        } else {
            self.outer_def.push(def.into());
        }

        level
    }

    fn process_pack(&mut self, pack: &PackPtr) -> Option<ir::DimId> {
        if let Some(&level) = self.pack_cache.get(pack) {
            level
        } else {
            // Define the pack at the same level where the underlying expr is fully defined.
            let level = self.process_expr(&pack.expr);
            self.add_def_at(pack.clone(), level)
        }
    }

    fn process_induction_var(&mut self, indvar: &IterationVarPtr) -> Option<ir::DimId> {
        if let Some(&level) = self.induction_cache.get(indvar) {
            level
        } else {
            for &(loop_dim, ref stride) in &indvar.loop_dims {
                self.loop_updates
                    .entry(loop_dim)
                    .or_insert_with(Vec::new)
                    .push((indvar.clone(), stride.clone()));
            }

            if let Some(&(outer_dim, _)) = &indvar.loop_dims.first() {
                self.loop_inits
                    .entry(outer_dim)
                    .or_insert_with(Vec::new)
                    .push(indvar.clone());
            } else {
                self.outer_init.push(indvar.clone());
            }

            let level = indvar.loop_dims.last().map(|&(dim, _)| dim);
            self.induction_cache.insert(indvar.clone(), level);
            level
        }
    }
}

pub struct ExprContext<'a> {
    expr_cache: FxHashMap<ExprKey<'a>, ExprPtr>,
    pack_cache: FxHashMap<PackKey<'a>, PackPtr>,
    induction_cache: FxHashMap<IterationVar, IterationVarPtr>,
    instantiation_cache: FxHashMap<Instantiation, InstantiationPtr>,
    space: &'a SearchSpace,
    merged_dims: &'a MergedDimensions<'a>,
    device_code_args: &'a mut FxHashSet<ParamVal>,
}

impl<'a> ExprContext<'a> {
    pub fn new(
        space: &'a SearchSpace,
        merged_dims: &'a MergedDimensions<'a>,
        device_code_args: &'a mut FxHashSet<ParamVal>,
    ) -> Self {
        ExprContext {
            expr_cache: Default::default(),
            pack_cache: Default::default(),
            induction_cache: Default::default(),
            instantiation_cache: Default::default(),
            space,
            merged_dims,
            device_code_args,
        }
    }

    pub fn expr(&mut self, expr: &'a ir::IndexExpr) -> ExprPtr {
        let key = match *expr {
            ir::IndexExpr::LogicalDim(ldid) => self.logical_dim(ldid),
            ir::IndexExpr::Unpack(ref pack, idx) => ExprKey::Unpack(self.pack(pack), idx),
            ir::IndexExpr::Parameter(ref p) => self.process_parameter(p),
            ir::IndexExpr::Sum(cst, ref args) => ExprKey::Sum(
                cst,
                Cow::Owned(args.iter().map(|arg| self.expr(arg)).collect()),
            ),
            ir::IndexExpr::Unchecked(_) => unimplemented!(),
        };

        match self.expr_cache.entry(key) {
            Entry::Occupied(occupied) => occupied.get().clone(),
            Entry::Vacant(vacant) => {
                let expr = Arc::new(vacant.key().to_expr());
                vacant.insert(expr).clone()
            }
        }
    }

    pub fn parameter(&mut self, p: &'a Arc<ir::Parameter>) -> ExprPtr {
        Arc::new(self.process_parameter(p).to_expr())
    }

    fn process_parameter(&mut self, p: &'a Arc<ir::Parameter>) -> ExprKey<'a> {
        self.device_code_args.insert(ParamVal::External(
            p.clone(),
            self.space
                .ir_instance()
                .device()
                .lower_type(p.t, self.space)
                .unwrap(),
        ));

        ExprKey::Parameter(p)
    }

    fn pack(&mut self, pack: &'a ir::DimPack) -> PackPtr {
        let expr = self.expr(pack.expr());
        match self.pack_cache.entry(PackKey {
            expr,
            sizes: pack.sizes(),
        }) {
            Entry::Occupied(occupied) => occupied.get().clone(),
            Entry::Vacant(vacant) => {
                let mut div_size = Size::from(1);
                // We skip the last size: if we are in range, the division and modulo will always
                // be no-ops.
                let steps: Vec<_> = vacant
                    .key()
                    .sizes
                    .iter()
                    .skip(1)
                    .rev()
                    .map({
                        let device_code_args = &mut self.device_code_args;
                        let div_size = &mut div_size;
                        move |size| {
                            let mod_by = Size::from(size);
                            if let Some(arg) = ParamVal::from_size(&mod_by) {
                                device_code_args.insert(arg);
                            }

                            *div_size *= &mod_by;
                            // TODO: self.process_size_for_division(&div_size);
                            /*
                            self.device_code_args
                                .extend(ParamVal::div_magic(size, ir::Type::I(32)));
                            self.device_code_args
                                .extend(ParamVal::div_shift(size, ir::Type::I(32)));
                                */

                            UnpackStep {
                                div_by: div_size.clone(),
                                mod_by,
                            }
                        }
                    })
                    .collect();

                let pack = Arc::new(Pack {
                    expr: vacant.key().expr.clone(),
                    steps,
                });
                vacant.insert(pack).clone()
            }
        }
    }

    fn logical_dim(&mut self, id: ir::LogicalDimId) -> ExprKey<'a> {
        // TODO: RollingSum
        let mut outer_dims = Vec::new();
        let mut loop_dims = Vec::new();
        let mut instantiation_dims = Vec::new();

        let mut stride = Size::from(1u32);
        for dim in self.space.ir_instance().logical_dim(id).dimensions() {
            assert!(stride.as_int().is_some());

            let dim = &self.merged_dims[dim];

            match dim.kind() {
                DimKind::LOOP => loop_dims.push((dim.id(), stride.clone())),
                DimKind::INNER_VECTOR | DimKind::OUTER_VECTOR | DimKind::UNROLL => {
                    instantiation_dims.push((
                        dim.id(),
                        dim.size().as_int().unwrap(),
                        stride.clone(),
                    ))
                }
                DimKind::BLOCK | DimKind::THREAD => {
                    outer_dims.push((dim.id(), stride.clone()))
                }
                _ => panic!("invalid dim kind"),
            }

            stride *= dim.size();
        }

        // Loop dimensions must be in nesting order for `IterationVar`s (i.e. outermost dimensions
        // first)
        loop_dims.sort_unstable_by(|&(lhs, _), &(rhs, _)| {
            self.space
                .nesting_order(lhs)
                .partial_cmp(&rhs)
                .unwrap_or_else(|| {
                    panic!("invalid order for induction variable dimensions")
                })
        });

        outer_dims.sort_unstable_by_key(|&(dim, _)| dim);
        instantiation_dims.sort_unstable_by_key(|&(dim, _, _)| dim);

        let induction_var = match self.induction_cache.entry(IterationVar {
            outer_dims,
            loop_dims,
        }) {
            Entry::Occupied(occupied) => occupied.get().clone(),
            Entry::Vacant(vacant) => {
                let induction = Arc::new(vacant.key().clone());
                vacant.insert(induction).clone()
            }
        };

        let instantiation = match self.instantiation_cache.entry(Instantiation {
            induction_var,
            instantiation_dims,
        }) {
            Entry::Occupied(occupied) => occupied.get().clone(),
            Entry::Vacant(vacant) => {
                let instantiation = Arc::new(vacant.key().clone());
                vacant.insert(instantiation).clone()
            }
        };

        ExprKey::Instantiation(instantiation)
    }
}
