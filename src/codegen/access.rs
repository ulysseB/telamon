use std::ops;
use std::sync::Arc;

use fxhash::{FxHashMap, FxHashSet};

use crate::ir;
use crate::search_space::{DimKind, Order, SearchSpace};

use super::dimension::MergedDimensions;
use super::{ParamVal, Size};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct IterationVarId(usize);

/// An iteration variable, within a range.  The range is used for predicates.
///
/// The iteration variable gets updated in lockstep with loops.
pub struct IterationVar {
    where_def: Option<ir::DimId>,
    outer_dims: Vec<(ir::DimId, Size)>,
    min: Size,
    max: Size,
}

impl IterationVar {
    pub fn t(&self) -> ir::Type {
        ir::Type::I(32)
    }

    pub fn outer_dims(&self) -> &[(ir::DimId, Size)] {
        &self.outer_dims
    }

    pub fn min(&self) -> &Size {
        &self.min
    }

    pub fn max(&self) -> &Size {
        &self.max
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct IterationVarKey {
    outer_dims: Vec<(ir::DimId, Size)>,
    loop_dims: Vec<(ir::DimId, Size)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct UnpackId(usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RollingUnpack {
    div_by: Size,
    mod_by: Size,
}

impl RollingUnpack {
    pub fn div_by(&self) -> &Size {
        &self.div_by
    }
    pub fn mod_by(&self) -> &Size {
        &self.mod_by
    }
}

/// Unpacking of a dimension into multiple dimensions
pub struct Unpack {
    source: IndexVarId,
    rolling: Vec<RollingUnpack>,
}

impl Unpack {
    pub fn source(&self) -> IndexVarId {
        self.source
    }

    pub fn rolling(&self) -> &[RollingUnpack] {
        &self.rolling
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct IndexVarId(usize);

/// An index variable
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexVar {
    /// An instantiated iteration variable
    Iteration {
        where_def: Option<ir::DimId>,
        source: IterationVarId,
        // id, size, stride
        instantiation_dims: Vec<(ir::DimId, u32, Size)>,
    },
    /// A variable that was unpacked from another
    Unpack {
        where_def: Option<ir::DimId>,
        source: UnpackId,
        index: usize,
    },
    /// A parameter
    Parameter(Arc<ir::Parameter>),
    /// A constant
    Constant(i32),
    /// The sum of multiple variables.  Usually used for `p + r` part of convolutions.
    Sum {
        where_def: Option<ir::DimId>,
        args: Vec<IndexVarId>,
    },
}

impl IndexVar {
    /// Returns where the variable is defined.
    ///
    /// If `None` it is defined at the global level and always available.
    pub fn where_defined(&self) -> Option<ir::DimId> {
        use IndexVar::*;

        match *self {
            Iteration { where_def, .. }
            | Unpack { where_def, .. }
            | Sum { where_def, .. } => where_def,
            Parameter(..) | Constant(..) => None,
        }
    }
}

#[derive(Default)]
pub struct IndexVars {
    iterations: Vec<IterationVar>,
    vars: Vec<IndexVar>,
    unpacks: Vec<Unpack>,

    cache: FxHashMap<IndexVar, IndexVarId>,
    unpack_cache: FxHashMap<(IndexVarId, Vec<RollingUnpack>), UnpackId>,
    iteration_cache: FxHashMap<IterationVarKey, IterationVarId>,

    // Iteration variables to update on a loop iteration
    loop_update_iters: FxHashMap<ir::DimId, Vec<(IterationVarId, Size)>>,

    // Variables defined at a loop
    loop_def: FxHashMap<ir::DimId, Vec<DefId>>,

    // Variables defined at the global level
    global_def: Vec<DefId>,
}

impl ops::Index<IterationVarId> for IndexVars {
    type Output = IterationVar;

    fn index(&self, idx: IterationVarId) -> &IterationVar {
        &self.iterations[idx.0]
    }
}

impl ops::Index<IndexVarId> for IndexVars {
    type Output = IndexVar;

    fn index(&self, idx: IndexVarId) -> &IndexVar {
        &self.vars[idx.0]
    }
}

impl ops::Index<UnpackId> for IndexVars {
    type Output = Unpack;

    fn index(&self, idx: UnpackId) -> &Unpack {
        &self.unpacks[idx.0]
    }
}

#[derive(Debug, Copy, Clone)]
pub enum DefId {
    IndexVar(IndexVarId),
    IterationVar(IterationVarId),
    Unpack(UnpackId),
}

impl IndexVars {
    pub fn instantiation_dims(&self, var: IndexVarId) -> Vec<(ir::DimId, usize)> {
        match self[var] {
            IndexVar::Iteration {
                ref instantiation_dims,
                ..
            } => instantiation_dims
                .iter()
                .map(|&(dim, size, ref _stride)| (dim, size as usize))
                .collect(),
            IndexVar::Unpack { source, .. } => {
                self.instantiation_dims(self[source].source)
            }
            IndexVar::Parameter(..) | IndexVar::Constant(..) => Vec::new(),
            IndexVar::Sum { ref args, .. } => args
                .iter()
                .flat_map(|&id| self.instantiation_dims(id))
                .collect(),
        }
    }

    pub fn iteration_vars(
        &self,
    ) -> impl Iterator<Item = (IterationVarId, &'_ IterationVar)> {
        self.iterations
            .iter()
            .enumerate()
            .map(|(id, var)| (IterationVarId(id), var))
    }

    pub fn index_vars(&self) -> impl Iterator<Item = (IndexVarId, &'_ IndexVar)> {
        self.vars
            .iter()
            .enumerate()
            .map(|(id, var)| (IndexVarId(id), var))
    }

    pub fn unpacks(&self) -> impl Iterator<Item = (UnpackId, &'_ Unpack)> {
        self.unpacks
            .iter()
            .enumerate()
            .map(|(id, var)| (UnpackId(id), var))
    }

    pub fn var_def_at(
        &self,
        where_def: Option<ir::DimId>,
    ) -> impl Iterator<Item = DefId> + '_ {
        if let Some(dim) = where_def {
            self.loop_def.get(&dim)
        } else {
            Some(&self.global_def)
        }
        .into_iter()
        .flatten()
        .cloned()
    }

    pub fn iteration_updates(
        &self,
        dim: ir::DimId,
    ) -> impl Iterator<Item = (IterationVarId, &Size)> {
        self.loop_update_iters
            .get(&dim)
            .into_iter()
            .flatten()
            .map(|&(id, ref size)| (id, size))
    }

    // loop_dims must be in nesting order
    pub fn add_iteration(
        &mut self,
        total_size: Size,
        mut outer_dims: Vec<(ir::DimId, Size)>,
        loop_dims: Vec<(ir::DimId, Size)>,
        mut instantiation_dims: Vec<(ir::DimId, u32, Size)>,
    ) -> IndexVarId {
        use std::collections::hash_map::Entry;

        outer_dims.sort_unstable_by_key(|&(dim, _)| dim);
        let iteration_id = match self.iteration_cache.entry(IterationVarKey {
            outer_dims,
            loop_dims,
        }) {
            Entry::Occupied(occupied) => *occupied.get(),
            Entry::Vacant(vacant) => {
                let id = IterationVarId(self.iterations.len());
                let key = vacant.key();

                for &(loop_dim, ref stride) in &key.loop_dims {
                    self.loop_update_iters
                        .entry(loop_dim)
                        .or_insert(Vec::new())
                        .push((id, stride.clone()));
                }

                /*
                if let Some(&(outer_dim, _)) = key.loop_dims.first() {
                    self.loop_def
                        .entry(outer_dim)
                        .or_insert(Vec::new())
                        .push(DefId::IterationVar(id));
                } else {
                */
                self.global_def.push(DefId::IterationVar(id));
                //};

                // The variable is fully defined on the innermost loop
                let where_def = key.loop_dims.last().map(|&(inner_dim, _)| inner_dim);

                self.iterations.push(IterationVar {
                    where_def,
                    outer_dims: key.outer_dims.clone(),
                    min: Size::from(0),
                    max: total_size,
                });

                *vacant.insert(id)
            }
        };

        instantiation_dims.sort_unstable_by_key(|&(dim, _, _)| dim);
        self.add(IndexVar::Iteration {
            where_def: self[iteration_id].where_def,
            source: iteration_id,
            instantiation_dims,
        })
    }

    pub fn add_unpack(
        &mut self,
        source: IndexVarId,
        rolling: Vec<RollingUnpack>,
        index: usize,
    ) -> IndexVarId {
        use std::collections::hash_map::Entry;

        let where_def = self[source].where_defined();
        let unpack_id = match self.unpack_cache.entry((source, rolling)) {
            Entry::Occupied(occupied) => *occupied.get(),
            Entry::Vacant(vacant) => {
                let id = UnpackId(self.unpacks.len());
                let (source, rolling) = vacant.key();

                if let Some(dim) = where_def {
                    self.loop_def
                        .entry(dim)
                        .or_insert(Vec::new())
                        .push(DefId::Unpack(id));
                } else {
                    self.global_def.push(DefId::Unpack(id));
                }

                self.unpacks.push(Unpack {
                    source: *source,
                    rolling: rolling.clone(),
                });

                *vacant.insert(id)
            }
        };

        self.add(IndexVar::Unpack {
            where_def,
            source: unpack_id,
            index,
        })
    }

    pub fn add_parameter(&mut self, p: Arc<ir::Parameter>) -> IndexVarId {
        self.add(IndexVar::Parameter(p))
    }

    pub fn add_constant(&mut self, value: i32) -> IndexVarId {
        self.add(IndexVar::Constant(value))
    }

    pub fn add_sum(
        &mut self,
        where_def: Option<ir::DimId>,
        mut args: Vec<IndexVarId>,
    ) -> IndexVarId {
        args.sort_unstable();

        self.add(IndexVar::Sum { where_def, args })
    }

    fn add(&mut self, index_var: IndexVar) -> IndexVarId {
        use std::collections::hash_map::Entry;

        match self.cache.entry(index_var) {
            Entry::Occupied(occupied) => *occupied.get(),
            Entry::Vacant(vacant) => {
                let id = IndexVarId(self.vars.len());
                let var = vacant.key();

                if let Some(dim) = var.where_defined() {
                    self.loop_def
                        .entry(dim)
                        .or_insert(Vec::new())
                        .push(DefId::IndexVar(id));
                } else {
                    self.global_def.push(DefId::IndexVar(id));
                }

                self.vars.push(var.clone());
                *vacant.insert(id)
            }
        }
    }
}

pub struct VarWalker<'a> {
    pub merged_dims: &'a MergedDimensions<'a>,
    pub space: &'a SearchSpace,
    pub device_code_args: &'a mut FxHashSet<ParamVal>,
    pub index_vars: &'a mut IndexVars,
}

impl<'a> VarWalker<'a> {
    pub fn process_parameter(&mut self, p: &Arc<ir::Parameter>) {
        self.device_code_args.insert(ParamVal::External(
            p.clone(),
            self.space
                .ir_instance()
                .device()
                .lower_type(p.t, self.space)
                .unwrap(),
        ));
    }

    pub fn process_size(&mut self, size: &Size) {
        if let Some(arg) = ParamVal::from_size(size) {
            self.device_code_args.insert(arg);
        }
    }

    pub fn process_size_for_division(&mut self, size: &Size) {
        self.process_size(size);
        self.device_code_args
            .extend(ParamVal::div_magic(size, ir::Type::I(32)));
        self.device_code_args
            .extend(ParamVal::div_shift(size, ir::Type::I(32)));
    }

    pub fn process_index_expr(&mut self, expr: &ir::IndexExpr) -> IndexVarId {
        use ir::IndexExpr::*;

        match *expr {
            LogicalDim(id) => {
                let mut global_dims = Vec::new();
                let mut loop_dims = Vec::new();
                let mut instantiation_dims = Vec::new();

                let mut stride = Size::from(1u32);
                for dim in self.space.ir_instance().logical_dim(id).dimensions() {
                    self.process_size(&stride);

                    let dim = &self.merged_dims[dim];

                    match dim.kind() {
                        DimKind::LOOP => loop_dims.push((dim.id(), stride.clone())),
                        DimKind::INNER_VECTOR
                        | DimKind::OUTER_VECTOR
                        | DimKind::UNROLL => instantiation_dims.push((
                            dim.id(),
                            dim.size().as_int().unwrap(),
                            stride.clone(),
                        )),
                        DimKind::BLOCK | DimKind::THREAD => {
                            global_dims.push((dim.id(), stride.clone()))
                        }
                        _ => panic!("invalid dim kind"),
                    }

                    stride *= dim.size();
                }

                // Loop dimensions must be in nesting order for `IterationVars`
                loop_dims.sort_unstable_by(|&(lhs, _), &(rhs, _)| {
                    if lhs == rhs {
                        return std::cmp::Ordering::Equal;
                    }

                    match self.space.domain().get_order(lhs.into(), rhs.into()) {
                        Order::INNER => std::cmp::Ordering::Greater,
                        Order::OUTER => std::cmp::Ordering::Less,
                        Order::MERGED => {
                            panic!("found MERGED order between representants")
                        }
                        _ => panic!("invalid order for induction variable dimensions"),
                    }
                });

                self.index_vars.add_iteration(
                    stride,
                    global_dims,
                    loop_dims,
                    instantiation_dims,
                )
            }
            Unpack(id) => {
                // TODO: Compute the stuff we need (eg the magic factors)
                let packed = &self.space.ir_instance().packed_dims()[id];
                let source = self.process_index_expr(&LogicalDim(packed.logical_dim()));

                let mut div_size = Size::from(1);
                let mut rolling: Vec<_> = packed
                    .sizes()
                    .iter()
                    .rev()
                    .map(|size| {
                        let size = Size::from_ir(
                            &ir::PartialSize::from(size.clone()),
                            self.space,
                        );
                        self.process_size(&size);

                        div_size *= &size;
                        self.process_size_for_division(&div_size);

                        // let mut prev = index;
                        //
                        //loop {
                        // let cur = (index + index.mul_high(div_magic)) >> div_shift
                        // items.push(prev - cur * mod_by)
                        // prev = cur;
                        // }
                        //
                        // items.push(prev);
                        RollingUnpack {
                            div_by: div_size.clone(),
                            mod_by: size,
                        }
                    })
                    .collect();

                // Reverse the index because we compute innermost first
                let rolling_index = rolling.len() - id.unpack_index() - 1;
                self.index_vars.add_unpack(source, rolling, rolling_index)
            }
            Parameter(ref p) => {
                self.process_parameter(p);

                self.index_vars.add_parameter(p.clone())
            }
            Constant(c) => self.index_vars.add_constant(c),
            Sum(ref exprs) => {
                // TODO: merge parameters & constant as single value !
                let mut where_def: Option<ir::DimId> = None;
                let mut vars = Vec::with_capacity(exprs.len());
                for expr in exprs {
                    let id = self.process_index_expr(expr);
                    vars.push(id);

                    if let Some(where_var) = self.index_vars[id].where_defined() {
                        if let Some(cur_def) = where_def {
                            if cur_def != where_var {
                                match self
                                    .space
                                    .domain()
                                    .get_order(where_var.into(), cur_def.into())
                                {
                                    Order::INNER => where_def = Some(where_var),
                                    _ => (),
                                }
                            }
                        } else {
                            where_def = Some(where_var);
                        }
                    }
                }

                self.index_vars.add_sum(where_def, vars)
            }
        }
    }
}
