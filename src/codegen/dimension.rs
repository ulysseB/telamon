use std::ops;

use fxhash::{FxHashMap, FxHashSet};

use crate::codegen;
use crate::ir;
use crate::search_space::{Advance, DimKind, Domain, Order, SearchSpace};

use super::expr::ExprPtr;

/// An iteration dimension composed of one or mure fused dimensions.
///
/// Note that induction levels are only associated with IR dimensions that are actually used by an
/// instruction.  For instance, if dimensions `%0` and `%2` are merged in the following code:
///
///   @0[%0]: load(a[%0])
///   @1[%1, %2]: load(b[%1])
///
/// then only `%0` would have an associated level in the corresponding iteration dimension.
///
/// Note that the representant does not necessarily have an associated induction level; in the
/// example above, the representant could be either `%0` or `%2` (with the other dimension in
/// `other_dims`).
#[derive(Debug, Clone)]
pub struct Dimension {
    /// The iteration kind.  This should be fully constrained.
    kind: DimKind,
    /// The representant.  This can be any of the fused IR dimensions represented by this
    /// dimension.
    representant: ir::DimId,
    /// The IR dimensions represented by this dimensions other than the representant.
    other_dims: Vec<ir::DimId>,
    /// The size of this iteration dimension.
    size: codegen::Size,

    /// Dimensions for which this dimension is advanced
    advanced_for: FxHashSet<ir::DimId>,

    /// Index expressions which need to be initialized (in order) before entering the dimension.
    init_exprs: Vec<ExprPtr>,
    /// Index expressions which need to be computed (in order) at the beginning of each iteration.
    compute_exprs: Vec<ExprPtr>,
    /// Index expressions which need to be updated (in order) at the end of each iteration.
    update_exprs: Vec<ExprPtr>,
    /// Index expressions which need to be reset (in order) after exiting the dimension.
    reset_exprs: Vec<ExprPtr>,
}

impl Dimension {
    /// Returns the ID of the representant.
    pub fn id(&self) -> ir::DimId {
        self.representant
    }

    /// Returns the kind of the dimension.
    pub fn kind(&self) -> DimKind {
        self.kind
    }

    /// Returns the size of the dimensions.
    pub fn size(&self) -> &codegen::Size {
        &self.size
    }

    /// Returns the ids of the `ir::Dimensions` represented by this dimension.
    pub fn dim_ids(&self) -> impl Iterator<Item = ir::DimId> {
        std::iter::once(self.representant).chain(self.other_dims.clone())
    }

    /// Merge another `Dimension` into this one.
    pub fn merge_from(&mut self, other: Self) {
        assert_eq!(self.kind, other.kind);
        assert_eq!(self.size, other.size);
        self.other_dims.push(other.representant);
        self.other_dims.extend(other.other_dims);
    }

    /// Returns the values to pass from the host to the device to implement `self`.
    pub fn host_values<'b>(
        &'b self,
        space: &'b SearchSpace,
    ) -> impl Iterator<Item = codegen::ParamVal> + 'b {
        let size_param = if self.kind == DimKind::LOOP {
            codegen::ParamVal::from_size(&self.size)
        } else {
            None
        };
        size_param.into_iter()
    }

    /// Creates a new dimension from an `ir::Dimension`.
    fn new(dim: &ir::Dimension, space: &SearchSpace) -> Self {
        let kind = space.domain().get_dim_kind(dim.id());
        assert!(kind.is_constrained());
        Dimension {
            kind,
            representant: dim.id(),
            size: codegen::Size::from_ir(dim.size(), space),
            other_dims: Default::default(),
            advanced_for: Default::default(),

            init_exprs: Default::default(),
            compute_exprs: Default::default(),
            update_exprs: Default::default(),
            reset_exprs: Default::default(),
        }
    }

    /// Adds `dim` to the list of fused dimensions if it is indeed the case.
    fn try_add_fused_dim(&mut self, dim: &ir::Dimension, space: &SearchSpace) -> bool {
        let order = space
            .domain()
            .get_order(self.representant.into(), dim.id().into());
        assert!(order.is_constrained());
        if order == Order::MERGED {
            self.other_dims.push(dim.id());
            debug_assert_eq!(self.kind, space.domain().get_dim_kind(dim.id()));
            debug_assert_eq!(self.size, codegen::Size::from_ir(dim.size(), space));
            if cfg!(debug) {
                for &other in &self.other_dims {
                    let order = space.domain().get_order(dim.id().into(), other.into());
                    assert_eq!(order, Order::MERGED);
                }
            }
            true
        } else {
            false
        }
    }

    /// Indicates whether the dimension is advanced for a given dimension.
    pub fn is_advanced(&self, dim_id: ir::DimId) -> bool {
        self.advanced_for.contains(&dim_id)
    }
}

/// Helper structure to associate a dimension ID to the corresponding merged `codegen::Dimension`
///
/// The iteration dimension associated with an `ir::DimId` can be accessed using the indexing
/// operator.
pub struct MergedDimensions<'a> {
    inner: FxHashMap<ir::DimId, &'a Dimension>,
}

impl<'a, 'b: 'a> Extend<&'b Dimension> for MergedDimensions<'a> {
    fn extend<I: IntoIterator<Item = &'b Dimension>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.inner.reserve(iter.size_hint().0);

        for dim in iter {
            for dim_id in dim.dim_ids() {
                let inserted = self.inner.insert(dim_id, dim).is_none();
                assert!(inserted);
            }
        }
    }
}

impl<'a, 'b: 'a> std::iter::FromIterator<&'b Dimension> for MergedDimensions<'a> {
    fn from_iter<I: IntoIterator<Item = &'b Dimension>>(iter: I) -> Self {
        MergedDimensions {
            inner: iter
                .into_iter()
                .flat_map(|dim| dim.dim_ids().map(move |id| (id, dim)))
                .collect(),
        }
    }
}

impl<'a> ops::Index<ir::DimId> for MergedDimensions<'a> {
    type Output = Dimension;

    fn index(&self, idx: ir::DimId) -> &Dimension {
        self.inner[&idx]
    }
}

/// Creates the final list of dimensions by grouping fused `ir::Dimension`.
pub fn group_merged_dimensions<'a>(space: &'a SearchSpace) -> Vec<Dimension> {
    let mut groups: Vec<Dimension> = Vec::new();
    'dim: for dim in space.ir_instance().dims() {
        for group in &mut groups {
            if group.try_add_fused_dim(dim, space) {
                continue 'dim;
            }
        }
        groups.push(Dimension::new(dim, space));
    }

    // Add advance information
    for group in &mut groups {
        if space.domain().get_num_dim_advances(group.id().into()).min > 0 {
            for dim in space.ir_instance().dims() {
                if group.id() == dim.id() {
                    continue;
                }

                if space
                    .domain()
                    .get_advance(group.id().into(), dim.id())
                    .is(Advance::YES)
                    .is_true()
                {
                    group.advanced_for.insert(dim.id());
                }
            }
        }
    }

    groups
}
