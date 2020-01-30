//! Search space datastructures and constraint propagation.
use std::cmp;
use std::io::{self, Write};
use std::path::Path;

use crate::ir;
use log::debug;
use std::sync::Arc;

mod dim_map;
mod operand;
use utils::generated_file;
generated_file!(choices);

pub use self::choices::{
    Action, Bool, Choice, DimKind, Domain, DomainStore, InstFlag, MemSpace, NumSet,
    Order, ThreadMapping,
};

use self::choices::{apply_action, init_domain, DomainDiff};

/// A partially specified implementation.
#[derive(Clone)]
pub struct SearchSpace {
    ir_instance: Arc<ir::Function>,
    domain: DomainStore,
}

impl SearchSpace {
    /// Creates a new `SearchSpace` for the given `ir_instance`.
    pub fn new(
        ir_instance: ir::Function<()>,
        mut actions: Vec<Action>,
    ) -> Result<Self, ()> {
        // Pre-allocate IDs for future lowerings.
        let mut ir_instance = ir_instance.freeze();

        let mut domain = DomainStore::new(&ir_instance);
        // Enforce invariants.
        for inst in ir_instance.insts() {
            actions.extend(operand::inst_invariants(&ir_instance, inst));
        }
        let mut unused_diff = DomainDiff::default();
        for action in actions {
            apply_action(action, &mut domain, &mut unused_diff)?;
        }
        let actions = init_domain(&mut domain, &mut ir_instance)?;
        let mut space = SearchSpace {
            ir_instance: Arc::new(ir_instance),
            domain,
        };
        space.apply_decisions(actions)?;
        Ok(space)
    }

    /// Returns the underlying ir instance.
    pub fn ir_instance(&self) -> &ir::Function {
        &self.ir_instance
    }

    /// Returns the domain of choices.
    pub fn domain(&self) -> &DomainStore {
        &self.domain
    }

    /// Allows rewritting the domain.
    pub fn domain_mut(&mut self) -> &mut DomainStore {
        &mut self.domain
    }

    /// Applies a list of decisions to the domain and propagate constraints.
    pub fn apply_decisions(&mut self, actions: Vec<Action>) -> Result<(), ()> {
        choices::apply_decisions(actions, &mut self.ir_instance, &mut self.domain)
    }

    /// Triggers a layout lowering.
    pub fn lower_layout(
        &mut self,
        mem: ir::MemId,
        st_dims: &[ir::DimId],
        ld_dims: &[ir::DimId],
    ) -> Result<(), ()> {
        let actions = {
            let ir_instance = Arc::make_mut(&mut self.ir_instance);
            dim_map::lower_layout(ir_instance, mem, st_dims, ld_dims, &self.domain)?
        };
        self.apply_decisions(actions)
    }

    /// Returns a wrapper around a statement ID which can be compared according to nesting order
    /// (outermost is smallest).
    pub fn nesting_order<T: Into<ir::StmtId>>(&self, id: T) -> NestingOrder<'_> {
        NestingOrder {
            space: self,
            id: id.into(),
        }
    }

    pub fn innermost<T: Into<ir::StmtId> + Clone>(
        &self,
        lhs: Option<T>,
        rhs: Option<T>,
    ) -> Option<T> {
        let nlhs = lhs.clone().map(|dim| AssertOrd(self.nesting_order(dim)));
        let nrhs = rhs.clone().map(|dim| AssertOrd(self.nesting_order(dim)));

        if nlhs <= nrhs {
            rhs
        } else {
            lhs
        }
    }
}

/// Wrapper around a statement ID to compare it using nesting order.
///
/// `NestingOrder` implements `PartialEq<ir::StmtId>` and `PartialOrd<ir::StmtId>` such that merged
/// statements are equal, and loop dimensions are lower than any statement nested inside them.
pub struct NestingOrder<'a> {
    space: &'a SearchSpace,
    id: ir::StmtId,
}

impl cmp::PartialEq<ir::StmtId> for NestingOrder<'_> {
    fn eq(&self, other: &ir::StmtId) -> bool {
        if self.id == *other {
            return true;
        }

        match self.space.domain().get_order(self.id, *other) {
            Order::MERGED => true,
            _ => false,
        }
    }

    fn ne(&self, other: &ir::StmtId) -> bool {
        if self.id == *other {
            return false;
        }

        !self
            .space
            .domain()
            .get_order(self.id, *other)
            .intersects(Order::MERGED)
    }
}

impl cmp::PartialEq<ir::DimId> for NestingOrder<'_> {
    fn eq(&self, other: &ir::DimId) -> bool {
        *self == ir::StmtId::Dim(*other)
    }

    fn ne(&self, other: &ir::DimId) -> bool {
        *self != ir::StmtId::Dim(*other)
    }
}

impl<'a, 'b> cmp::PartialEq<NestingOrder<'a>> for NestingOrder<'b> {
    fn eq(&self, other: &NestingOrder<'a>) -> bool {
        *self == other.id
    }

    fn ne(&self, other: &NestingOrder<'a>) -> bool {
        *self != other.id
    }
}

impl cmp::PartialOrd<ir::StmtId> for NestingOrder<'_> {
    fn partial_cmp(&self, other: &ir::StmtId) -> Option<cmp::Ordering> {
        if self.id == *other {
            return Some(cmp::Ordering::Equal);
        }

        match self.space.domain().get_order(self.id, *other) {
            Order::OUTER => Some(cmp::Ordering::Less),
            Order::MERGED => Some(cmp::Ordering::Equal),
            Order::INNER => Some(cmp::Ordering::Greater),
            _ => None,
        }
    }

    fn le(&self, other: &ir::StmtId) -> bool {
        if self.id == *other {
            return true;
        }

        (Order::OUTER | Order::MERGED)
            .contains(self.space.domain().get_order(self.id, *other))
    }

    fn ge(&self, other: &ir::StmtId) -> bool {
        if self.id == *other {
            return true;
        }

        (Order::INNER | Order::MERGED)
            .contains(self.space.domain().get_order(self.id, *other))
    }
}

impl cmp::PartialOrd<ir::DimId> for NestingOrder<'_> {
    fn partial_cmp(&self, other: &ir::DimId) -> Option<cmp::Ordering> {
        self.partial_cmp(&ir::StmtId::Dim(*other))
    }

    fn lt(&self, other: &ir::DimId) -> bool {
        *self < ir::StmtId::Dim(*other)
    }

    fn le(&self, other: &ir::DimId) -> bool {
        *self <= ir::StmtId::Dim(*other)
    }

    fn gt(&self, other: &ir::DimId) -> bool {
        *self > ir::StmtId::Dim(*other)
    }

    fn ge(&self, other: &ir::DimId) -> bool {
        *self >= ir::StmtId::Dim(*other)
    }
}

impl<'a, 'b> cmp::PartialOrd<NestingOrder<'a>> for NestingOrder<'b> {
    fn partial_cmp(&self, other: &NestingOrder<'a>) -> Option<cmp::Ordering> {
        self.partial_cmp(&other.id)
    }

    fn lt(&self, other: &NestingOrder<'a>) -> bool {
        *self < other.id
    }

    fn le(&self, other: &NestingOrder<'a>) -> bool {
        *self <= other.id
    }

    fn gt(&self, other: &NestingOrder<'a>) -> bool {
        *self > other.id
    }

    fn ge(&self, other: &NestingOrder<'a>) -> bool {
        *self >= other.id
    }
}

#[derive(PartialEq, PartialOrd)]
pub struct AssertOrd<T>(pub T);

impl<T: PartialEq> Eq for AssertOrd<T> {}

impl<T: PartialOrd> cmp::Ord for AssertOrd<T> {
    fn cmp(&self, other: &AssertOrd<T>) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Update the domain after a lowering.
fn process_lowering(
    ir_instance: &mut ir::Function,
    domain: &mut DomainStore,
    new_objs: &ir::NewObjs,
    diff: &mut DomainDiff,
) -> Result<Vec<Action>, ()> {
    let mut actions = Vec::new();
    debug!("adding objects {:?}", new_objs);
    domain.alloc(ir_instance, new_objs);
    actions.extend(choices::init_domain_partial(
        domain,
        ir_instance,
        new_objs,
        diff,
    )?);
    // Enforce invariants and call manual triggers.
    for &inst in &new_objs.instructions {
        actions.extend(operand::inst_invariants(
            ir_instance,
            ir_instance.inst(inst),
        ));
    }
    Ok(actions)
}

/// Trigger to call when two dimensions are merged.
fn merge_dims(
    lhs: ir::DimId,
    rhs: ir::DimId,
    ir_instance: &mut ir::Function,
) -> Result<(ir::NewObjs, Vec<Action>), ()> {
    debug!("merge {:?} and {:?}", lhs, rhs);
    ir_instance.merge(lhs, rhs);
    Ok(Default::default())
}

/// Adds a iteration dimension to a basic block.
fn add_iteration_dim(
    ir_instance: &mut ir::Function,
    inst: ir::InstId,
    dim: ir::DimId,
) -> ir::NewObjs {
    debug!("set {:?} as iteration dim of inst {:?}", dim, inst);
    let mut new_objs = ir::NewObjs::default();
    if ir_instance.set_iteration_dim(inst, dim) {
        new_objs.add_iteration_dim(inst, dim);
    }
    new_objs
}

/// Adds a dimension to the list of thread dimensions.
fn add_thread_dim(ir_instance: &mut ir::Function, dim: ir::DimId) -> ir::NewObjs {
    debug!("set {:?} as a thread dimension", dim);
    let mut new_objs = ir::NewObjs::default();
    if ir_instance.add_thread_dim(dim) {
        new_objs.add_thread_dim(dim);
    }
    new_objs
}

/// Returns the memory space accessed by an access pattern.
pub fn access_pattern_space(
    pattern: &ir::AccessPattern,
    space: &SearchSpace,
) -> MemSpace {
    // We either have a `MemId` or the array is an external array in global memory.
    pattern
        .mem_block()
        .map(|id| space.domain().get_mem_space(id))
        .unwrap_or(MemSpace::GLOBAL)
}
