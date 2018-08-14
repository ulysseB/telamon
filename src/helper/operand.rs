//! Provides helpers to create instruction operands.
use device::ScalarArgument;
use ir::{self, dim, Function, InstId, mem, Operand};
use ir::Operand::*;
use utils::*;

/// Represents values that can be turned into an `Operand`.
pub trait AutoOperand<'a, L = ()> {
    /// Returns the corresponding `Operand`.
    fn get<'b>(&self, fun: &Function<'b, L>, active_dims: &HashMap<ir::DimId, ir::DimId>)
        -> Operand<'b, L> where 'a: 'b;
}

/// Helper to build `Reduce` operands.
pub struct Reduce(pub InstId);

/// Helper to build dim maps that can be lowered to temporary memory.
pub struct TmpArray(pub InstId);

impl<'a, L> AutoOperand<'a, L> for Reduce {
    fn get<'b>(&self, fun: &Function<'b, L>, active_dims: &HashMap<ir::DimId, ir::DimId>)
            -> Operand<'b, L> where 'a : 'b {
        let inst = fun.inst(self.0);
        let mut mapped_dims = Vec::new();
        let mut reduce_dims = Vec::new();
        for (&new_dim, &old_dim) in active_dims {
            if inst.iteration_dims().contains(&old_dim) {
                if old_dim != new_dim {
                    mapped_dims.push((old_dim, new_dim));
                }
            } else {
                reduce_dims.push(new_dim);
            }
        };
        Operand::new_reduce(inst, dim::Map::new(mapped_dims), reduce_dims)
    }
}

impl<'a> AutoOperand<'a, ()> for Operand<'a, ()> {
    fn get<'b>(&self, _: &Function<'b, ()>, _: &HashMap<ir::DimId, ir::DimId>)
            -> Operand<'b, ()> where 'a :'b {
        self.clone()
    }
}

impl<'a, T, L> AutoOperand<'a, L> for T where T: ScalarArgument {
    fn get<'b>(&self, _: &Function<'b, L>, _: &HashMap<ir::DimId, ir::DimId>)
            -> Operand<'b, L> where 'a: 'b {
        self.as_operand()
    }
}

impl<'a, 'c, L> AutoOperand<'a, L> for &'c str {
    fn get<'b>(&self, fun: &Function<'b, L>, _: &HashMap<ir::DimId, ir::DimId>)
            -> Operand<'b, L> where 'a: 'b {
        Param(unwrap!(fun.signature().params.iter().find(|p| p.name == *self)))
    }
}

impl<'a> AutoOperand<'a, ()> for InstId {
    fn get<'b>(&self, fun: &Function<'b, ()>, active_dims: &HashMap<ir::DimId, ir::DimId>)
            -> Operand<'b, ()> where 'a: 'b {
        let inst = fun.inst(*self);
        let mapped_dims = active_dims.iter().flat_map(|(&new_dim, &old_dim)| {
            if new_dim != old_dim && inst.iteration_dims().contains(&old_dim) {
                Some((old_dim, new_dim))
            } else { None }
        });
        Operand::new_inst(inst, dim::Map::new(mapped_dims), ir::DimMapScope::Thread)
    }
}

impl<'a> AutoOperand<'a, ()> for TmpArray {
    fn get<'b>(&self, fun: &Function<'b, ()>, active_dims: &HashMap<ir::DimId, ir::DimId>)
            -> Operand<'b, ()> where 'a: 'b {
        let inst = fun.inst(self.0);
        let mapped_dims = active_dims.iter().flat_map(|(&new_dim, &old_dim)| {
            if new_dim != old_dim && inst.iteration_dims().contains(&old_dim) {
                Some((old_dim, new_dim))
            } else { None }
        });
        Operand::new_inst(inst, dim::Map::new(mapped_dims), ir::DimMapScope::Global(()))
    }
}

impl<'a, L> AutoOperand<'a, L> for mem::InternalId {
    fn get<'b>(&self, _: &Function<'b, L>, _: &HashMap<ir::DimId, ir::DimId>)
            -> Operand<'b, L> where 'a: 'b {
        Operand::Addr(*self)
    }
}

impl<'a, L> AutoOperand<'a, L> for ir::DimId {
    fn get<'b>(&self, _: &Function<'b, L>, _: &HashMap<ir::DimId, ir::DimId>)
            -> Operand<'b, L> where 'a: 'b {
        Operand::Index(*self)
    }
}

impl<'a, L> AutoOperand<'a, L> for ir::IndVarId {
    fn get<'b>(&self, fun: &Function<'b, L>, _: &HashMap<ir::DimId, ir::DimId>)
            -> Operand<'b, L> where 'a: 'b {
        Operand::InductionVar(*self, fun.induction_var(*self).base().t())
    }
}
