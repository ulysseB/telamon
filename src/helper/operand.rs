//! Provides helpers to create instruction operands.
use std::sync::Arc;

use crate::device::ScalarArgument;
use crate::helper::{Builder, LogicalDim};
use crate::ir::Operand::*;
use crate::ir::{self, dim, InstId, Operand};

use utils::unwrap;

/// Represents values that can be turned into an `Operand`.
pub trait AutoOperand {
    /// Returns the corresponding `Operand`.
    fn get(&self, builder: &mut Builder) -> Operand<()>;
}

/// Helper to build `Reduce` operands.
pub struct Reduce(pub InstId);

/// Helper to build dim maps that can be lowered to temporary memory.
pub struct TmpArray(pub InstId);

impl AutoOperand for Reduce {
    fn get(&self, builder: &mut Builder) -> Operand<()> {
        let inst = builder.function().inst(self.0);
        let mut mapped_dims = Vec::new();
        let mut reduce_dims = Vec::new();
        for (new_dim, old_dim) in builder.open_dims() {
            if inst.iteration_dims().contains(&old_dim) {
                if old_dim != new_dim {
                    mapped_dims.push((old_dim, new_dim));
                }
            } else {
                reduce_dims.push(new_dim);
            }
        }
        Operand::new_reduce(inst, dim::Map::new(mapped_dims), reduce_dims)
    }
}

impl AutoOperand for Operand<()> {
    fn get(&self, _: &mut Builder) -> Operand<()> {
        self.clone()
    }
}

impl<T> AutoOperand for T
where
    T: ScalarArgument,
{
    fn get(&self, _: &mut Builder) -> Operand<()> {
        self.as_operand()
    }
}

impl AutoOperand for &'_ str {
    fn get(&self, builder: &mut Builder) -> Operand<()> {
        Param(Arc::clone(unwrap!(builder
            .function()
            .signature()
            .params
            .iter()
            .find(|p| p.name == *self))))
    }
}

impl AutoOperand for InstId {
    fn get(&self, builder: &mut Builder) -> Operand<()> {
        let inst = builder.function().inst(*self);
        let mapped_dims = builder.open_dims().flat_map(|(new_dim, old_dim)| {
            if new_dim != old_dim && inst.iteration_dims().contains(&old_dim) {
                Some((old_dim, new_dim))
            } else {
                None
            }
        });
        Operand::new_inst(inst, dim::Map::new(mapped_dims), ir::DimMapScope::Thread)
    }
}

impl AutoOperand for ir::VarId {
    fn get(&self, builder: &mut Builder) -> Operand<()> {
        let val = builder.function().variable(*self);
        Operand::Variable(*self, val.t())
    }
}

impl AutoOperand for TmpArray {
    fn get(&self, builder: &mut Builder) -> Operand<()> {
        let inst = builder.function().inst(self.0);
        let mapped_dims = builder.open_dims().flat_map(|(new_dim, old_dim)| {
            if new_dim != old_dim && inst.iteration_dims().contains(&old_dim) {
                Some((old_dim, new_dim))
            } else {
                None
            }
        });
        Operand::new_inst(
            inst,
            dim::Map::new(mapped_dims),
            ir::DimMapScope::Global(()),
        )
    }
}

impl AutoOperand for ir::MemId {
    fn get(&self, _: &mut Builder) -> Operand<()> {
        Operand::Addr(*self)
    }
}

impl AutoOperand for LogicalDim {
    fn get(&self, builder: &mut Builder) -> Operand<()> {
        if self.real_ids.len() == 1 {
            Operand::Index(self[0])
        } else {
            let one = ir::Size::new_const(1);
            let ind_var = builder.induction_var(&0i32, vec![(self, one)]);
            ind_var.get(builder)
        }
    }
}

impl AutoOperand for ir::IndVarId {
    fn get(&self, builder: &mut Builder) -> Operand<()> {
        let t = builder.function().induction_var(*self).base().t();
        Operand::InductionVar(*self, t)
    }
}

impl AutoOperand for ir::AccessId {
    fn get(&self, _builder: &mut Builder) -> Operand<()> {
        Operand::ComputedAddress(*self)
    }
}
