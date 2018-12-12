//! Provides helpers to create instruction operands.
use crate::device::ScalarArgument;
use crate::helper::{Builder, LogicalDim};
use crate::ir::Operand::*;
use crate::ir::{self, dim, InstId, Operand};

/// Represents values that can be turned into an `Operand`.
pub trait AutoOperand<'a> {
    /// Returns the corresponding `Operand`.
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b, ()>
    where
        'a: 'b;
}

/// Helper to build `Reduce` operands.
pub struct Reduce(pub InstId);

/// Helper to build dim maps that can be lowered to temporary memory.
pub struct TmpArray(pub InstId);

impl<'a> AutoOperand<'a> for Reduce {
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b, ()>
    where
        'a: 'b,
    {
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

impl<'a> AutoOperand<'a> for Operand<'a, ()> {
    fn get<'b>(&self, _: &mut Builder<'b>) -> Operand<'b, ()>
    where
        'a: 'b,
    {
        self.clone()
    }
}

impl<'a, T> AutoOperand<'a> for T
where
    T: ScalarArgument,
{
    fn get<'b>(&self, _: &mut Builder<'b>) -> Operand<'b, ()>
    where
        'a: 'b,
    {
        self.as_operand()
    }
}

impl<'a, 'c> AutoOperand<'a> for &'c str {
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b, ()>
    where
        'a: 'b,
    {
        Param(unwrap!(builder
            .function()
            .signature()
            .params
            .iter()
            .find(|p| p.name == *self)))
    }
}

impl<'a> AutoOperand<'a> for InstId {
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b, ()>
    where
        'a: 'b,
    {
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

impl<'a> AutoOperand<'a> for ir::VarId {
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b, ()>
    where
        'a: 'b,
    {
        let val = builder.function().variable(*self);
        Operand::Variable(*self, val.t())
    }
}

impl<'a> AutoOperand<'a> for TmpArray {
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b, ()>
    where
        'a: 'b,
    {
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

impl<'a> AutoOperand<'a> for ir::MemId {
    fn get<'b>(&self, _: &mut Builder<'b>) -> Operand<'b, ()>
    where
        'a: 'b,
    {
        Operand::Addr(*self)
    }
}

impl<'a> AutoOperand<'a> for LogicalDim {
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b, ()>
    where
        'a: 'b,
    {
        if self.real_ids.len() == 1 {
            Operand::Index(self[0])
        } else {
            let one = ir::Size::new_const(1);
            let ind_var = builder.induction_var(&0i32, vec![(self, one)]);
            ind_var.get(builder)
        }
    }
}

impl<'a> AutoOperand<'a> for ir::IndVarId {
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b, ()>
    where
        'a: 'b,
    {
        let t = builder.function().induction_var(*self).base().t();
        Operand::InductionVar(*self, t)
    }
}
