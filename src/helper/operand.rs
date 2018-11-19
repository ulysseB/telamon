//! Provides helpers to create instruction operands.
use device::ScalarArgument;
use helper::{Builder, LogicalDim};
use ir::Operand::*;
use ir::{self, Operand};

/// Represents objects that can be converted into a variable.
pub trait ToVariable {
    /// Returns the corresponding variable. Allocates it in the builder if needed.
    fn to_variable(&self, builder: &mut Builder) -> ir::VarId;

    /// Converts the `self` into a variable operand.
    fn to_operand<'a>(&self, builder: &mut Builder<'a>) -> Operand<'a> {
        let var_id = self.to_variable(builder);
        let var = builder.function().variable(var_id);
        Operand::Variable(var_id, var.t())
    }
}

impl ToVariable for ir::VarId {
    fn to_variable(&self, _: &mut Builder) -> ir::VarId {
        *self
    }
}

impl ToVariable for ir::InstId {
    /// Returns a variable that holds the value produced by `self`, with point-to-point
    /// communication to the current loop nest.
    fn to_variable(&self, builder: &mut Builder) -> ir::VarId {
        let inst_var = builder.get_inst_variable(*self);
        builder.map_variable(inst_var)
    }
}

/// Helper to take the last value of a variable in a loop nest.
pub struct Last<'a, T: ToVariable>(pub T, pub &'a [&'a LogicalDim]);

impl<'a, T: ToVariable> ToVariable for Last<'a, T> {
    fn to_variable(&self, builder: &mut Builder) -> ir::VarId {
        let original_var = self.0.to_variable(builder);
        if self.1.is_empty() {
            original_var
        } else {
            builder.create_last_variable(original_var, self.1)
        }
    }
}

/// Represents values that can be turned into an `Operand`.
pub trait AutoOperand<'a> {
    /// Returns the corresponding `Operand`.
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b>
    where
        'a: 'b;
}

// We cannot provide a blanket implementation for all `ToVariable` objects until rust
// supports implementation specialization.
impl<'a> AutoOperand<'a> for ir::VarId {
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b>
    where
        'a: 'b,
    {
        self.to_operand(builder)
    }
}

impl<'a> AutoOperand<'a> for ir::InstId {
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b>
    where
        'a: 'b,
    {
        self.to_operand(builder)
    }
}

impl<'a, 'c, T: ToVariable> AutoOperand<'a> for Last<'c, T> {
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b>
    where
        'a: 'b,
    {
        self.to_operand(builder)
    }
}

impl<'a> AutoOperand<'a> for Operand<'a> {
    fn get<'b>(&self, _: &mut Builder<'b>) -> Operand<'b>
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
    fn get<'b>(&self, _: &mut Builder<'b>) -> Operand<'b>
    where
        'a: 'b,
    {
        self.as_operand()
    }
}

impl<'a, 'c> AutoOperand<'a> for &'c str {
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b>
    where
        'a: 'b,
    {
        Param(unwrap!(
            builder
                .function()
                .signature()
                .params
                .iter()
                .find(|p| p.name == *self)
        ))
    }
}

impl<'a> AutoOperand<'a> for ir::MemId {
    fn get<'b>(&self, _: &mut Builder<'b>) -> Operand<'b>
    where
        'a: 'b,
    {
        Operand::Addr((*self).into())
    }
}

impl<'a> AutoOperand<'a> for LogicalDim {
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b>
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
    fn get<'b>(&self, builder: &mut Builder<'b>) -> Operand<'b>
    where
        'a: 'b,
    {
        let t = builder.function().induction_var(*self).base().t();
        Operand::InductionVar(*self, t)
    }
}
