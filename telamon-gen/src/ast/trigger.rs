use super::Condition;
use super::VarDef;
use super::error::TypeError;
use super::CheckerContext;
use super::TypingContext;

#[derive(Clone, Debug)]
pub struct TriggerDef {
    pub foralls: Vec<VarDef>,
    pub conditions: Vec<Condition>,
    pub code: String,
}

impl TriggerDef {
    /// Type checks the define's condition.
    pub fn define(
        self,
        context: &CheckerContext,
        tc: &mut TypingContext,
    ) -> Result<(), TypeError> {
        tc.triggers.push(self);
        Ok(())
    }
}
