use super::*;

#[derive(Clone, Debug)]
pub struct CounterDef {
    pub name: Spanned<RcStr>,
    pub doc: Option<String>,
    pub visibility: ir::CounterVisibility,
    pub vars: Vec<VarDef>,
    pub body: CounterBody,
}

impl CounterDef {
    pub fn define(
        self,
        context: &mut CheckerContext,
        tc: &mut TypingContext,
    ) -> Result<(), TypeError> {
        tc.choice_defs.push(ChoiceDef::CounterDef(self));
        Ok(())
    }
}

impl PartialEq for CounterDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
