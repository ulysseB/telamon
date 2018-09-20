use super::{Check, ChoiceDef, Constraint, SetDef, TriggerDef, TypedConstraint};
use ir;

use itertools::Itertools;

#[derive(Default, Clone, Debug)]
pub struct TypingContext {
    pub ir_desc: ir::IrDesc,
    pub set_defs: Vec<SetDef>,
    pub choice_defs: Vec<ChoiceDef>,
    pub triggers: Vec<TriggerDef>,
    pub constraints: Vec<Constraint>,
    pub checks: Vec<Check>,
}

impl TypingContext {
    /// Type-checks the statements in the correct order.
    pub fn finalize(mut self) -> (ir::IrDesc, Vec<TypedConstraint>) {
        for choice_def in self.choice_defs.clone().iter_mut() {
            match choice_def {
                ChoiceDef::CounterDef(counter_def) => {
                    counter_def.register_counter(
                        counter_def.name.data.clone(),
                        counter_def.doc.clone(),
                        counter_def.visibility.clone(),
                        counter_def.vars.clone(),
                        counter_def.body.clone(),
                        &mut self,
                    );
                }
                _ => {}
            }
        }
        for trigger in self.triggers.clone().iter() {
            trigger.register_trigger(
                trigger.foralls.clone(),
                trigger.conditions.clone(),
                trigger.code.clone(),
                &mut self,
            );
        }
        let constraints = {
            let ir_desc = &self.ir_desc;
            self.constraints
                .into_iter()
                .flat_map(move |constraint| constraint.type_check(ir_desc))
                .collect_vec()
        };
        for check in self.checks {
            check.check(&self.ir_desc);
        }
        (self.ir_desc, constraints)
    }
}
