use ast::{Check, ChoiceDef, Constraint, SetDef, TriggerDef, TypedConstraint};
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
        for choice_def in self.choice_defs.iter() {
            match choice_def {
                ChoiceDef::CounterDef(counter_def) => {
                    counter_def.register_counter(
                        &mut self.ir_desc,
                        &mut self.constraints
                    );
                }
                _ => {}
            }
        }
        for trigger in self.triggers.iter() {
            trigger.register_trigger(&mut self.ir_desc);
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
