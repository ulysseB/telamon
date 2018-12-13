use ast::context::CheckerContext;
use ast::error::TypeError;
use ast::{type_check_code, Condition, VarDef, VarMap};
use constraint::dedup_inputs;

use ir::{self, Adaptable};
use utils::RcStr;

use itertools::Itertools;

#[derive(Clone, Debug)]
pub struct TriggerDef {
    pub foralls: Vec<VarDef>,
    pub conditions: Vec<Condition>,
    pub code: String,
}

impl TriggerDef {
    /// Typecheck and registers a trigger.
    pub fn register_trigger(&self, ir_desc: &mut ir::IrDesc) {
        trace!("defining trigger '{}'", self.code);
        // Type check the code and the conditions.
        let ref mut var_map = VarMap::default();
        let foralls = self
            .foralls
            .iter()
            .map(|def| var_map.decl_forall(&ir_desc, def.to_owned()))
            .collect();
        let mut inputs = Vec::new();
        let conditions = self
            .conditions
            .iter()
            .map(|c| c.to_owned().type_check(&ir_desc, var_map, &mut inputs))
            .collect_vec();
        let code = type_check_code(RcStr::new(self.code.to_owned()), var_map);
        // Groups similiar inputs.
        let (inputs, input_adaptator) = dedup_inputs(inputs, &ir_desc);
        let conditions = conditions
            .into_iter()
            .map(|c| c.adapt(&input_adaptator))
            .collect_vec();
        // Adapt the trigger to the point of view of each inputs.
        let onchange_actions = inputs
            .iter()
            .enumerate()
            .map(|(pos, input)| {
                let (foralls, set_constraints, condition, adaptator) =
                    ir::ChoiceCondition::new(
                        &ir_desc,
                        inputs.clone(),
                        pos,
                        &conditions,
                        var_map.env(),
                    );
                let code = code.adapt(&adaptator);
                (
                    input.choice.clone(),
                    foralls,
                    set_constraints,
                    condition,
                    code,
                )
            })
            .collect_vec();
        // Add the trigger to the IR.
        let trigger = ir::Trigger {
            foralls,
            inputs,
            conditions,
            code,
        };
        let id = ir_desc.add_trigger(trigger);
        // Register the triggers to to be called when each input is modified.
        for (choice, forall_vars, set_constraints, condition, code) in onchange_actions {
            let action = ir::ChoiceAction::Trigger {
                id,
                condition,
                code,
                inverse_self_cond: false,
            };
            let on_change = ir::OnChangeAction {
                forall_vars,
                set_constraints,
                action,
            };
            ir_desc.add_onchange(&choice, on_change);
        }
    }

    /// Type checks the define's condition.
    pub fn define(
        self,
        context: &CheckerContext,
        triggers: &mut Vec<TriggerDef>,
    ) -> Result<(), TypeError> {
        triggers.push(self);
        Ok(())
    }
}
