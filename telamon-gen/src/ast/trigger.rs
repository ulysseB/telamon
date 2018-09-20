use super::context::CheckerContext;
use super::error::TypeError;
use super::typing_context::TypingContext;
use super::{type_check_code, Condition, VarDef, VarMap};
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
    pub fn register_trigger(
        &self,
        foralls: Vec<VarDef>,
        conditions: Vec<Condition>,
        code: String,
        tc: &mut TypingContext,
    ) {
        trace!("defining trigger '{}'", code);
        // Type check the code and the conditions.
        let ref mut var_map = VarMap::default();
        let foralls = foralls
            .into_iter()
            .map(|def| var_map.decl_forall(&tc.ir_desc, def))
            .collect();
        let mut inputs = Vec::new();
        let conditions = conditions
            .into_iter()
            .map(|c| c.type_check(&tc.ir_desc, var_map, &mut inputs))
            .collect_vec();
        let code = type_check_code(RcStr::new(code), var_map);
        // Groups similiar inputs.
        let (inputs, input_adaptator) = dedup_inputs(inputs, &tc.ir_desc);
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
                        &tc.ir_desc,
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
        let id = tc.ir_desc.add_trigger(trigger);
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
            tc.ir_desc.add_onchange(&choice, on_change);
        }
    }

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
