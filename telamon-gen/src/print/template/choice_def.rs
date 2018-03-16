{{#*inline "ids_decl"~}}
    {{#each arguments}}{{this.[0]}}: {{this.[1].def.keys.IdType}}, {{/each}}
{{~/inline~}}

{{#if doc}}///{{doc}}{{/if}}
#[allow(dead_code)]
mod {{name}} {
    use super::ir;
    #[allow(unused_imports)]
    use super::ir::prelude::*;
    use std::sync::Arc;
    use super::*;

    /// Returns the values a `{{name}}` can take.
    #[allow(unused_variables, unused_mut, unused_parens)]
    pub fn filter({{>choice.arg_defs}}ir_instance: &ir::Function,
                  store: &DomainStore) -> {{full_value_type}} {
        let mut values = {{full_value_type}}::ALL;
        {{#each filter_actions}}{{>filter_action}}{{/each}}
        values
    }

    /// Triggers the required actions after the domain is updated.
    #[allow(unused_variables, unused_mut, unused_parens)]
    pub fn on_change(old: {{value_type}}, new: {{value_type}},
                     {{>ids_decl}}ir_instance: &mut Arc<ir::Function>,
                     store: &mut DomainStore,
                     diff: &mut DomainDiff) -> Result<(), ()> {
        {{#each trigger_calls~}}
            let mut trigger_{{@index}} = Vec::new();
        {{/each~}}
        {
            {{#each arguments~}}
                let {{this.[0]}} = {{>set.item_getter this.[1] id=this.[0]}};
            {{~/each~}}
            {{#each on_change}}{{>on_change}}{{/each}}
        }
        let mut actions = Vec::new();
        {{#each trigger_calls}}{{>trigger_call make_mut=true call_id=@index}}{{/each~}}
        for action in actions { apply_action(action, store, diff)?; }
        Ok(())
    }

    /// Generate actions that restrict the values the choice can take.
    #[allow(unused_variables, unused_mut, unused_parens)]
    pub fn restrict_delayed({{>ids_decl}}ir_instance: &ir::Function,
                            store: &DomainStore,
                            mut new_values: {{full_value_type}}) -> Result<Vec<Action>, ()> {
        let current = store.get_{{name}}({{>choice.arg_names}});
        {{#if restrict_counter.is_half~}}
            new_values.min = ::std::cmp::max(new_values.min, current.min);
        {{~else~}}
            new_values.restrict(current);
        {{~/if}}
        debug!("delayed restrict {{name}}{:?} to {:?}", ({{>choice.arg_names}}), new_values);
        if new_values.is_failed() { return Err(()); }
        {{#if restrict_counter~}}
            let mut actions = Vec::new();
            {{>restrict_counter restrict_counter choice=this delayed=true}}
            Ok(actions)
        {{~else~}}
            if new_values != current {
                Ok(vec![Action::{{to_type_name name}}({{>choice.arg_names}}new_values)])
            } else { Ok(vec![]) }
        {{/if}}
    }

    /// Restrict the values the choice can take.
    #[allow(unused_variables, unused_mut, unused_parens)]
    pub fn restrict({{>ids_decl}}ir_instance: &ir::Function,
                    store: &mut DomainStore,
                    mut new_values: {{full_value_type}},
                    diff: &mut DomainDiff) -> Result<(), ()> {
        {{#if restrict_counter~}}
            let current = store.get_{{name}}({{>choice.arg_names}});
            {{#if restrict_counter.is_half~}}
                new_values.min = ::std::cmp::max(new_values.min, current.min);
            {{else~}}
                new_values.restrict(current);
            {{/if~}}
            if new_values.is_failed() { return Err(()); }
            {{>restrict_counter restrict_counter choice=this}}
            Ok(())
        {{~else~}}
            store.restrict_{{name}}({{>choice.arg_names}}new_values, diff)
        {{~/if}}
    }

    {{#each filters}}{{>filter}}{{/each}}
    {{#with compute_counter~}}
        {{>compute_counter}}
    {{/with~}}
}
