{{#if arg_names~}}
    {{#each arguments~}}
        {{#ifeq this.[0] (lookup ../arg_names @index)}}{{else~}}
            let {{this.[0]}} = {{(lookup ../arg_names @index)}};
        {{/ifeq}}
    {{/each~}}
{{/if~}}
let old_value = store.get_{{name}}({{>choice.arg_ids}});
{{#if compute_counter}}
    let counter_value = {{name}}::compute_counter(
        {{~>choice.arg_names}}ir_instance, store, diff);
    if old_value != counter_value {
        actions.push(Action::{{to_type_name name}}({{~>choice.arg_ids}}old_value.get_diff_
            {{~#ifeq choice_def.Counter.kind "Add"}}add{{else}}mul{{/ifeq~}}
            (counter_value)));
    }
{{/if~}}
let new_value = {{name}}::filter({{>choice.arg_names}}ir_instance, store);
actions.extend({{name}}::restrict_delayed(
        {{~>choice.arg_ids}}ir_instance, store, new_value)?);
