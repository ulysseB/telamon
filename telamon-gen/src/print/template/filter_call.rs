{{#>loop_nest loop_nest~}}
    {{#with filter_ref.Inline~}}
        {{#each rules~}}
            {{>rule}}
            trace!("inline filter restricts to {:?}", values);
        {{/each~}}
    {{/with~}}
    {{#with filter_ref.Call~}}
        let filter_res = {{choice}}::filter_{{id}}({{>choice.arg_names}}ir_instance, store);
        {{value_var}}.restrict(filter_res);
    {{/with}}
{{/loop_nest~}}
