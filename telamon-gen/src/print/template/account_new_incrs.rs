{{#ifeq visibility "Full"}}{{else~}}
    let incr = {{>choice.getter incr use_old=true}}.is({{incr_condition}});
    let incr_value = {{incr_amount}};
    if incr.is_true() {
        actions.push(Action::{{to_type_name counter.name}}({{>choice.arg_ids counter}}
            {{~#ifeq visibility "NoMax"~}}
                HalfRange { min: {{min_incr_amount}} }
            {{~else~}}
                Range { min: {{min_incr_amount}}, max: {{zero}} }
            {{~/ifeq~}}
        ));
    }
    {{~#ifeq visibility "HiddenMax"~}}
        else if incr.is_false() {
            actions.push(Action::{{to_type_name counter.name}}({{>choice.arg_ids counter}}
                Range { min: {{zero}}, max: {{max_incr_amount}} }
            ));
        }
    {{/ifeq}}
{{/ifeq}}
