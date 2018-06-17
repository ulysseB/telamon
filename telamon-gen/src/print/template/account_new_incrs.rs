{{#ifeq visibility "Full"}}{{else~}}
    let incr = {{>choice.getter incr use_old=true}}.is({{incr_condition}});
    let incr_value = {{>counter_value value use_old=true}};
    if incr.is_true() {
        actions.push(Action::{{to_type_name counter.name}}({{>choice.arg_ids counter}}
            {{~#ifeq visibility "NoMax"~}}
                HalfRange { min: NumDomain::min(&incr_value) }
            {{~else~}}
                Range { min: NumDomain::min(&incr_value), max: {{zero}} }
            {{~/ifeq~}}
        ));
    }
    {{~#ifeq visibility "HiddenMax"~}}
        else if incr.is_false() {
            actions.push(Action::{{to_type_name counter.name}}({{>choice.arg_ids counter}}
                Range { min: {{zero}}, max: NumDomain::max(&incr_value) }
            ));
        }
    {{/ifeq}}
{{/ifeq}}
