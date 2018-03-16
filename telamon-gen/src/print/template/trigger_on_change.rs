{{#each inputs~}}
    let {{this.[0]}} = {{>choice.getter this.[1] use_old=true}};
{{/each~}}
if true {{#each others_conditions}} && {{this}} {{/each}} {
    if old.is({{self_condition}}).maybe_true() && new.is({{self_condition}}).is_true() {
        trigger_{{call_id}}.push(({{>choice.arg_ids}}));
    }
}
