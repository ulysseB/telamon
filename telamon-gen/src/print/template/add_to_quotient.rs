{{~#*inline "args"~}}
    {{#if var}}${{var}},{{/if}}${{item}},
{{~/inline~}}

if store.get_old_{{counter_name}}({{>args}}diff).max == 0 {
    if DELAYED {
        actions.extend({{repr_name}}::restrict_delayed({{>args}}ir_instance, store, Bool::TRUE)?);
    } else {
        store.restrict_{{repr_name}}({{>args}}Bool::TRUE, diff)?;
    }
    let new_objs = {{add_to_set}};
    Ok((new_objs, vec![]))
} else { Ok((NewObjs::default(), vec![])) }
