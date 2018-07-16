{{~#if is_symmetric}}
    // The filtered choice is symmetric, so only the lower triangular part needs to be
    // filtered.
    if {{>set.id_getter arguments.[0].[1] item=arguments.[0].[0]}} <
        {{>set.id_getter arguments.[1].[1] item=arguments.[1].[0]}} {
{{/if}}
    let mut values = {{>value_type.full_domain choice_full_type}};
    {{>filter_call filter_call}}
    trace!("call restrict from {}, line {}", file!(), line!());
    {{choice}}::restrict({{>choice.arg_ids}}ir_instance, store, values, diff)?;
{{~#if is_symmetric}}
}
{{/if}}
