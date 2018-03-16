{{#if compute_counter}}
    let values = {{name}}::compute_counter({{>choice.arg_names}}ir_instance, store, diff);
    store.set_{{name}}({{>choice.arg_ids}}values);
{{/if~}}
let values = {{name}}::filter({{>choice.arg_names}}ir_instance, store);
{{name}}::restrict({{>choice.arg_ids}}ir_instance, store, values, &mut unused_diff)?;
