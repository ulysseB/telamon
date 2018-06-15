// Generates an action to filter a choice if needed.
let mut values = {{>value_type.full_domain choice_full_type}};
{{>filter_call filter_call}}
trace!("call restrict from {}, line {}", file!(), line!());
{{choice}}::restrict({{>choice.arg_ids}}ir_instance, store, values, diff)?;
