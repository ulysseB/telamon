{{#ifeq visibility "Full"~}}
    store.restrict_{{incr.name}}({{>choice.arg_ids incr}}!(
            {{~incr_condition}}), &mut unused_diff)?;
{{~/ifeq}}
{{#ifeq visibility "HiddenMax"~}}
    let incr = {{>choice.getter incr use_old=true}}.is({{incr_condition}});
    let incr_value = {{incr_amount}};
    let max_incr_value = {{max_incr_amount}};
    let mut counter_value = {{>choice.getter counter}};
    {{#ifeq kind "Add"~}}
        counter_value.max += max_incr_value;
    {{else~}}
        counter_value.max *= max_incr_value;
    {{/ifeq~}}
    store.set_{{counter.name}}({{>choice.arg_ids counter}}counter_value);
    let remove = if let Some(diff) = diff.{{counter.name}}
            .get_mut(&({{>choice.arg_ids counter}})) {
        {{#ifeq kind "Add"~}}
            diff.1.max += max_incr_value;
        {{else~}}
            {{#ifeq kind "Mul"~}}
                diff.1.max *= max_incr_value;
            {{else}}
                compiler_error!("invalid counter kind: {{kind}}");
            {{/ifeq~}}
        {{/ifeq~}}
        diff.0.max <= diff.1.max
    } else { false };
    if remove { diff.{{counter.name}}.remove(&({{>choice.arg_ids counter}})); }
{{~/ifeq~}}
