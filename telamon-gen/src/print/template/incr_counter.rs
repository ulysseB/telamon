if old.is({{incr_condition}}).is_maybe() {
    let new_status = new.is({{incr_condition}});
    let value = {{value_getter}};
    if new_status.is_true() {
        store.restrict_{{counter_name}}({{>choice.arg_ids}}
            {{~#if is_half~}}
                HalfRange { min: {{min}} }
            {{~else~}}
                Range { min: {{min}}, max: {{zero}} }
            {{~/if}}, diff)?;
    }
    {{~#unless is_half~}}
    else if new_status.is_false() {
        store.restrict_{{counter_name}}(
            {{>choice.arg_ids}}Range { min: {{zero}}, max: {{max}} }, diff)?;
    }
    {{~/unless}}
}
