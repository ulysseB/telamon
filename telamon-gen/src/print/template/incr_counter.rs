if old.is({{incr_condition}}).is_maybe() {
    let new_status = new.is({{incr_condition}});
    if new_status.is_true() {
        let value = {{>counter_value value use_old=true}};
        store.restrict_{{counter_name}}({{>choice.arg_ids}}
            {{~#if is_half~}}
                HalfRange { min: value.min }
            {{~else~}}
                Range { min: value.min, max: {{zero}} }
            {{~/if}}, diff)?;
    }
    {{~#unless is_half~}}
    else if new_status.is_false() {
        let value = {{>counter_value value use_old=true}};
        store.restrict_{{counter_name}}(
            {{>choice.arg_ids}}Range { min: {{zero}}, max: value.max }, diff)?;
    }
    {{~/unless}}
}
