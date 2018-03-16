let incr = diff.{{incr_name}}.get(&({{>choice.arg_ids arguments=incr_args}})).map(|x| x.1)
    .unwrap_or_else(||store.get_{{incr_name}}({{>choice.arg_ids arguments=incr_args}}));
let (mut old_incr, mut new_incr) = {{#if to_half~}}
    (HalfRange::new_geq(old.min), HalfRange::new_geq(new.min))
{{~else~}}
    (old, new)
{{/if}};
let is_incr = incr.is({{incr_condition}});
if is_incr.maybe_false() {
    old_incr.min = {{zero}};
    new_incr.min = {{zero}};
}
{{~#unless is_half~}}
if is_incr.is_false() {
    old_incr.max = {{zero}};
    new_incr.max = {{zero}};
}
{{~/unless~}}
if old_incr != new_incr {
    store.update_{{name}}({{>choice.arg_ids}}old_incr, new_incr, diff)?;
}
