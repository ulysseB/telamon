let incr = store.get_old_{{incr_name}}({{>choice.arg_ids arguments=incr_args}}diff);
let (mut old_incr, mut new_incr) = (
    {{~>value_type.num_constructor
         t=counter_type fun="new_eq" value="old" value_type=incr_type}},
    {{~>value_type.num_constructor
         t=counter_type fun="new_eq" value="new" value_type=incr_type}});
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
