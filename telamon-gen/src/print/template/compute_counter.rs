/// Compute the value of the counter can take depending on the values the increments can take.
pub fn compute_counter({{>choice.arg_defs ../this}}
                       ir_instance: &ir::Function,
                       store: &DomainStore,
                       diff: &DomainDiff)
        -> {{~#if half}} HalfRange {{else}} Range {{/if}}
{
    let mut counter_val = {{~#if half~}}
        HalfRange::new_eq(&(), {{base}}, &());
    {{~else~}}
        Range::new_eq(&(), {{base}}, &());
    {{~/if~}}
    {{#>loop_nest nest}}
        {{body}}
    {{/loop_nest}}
    counter_val
}
