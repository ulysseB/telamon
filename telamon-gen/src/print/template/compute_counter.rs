/// Compute the value of the counter can take depending on the values the increments can take.
pub fn compute_counter({{>choice.arg_defs ../this}}
                       ir_instance: &ir::Function,
                       store: &DomainStore,
                       diff: &DomainDiff)
        -> {{~#if half}} HalfRange {{else}} Range {{/if}}
{
    let mut counter_val =
        {{~#if half}}HalfRange::new_geq{{else}}Range::new_eq{{/if}}({{base}});
    {{#>loop_nest nest}}
        let value = {{>counter_value value use_old=true}};
        let incr = {{>choice.getter incr use_old=true}};
        {{#unless half~}}
            if ({{incr_condition}}).intersects(incr) { counter_val.max {{op}}= value.max; }
        {{/unless~}}
        if ({{incr_condition}}).contains(incr) { counter_val.min {{op}}= value.min; }
    {{/loop_nest}}
    counter_val
}
