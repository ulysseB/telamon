{{#each choice.arguments}}
    let {{this.[0]}} = {{>set.item_getter this.[1] id=this.[0]}};
{{/each~}}
{{#>loop_nest incr_iter}}
let incr_status = {{>choice.getter incr}}.is({{incr_condition}});
let incr_amount = {{incr_amount}};
if incr_status.is_maybe() {
    debug!("restrict incr {{incr.name}}{:?} to {:?} with amount={:?}",
           ({{>choice.arg_ids incr}}), new_values, incr_amount);
    let mut val = {{>value_type.full_domain incr_type}};
    if current.min {{op}} {{min}} > new_values.max {
        val.restrict(!({{incr_condition}}));
    }
    {{#unless is_half~}}
    if current.max < new_values.min {{op}} {{max}} {
        val.restrict({{incr_condition}});
    }
    {{~/unless}}
    {{#if delayed~}}
        actions.extend({{incr.name}}::restrict_delayed(
                {{~>choice.arg_ids incr}}ir_instance, store, val)?);
    {{~else~}}
        store.restrict_{{incr.name}}({{>choice.arg_ids incr}}val, diff)?;
    {{~/if}}
}
{{#if delayed}}{{restrict_amount_delayed}}{{else}}{{restrict_amount}}{{/if}}
{{/loop_nest}}
