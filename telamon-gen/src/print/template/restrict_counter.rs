{{#each choice.arguments}}
    let {{this.[0]}} = {{>set.item_getter this.[1] id=this.[0]}};
{{/each~}}
{{#>loop_nest incr_iter}}
let incr_status = {{>choice.getter incr}}.is({{incr_condition}});
let incr_amount = {{>counter_value amount use_old=false}};
if incr_status.is_maybe() {
    debug!("restrict incr {{incr.name}}{:?} to {:?} with amount={:?}",
           ({{>choice.arg_ids incr}}), new_values, incr_amount);
    let mut val = {{incr_type}}::ALL;
    if current.min {{op}} incr_amount.min > new_values.max {
        val.restrict(!({{incr_condition}}));
    }
    {{#unless is_half~}}
    if current.max < new_values.min {{op}} incr_amount.max {
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
{{~#if amount.Counter~}}
    else if incr_status.is_true() {
        let val = Range::new_leq(new_values.max{{neg_op}}current.min{{op}}incr_amount.min);
        {{#if delayed~}}
            actions.extend({{amount.Counter.name}}::restrict_delayed(
                    {{~>choice.arg_ids amount.Counter~}} ir_instance, store, val)?);
        {{else~}}
            {{amount.Counter.name}}::restrict({{>choice.arg_ids amount.Counter~}}
                                              ir_instance, store, val, diff)?;
        {{/if~}}
    }
{{/if}}
{{/loop_nest}}
