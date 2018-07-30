{{#each choice.arguments}}
    let {{this.[0]}} = {{>set.item_getter this.[1] id=this.[0]}};
{{/each~}}
{{#>loop_nest incr_iter}}
let incr_status = {{>choice.getter incr}}.is({{incr_condition}});
let incr_amount = {{>counter_value amount use_old=false}};
if incr_status.is_maybe() {
    debug!("restrict incr {{incr.name}}{:?} to {:?} with amount={:?}",
           ({{>choice.arg_ids incr}}), new_values, incr_amount);
    let mut val = {{>value_type.full_domain incr_type}};
    if current.min {{op}} NumSet::min(&incr_amount, {{>value_type.univers incr_type}}) > new_values.max {
        val.restrict(!({{incr_condition}}));
    }
    {{#unless is_half~}}
    if current.max < new_values.min {{op}} NumSet::max(&incr_amount, {{>value_type.univers incr_type}}) {
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
{{~#if amount.Choice~}}
    else if incr_status.is_true() {
        let max_val = new_values.max{{neg_op}}current.min{{op~}}
        NumSet::min(&incr_amount, {{>value_type.univers incr_type}});
        let val = {{>value_type.num_constructor t=amount.Choice.full_type
                    fun="new_leq" value="max_val"}};
        {{#if delayed~}}
            actions.extend({{amount.Choice.name}}::restrict_delayed(
                    {{~>choice.arg_ids amount.Choice~}} ir_instance, store, val)?);
        {{else~}}
            {{amount.Choice.name}}::restrict({{>choice.arg_ids amount.Choice~}}
                                              ir_instance, store, val, diff)?;
        {{/if~}}
    }
{{/if}}
{{/loop_nest}}
