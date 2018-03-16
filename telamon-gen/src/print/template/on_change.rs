{{#>set_constraints constraints~}}
    {{#>loop_nest loop_nest~}}
        {{#ifeq action "FilterSelf"}}{{>filter_self choice=../../../../this}}{{/ifeq~}}
        {{#with action.Filter}}{{>choice_filter}}{{/with~}}
        {{#with action.IncrCounter}}{{>incr_counter}}{{/with~}}
        {{#with action.UpdateCounter}}{{>update_counter}}{{/with~}}
        {{#with action.Trigger}}{{>trigger_on_change}}{{/with~}}
    {{/loop_nest~}}
{{/set_constraints~}}
