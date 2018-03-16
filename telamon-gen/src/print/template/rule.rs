{{#>set_constraints set_conditions~}}
    {{#if conditions~}}
        if {{#each conditions~}}
            {{#unless @first}} && {{/unless~}}{{this}}
        {{/each~}}
        { {{var}}.restrict({{values}}); }
    {{else~}}
        {{var}}.restrict({{values}});
    {{/if~}}
{{/set_constraints~}}
