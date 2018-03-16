{{#with NewObjs~}}
    if {{list}}.iter().any(|&c| c {{~#if ../is_triangular}} >= {{else}} == {{/if~}}
        {{~#if set.var~}}
            ({{>set.id_getter def=set.def.arg item=set.var}}, {{>set.id_getter set item=../var~}})
        {{~else~}}
            {{>set.id_getter set item=../var~}}
        {{/if~}}) { continue; }
{{~/with~}}
{{#with Variable~}}
    if {{>set.id_getter def=set item=../var~}}
        {{#if ../is_triangular}} <= {{else}} == {{/if~}}
        {{>set.id_getter def=set item=conflict_var}} { continue; }
{{~/with~}}

