{{#ifeq choice_def "Enum"}}!{{value}}{{/ifeq~}}
{{#ifeq choice_def "Integer"}}
    {
        {{~#each arguments}}
            let {{this.[0]}} = {{>set.item_getter this.[1] id=this.[0]}};
        {{~/each}}
        {{~value}}.complement({{>value_type.univers value_type}})
    }
{{~/ifeq~}}

