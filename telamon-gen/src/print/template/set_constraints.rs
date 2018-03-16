{{#each this~}}
    if let Some({{var}}) =
        {{~#each sets~}}
            {{~#if @first~}}
                {{>set.from_superset this item=../var}}
            {{~else~}}
                .and_then(|v| {{>set.from_superset this item="v"}})
            {{~/if~}}
        {{~/each~}}
    {
{{/each~}}
{{> @partial-block ../this ~}}
{{#each this}} } {{/each}}
