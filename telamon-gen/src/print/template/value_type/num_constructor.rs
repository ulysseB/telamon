{{>value_type.name t}}::{{fun}}({{>value_type.univers t}}, {{value}},
    {{~#if value_type~}}
        {{>value_type.univers value_type}}
    {{~else~}}
        &()
    {{~/if~}}
)
