{{#with Counter~}}
    {{>choice.getter use_old=../use_old}}
{{~else~}}
    Range::new_eq({{Code}})
{{~/with~}}
