{{#with Counter~}}
    {{>choice.getter use_old=../use_old}}
{{~else~}}
    // FIXME(unimplemented): handle NumericSet values
    Range::new_eq(&Range::ALL, {{Code}})
{{~/with~}}
