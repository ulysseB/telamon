/* TODO(cc_perf): directly output the expected choice rather than Range */
Range::new_eq(&Range::ALL,
    {{#with Choice~}}
        {{>choice.getter use_old=../use_old}}
    {{~else~}}
        Range::new_eq(&Range::ALL, {{Code}})
    {{~/with~}}
)
