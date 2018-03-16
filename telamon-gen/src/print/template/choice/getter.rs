{{#if use_old~}}
    store.get_old_{{name}}({{>choice.arg_ids}}diff)
{{~else~}}
    store.get_{{name}}({{>choice.arg_ids}})
{{~/if~}}
