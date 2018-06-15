{{#*inline "lhs"~}}
    {{#if arg_names}}{{arg_names.[0]}}{{else}}{{arguments.[0].[0]}}{{/if}}
{{~/inline~}}
{{~#*inline "rhs"~}}
    {{#if arg_names}}{{arg_names.[1]}}{{else}}{{arguments.[1].[0]}}{{/if}}
{{/inline~}}

{{#if is_symmetric~}}
let ({{>lhs}}, {{>rhs}}) = if {{#if arg_names~}}
        {{>set.id_getter arguments.[0].[1] item=arg_names.[0]~}}
        < {{>set.id_getter arguments.[1].[1] item=arg_names.[1]}} {
    {{~else~}}
        {{>set.id_getter arguments.[0].[1] item=arguments.[0].[0]~}}
        < {{>set.id_getter arguments.[1].[1] item=arguments.[1].[0]}} {
    {{~/if~}}
    ({{>lhs}}, {{>rhs}})
} else {
    ({{>rhs}}, {{>lhs}})
};
{{/if~}}
Arc::make_mut(&mut self.{{name}}).insert(({{~>choice.arg_ids}}),
                                         {{~value_type}}::all({{universe}}));
