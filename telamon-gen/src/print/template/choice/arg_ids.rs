{{#each arguments~}}
    {{~#if ../arg_names~}}
        {{>set.id_getter this.[1] item=(lookup ../arg_names @index)}}
    {{~else~}}
        {{>set.id_getter this.[1] item=this.[0]}}
    {{~/if~}},
{{~/each}}
