{{#each levels~}}
    for {{this.[0]}} in {{>set.iterator this.[1]}} {
        {{~#each this.[2]~}}
            {{>conflict var=../this.[0] is_triangular=../../triangular}}
        {{~/each~}}
{{~/each~}}
{{> @partial-block ../this ~}}
{{#each levels}} } {{/each}}
