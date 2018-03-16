for (pos, &{{#if set.arg~}}(obj_var, obj){{else}}obj{{/if~}})
    in {{>set.new_objs def=set objs="new_objs"}}.iter().enumerate() {
    {{#if set.arg~}}
        let obj_var = {{>set.item_getter def=set.arg id="obj_var"}};
    {{/if~}}
    let obj = {{>set.item_getter def=set id="obj" var="obj_var"}};
    {{#each arg_conflicts~}}
        {{>conflict var="obj_var" is_triangular=false}}
    {{/each~}}
    {{#each loop_nest.levels~}}
        for {{this.[0]}} in {{>set.iterator this.[1]}} {
            {{~#each this.[2]~}}
                {{>conflict var=../this.[0]}}
            {{~/each~}}
    {{~/each~}}
    {{>@partial-block ../this}}
    {{#each loop_nest.levels}} } {{/each}}
}
