{{#if var~}}
    {{replace def.keys.Iter $fun="ir_instance" $var=var~}}
{{else~}}
    {{replace def.keys.Iter $fun="ir_instance"~}}
{{/if~}}
{{#each constraints~}}
    .flat_map(|obj| {{>set.from_superset this item="obj"}})
{{/each~}}
