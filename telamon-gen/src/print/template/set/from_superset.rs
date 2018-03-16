{{#if var~}}
    {{replace def.keys.FromSuperset $fun="ir_instance" $item=item $var=var~}}
{{else~}}
    {{replace def.keys.FromSuperset $fun="ir_instance" $item=item~}}
{{/if~}}
{{#each constraints}}
    .and_then(|x| {{>set.from_superset this $fun="ir_instance" item="x"}})
{{/each}}
