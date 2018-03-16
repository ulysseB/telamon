{{#if var~}}
    {{replace def.keys.ItemGetter $fun="ir_instance" $id=id $var=var~}}
{{else~}}
    {{replace def.keys.ItemGetter $fun="ir_instance" $id=id~}}
{{/if~}}
