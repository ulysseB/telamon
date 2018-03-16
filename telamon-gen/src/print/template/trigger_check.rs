#[allow(unused_variables)]
fn check_trigger_{{id}}({{>choice.arg_defs}}ir_instance: &ir::Function,
                        store: &DomainStore, diff: &DomainDiff) -> bool {
    {{#each inputs}}
        let {{this.[0]}} = {{>choice.getter this.[1] use_old=true}};
    {{/each~}}
    true {{#each conditions}} && {{this}} {{/each}}
}
