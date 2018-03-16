/// Restricts the values `{{../name}}` can take.
#[allow(unused_variables, unused_mut, unused_parens)]
pub fn filter_{{id}}({{>choice.arg_defs}}ir_instance: &ir::Function,
                     store: &DomainStore) -> {{type_name}} {
    let mut values = {{type_name}}::FAILED;
    {{#each bindings~}}
        let {{this.[0]}} = {{>choice.getter this.[1]}};
    {{/each~}}
    {{>positive_filter body}}
    trace!("restrict {{../name}}{:?} to {:?} in filter {{id}}", ({{>choice.arg_ids}}), values);
    {{#each bindings~}}
        trace!("with {{this.[0]}} = {:?}", {{this.[0]}});
    {{/each~}}
    values
}
