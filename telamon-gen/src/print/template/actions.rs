/// A decision to apply to the domain.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum Action {
    {{~#each choices}}
        {{to_type_name name}}(
        {{~#each arguments}}{{this.[1].def.keys.IdType}},{{/each~}}
        {{value_type}}),
    {{/each~}}
}

/// Applies an action to the domain.
fn apply_action(action: Action, store: &mut DomainStore, diff: &mut DomainDiff)
        -> Result<(), ()> {
    debug!("applying action {:?}", action);
    match action {
        {{~#each choices}}
            Action::{{to_type_name name}}({{#each arguments}}{{this.[0]}}, {{/each}}value) =>
            store.restrict_{{name}}({{#each arguments}}{{this.[0]}}, {{/each}}value, diff),
        {{~/each}}
    }
}
