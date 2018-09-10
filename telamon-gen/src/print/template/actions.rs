/// A decision to apply to the domain.
#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub enum Action {
    {{~#each choices}}
        /// cbindgen:field-names=[{{~#each arguments}}{{this.[0]}}, {{/each~}}domain]
        {{to_type_name name}}(
        {{~#each arguments}}{{this.[1].def.keys.IdType}},{{/each~}}
        {{>value_type.name value_type}}),
    {{/each~}}
}

impl Action {
    /// Returns the action performing the complementary decision.
    #[allow(unused_variables)]
    pub fn complement(&self, ir_instance: &ir::Function) -> Option<Self> {
        match *self {
            {{~#each choices}}
                Action::{{to_type_name name}}({{>choice.arg_names}}domain) =>
                {{~#if choice_def.Counter}}None{{else}}Some(
                    Action::{{to_type_name name}}({{>choice.arg_names}}
                                                  {{>choice.complement value="domain"}})
                ){{~/if}},
            {{~/each}}
        }
    }
}

/// Applies an action to the domain.
pub fn apply_action(action: Action, store: &mut DomainStore, diff: &mut DomainDiff)
        -> Result<(), ()> {
    debug!("applying action {:?}", action);
    match action {
        {{~#each choices}}
            Action::{{to_type_name name}}({{#each arguments}}{{this.[0]}}, {{/each}}value) =>
            store.restrict_{{name}}({{#each arguments}}{{this.[0]}}, {{/each}}value, diff),
        {{~/each}}
    }
}
