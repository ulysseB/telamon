#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize, Hash, Ord, PartialOrd)]
#[repr(C)]
pub enum Choice {
    {{~#each choices}}
        /// cbindgen:field-names=[{{~#each arguments}}{{this.[0]}}, {{/each~}}domain]
        {{to_type_name name}}(
        {{~#each arguments}}{{this.[1].def.keys.IdType}},{{/each~}}
        ),
    {{/each~}}
}

impl From<Action> for Choice {
    fn from(action: Action) -> Choice {
        match action {
            {{~#each choices}}
                Action::{{to_type_name name}}(
                {{#each arguments}}{{this.[0]}},{{~/each}} _ ) => Choice::{{to_type_name name}}({{#each arguments}}{{this.[0]}}, {{~/each}}),
            {{~/each}}
        }
    }
}
