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

impl std::fmt::Display for Choice {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            {{~#each choices}}
            Choice::{{to_type_name name}}({{>choice.arg_names}}) => {
                fmt.write_str("{{to_type_name name}}(")?;
                {{~#each arguments}}
                write!(fmt, "{}", {{this.[0]}})?;
                {{~#unless @last}}
                fmt.write_str(", ")?;
                {{~/unless}}
                {{~/each}}
                fmt.write_str(")")
            },
            {{~/each}}
        }
    }
}
