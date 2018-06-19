{{#*inline "choice_ids"~}}
    ({{~#each arguments}}{{this.[1].def.keys.IdType}},{{/each~}})
{{/inline~}}

/// Stores the domains of each variable.
#[derive(Clone, Debug, Default)]
pub struct DomainStore {
    {{#each choices}}
        {{name}}: Arc<HashMap<{{>choice_ids this}}, {{>value_type.name value_type}}>>,
    {{/each}}
}

#[allow(dead_code)]
impl DomainStore {
    /// Creates a new domain store and allocates the variables for the given BBMap.
    pub fn new(ir_instance: &ir::Function) -> Self {
        let mut store = DomainStore::default();
        store.init(ir_instance);
        store
    }

    /// Initializes the domain.
    #[allow(unused_variables, unused_mut)]
    fn init(&mut self, ir_instance: &ir::Function) {
        {{#each choices~}}
            {{#>loop_nest iteration_space~}}
                {{>alloc}}
            {{/loop_nest~}}
        {{/each~}}
    }

    /// Allocates the choices when new objects are created.
    #[allow(unused_variables, unused_mut)]
    pub fn alloc(&mut self, ir_instance: &ir::Function, new_objs: &ir::NewObjs) {
        {{#each partial_iterators~}}
            {{#>iter_new_objects this.[0]~}}
                {{>alloc this.[1].choice arg_names=this.[1].arg_names value_type=this.[1].value_type}}
            {{/iter_new_objects~}}
        {{/each~}}
    }

    {{#each choices}}{{>getter this}}{{/each}}
}

/// Stores the old values of a modified `DomainStore`.
#[derive(Default)]
pub struct DomainDiff {
    {{#each choices}}
        pub {{name}}: HashMap<{{>choice_ids this}},
            ({{>value_type.name value_type}}, {{>value_type.name value_type}})>,
    {{/each}}
}

impl DomainDiff {
    /// Indicates if the `DomainDiff` does not holds any modification.
    pub fn is_empty(&self) -> bool {
        {{~#if choices~}}
            {{#each choices~}}
                {{~#unless @first}} && {{/unless~}}
                self.{{name}}.is_empty()
            {{~/each}}
        {{else}}true{{/if~}}
    }
    {{#each choices}}
        /// Removes all the modifications of '{name}' and returns them.
        pub fn pop_{{name}}_diff(&mut self)
            -> Option<(({{>choice_ids}}),
                       {{~>value_type.name value_type}}, {{>value_type.name value_type}})> {
            self.{{name}}.keys().cloned().next().map(|k| {
                let (old, new) = unwrap!(self.{{name}}.remove(&k));
                (k, old, new)
            })
        }
    {{/each}}
}
