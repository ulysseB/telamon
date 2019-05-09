/// A decision to apply to the domain.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy, Serialize, Deserialize, Ord, PartialOrd)]
#[repr(C)]
pub enum Action {
    {{~#each choices}}
        /// cbindgen:field-names=[{{~#each arguments}}{{this.[0]}}, {{/each~}}domain]
        {{to_type_name name}}(
        {{~#each arguments}}{{this.[1].def.keys.IdType}},{{/each~}}
        {{>value_type.name value_type}}),
    {{/each~}}
}

/// Helper struct for printing actions with `format!` and `{}`.
///
/// Actions contains IDs which references a [`Function`], and thus can't be properly displayed
/// without extra information (the objects referred to by the IDs).  This struct embeds a reference
/// to the [`Function`], which allows it to implement the [`Display`] trait for actions.
///
/// [`Display`]: std::fmt::Display
/// [`Function`]: crate::ir::Function
pub struct DisplayAction<'a> {
    action: &'a Action,
    ir_instance: &'a ir::Function,
}

impl<'a> std::fmt::Debug for DisplayAction<'a> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.action, fmt)
    }
}

impl<'a> std::fmt::Display for DisplayAction<'a> {
    #[allow(unused_variables)]
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            fmt, "{:?} = {}",
            Choice::from(*self.action),
            self.action.display_domain(self.ir_instance))
    }
}

/// Helper struct for printing the domain of an action with `format!` and `{}`.
///
/// This is similar to [`DisplayAction`] but only displays the domain associated with the action,
/// not the choice.
pub struct DisplayActionDomain<'a> {
    action: &'a Action,
    ir_instance: &'a ir::Function,
}

impl<'a> std::fmt::Debug for DisplayActionDomain<'a> {
    #[allow(unused_variables)]
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        let DisplayActionDomain { action, ir_instance } = self;
        match **action {
            {{~#each choices}}
            Action::{{to_type_name name}}({{>choice.arg_names}}domain) => {
                std::fmt::Debug::fmt(&domain, fmt)
            },
            {{~/each}}
        }
    }
}

impl<'a> std::fmt::Display for DisplayActionDomain<'a> {
    #[allow(unused_variables)]
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        let DisplayActionDomain { action, ir_instance } = self;
        match **action {
            {{~#each choices}}
            Action::{{to_type_name name}}({{>choice.arg_names}}domain) => {
                {{~#ifeq choice_def "Integer"}}
                write!(fmt, "{}", {
                    {{~#each arguments}}
                    let {{this.[0]}} = {{>set.item_getter this.[1] id=this.[0]}};
                    {{~/each}}
                    domain.display({{>value_type.univers value_type}})
                })
                {{~else}}
                write!(fmt, "{}", domain)
                {{~/ifeq}}
            },
            {{~/each}}
        }
    }
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

    /// Returns an object that implements [`Display`] for printing the action with its associated
    /// function.
    ///
    /// This is needed because actions only contain IDs referring to objects stored on the
    /// [`Function`] and hence can't be displayed properly without the [`Function`].
    ///
    /// [`Display`]: std::fmt::Display
    /// [`Function`]: crate::ir::Function
    pub fn display<'a>(&'a self, ir_instance: &'a ir::Function) -> DisplayAction<'a> {
        DisplayAction { action: self, ir_instance }
    }

    /// Returns an object that implements [`Display`] for printing an action's domain.
    ///
    /// This is useful to manipulate something that represents an action's domain for display or
    /// debugging, since the domains have different types for different actions.
    ///
    /// [`Display`]: std::fmt::Display
    pub fn display_domain<'a>(&'a self, ir_instance: &'a ir::Function) -> DisplayActionDomain<'a> {
        DisplayActionDomain { action: self, ir_instance }
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
