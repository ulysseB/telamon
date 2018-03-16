{{~#*inline "args_decl"~}}
    {{#each arguments~}}, mut {{this.[0]}}: {{this.[1].def.keys.IdType}}{{/each}}
{{~/inline~}}

{{~#*inline "args"}}({{#each arguments}}{{this.[0]}},{{/each}}){{/inline~}}

{{~#*inline "restrict_op"~}}
    {{~#ifeq choice_def "Enum"}}restrict{{/ifeq~}}
    {{~#with choice_def.Counter~}}
        {{~#ifeq kind "Add"}}apply_diff_add{{/ifeq~}}
        {{~#ifeq kind "Mul"}}apply_diff_mul{{/ifeq~}}
    {{~/with~}}
{{~/inline}}

/// Returns the domain of {name} for the given arguments.
#[allow(unused_mut)]
pub fn get_{{name}}(&self{{>args_decl}}) -> {{value_type}} {
    {{#if is_symmetric~}}
        if {{arguments.[0].[0]}} > {{arguments.[1].[0]}} {
            std::mem::swap(&mut {{arguments.[0].[0]}}, &mut {{arguments.[1].[0]}});
            self.{{name}}[&{{>args}}]{{#if is_antisymmetric}}.inverse(){{/if}}
        } else {
    {{~/if}} self.{{name}}[&{{>args}}] {{#if is_symmetric}} } {{/if}}
}

/// Returns the domain of {name} for the given arguments. If the domain has been restricted
/// but the change not yet applied, returns the old value.
#[allow(unused_mut)]
pub fn get_old_{{name}}(&self{{>args_decl}},diff: &DomainDiff) -> {{value_type}} {
    {{#if is_symmetric~}}
        if {{arguments.[0].[0]}} > {{arguments.[1].[0]}} {
            std::mem::swap(&mut {{arguments.[0].[0]}}, &mut {{arguments.[1].[0]}});
            diff.{{name}}.get(&{{>args}}).map(|&(old, _)| old)
                .unwrap_or_else(|| self.{{name}}[&{{>args}}])
                {{#if is_antisymmetric}}.inverse(){{/if}}
        } else {
    {{~/if~}}
        diff.{{name}}.get(&{{>args}}).map(|&(old, _)| old)
            .unwrap_or_else(|| self.{{name}}[&{{>args}}])
    {{~#if is_symmetric}} } {{/if}}
}

/// Sets the domain of {name} for the given arguments.
#[allow(unused_mut)]
pub fn set_{{name}}(&mut self{{>args_decl}}, mut value: {{value_type}}) {
    {{#if is_symmetric~}}
        if {{arguments.[0].[0]}} > {{arguments.[1].[0]}} {
            std::mem::swap(&mut {{arguments.[0].[0]}}, &mut {{arguments.[1].[0]}});
            {{~#if is_antisymmetric}}value = value.inverse();{{/if}}
        }
    {{~/if}}
    debug!("set {{name}}{:?} to {:?}", {{>args}}, value);
    *unwrap!(Arc::make_mut(&mut self.{{name}}).get_mut(&{{>args}})) = value;
}

/// Restricts the domain of {name} for the given arguments. Put the old value in `diff`
/// and indicates if the new domain is failed.
#[allow(unused_mut)]
fn restrict_{{name}}(&mut self{{>args_decl}}, mut value: {{value_type}},
                         diff: &mut DomainDiff) -> Result<(), ()> {
    {{#if is_symmetric~}}
        if {{arguments.[0].[0]}} > {{arguments.[1].[0]}} {
            std::mem::swap(&mut {{arguments.[0].[0]}}, &mut {{arguments.[1].[0]}});
            {{~#if is_antisymmetric}}value = value.inverse();{{/if}}
        }
    {{~/if}}
    let mut ptr = unwrap!(Arc::make_mut(&mut self.{{name}}).get_mut(&{{>args}}));
    let old = *ptr;
    ptr.{{>restrict_op}}(value);
    if old != *ptr {
        debug!("restrict {{name}}{:?} to {:?}", {{>args}}, *ptr);
        diff.{{name}}.entry({{>args}}).or_insert((old, *ptr)).1 = *ptr;
    }
    if ptr.is_failed() { Err(()) } else { Ok(()) }
}

{{#if compute_counter~}}
/// Updates a counter by changing the value of an increment.
#[allow(unused_mut)]
fn update_{{name}}(&mut self{{>args_decl}}, old_incr: {{value_type}},
                   new_incr: {{value_type}}, diff: &mut DomainDiff) -> Result<(), ()> {
    {{#if is_symmetric~}}
        if {{arguments.[0].[0]}} > {{arguments.[1].[0]}} {
            std::mem::swap(&mut {{arguments.[0].[0]}}, &mut {{arguments.[1].[0]}});
        }
    {{~/if}}
    let mut ptr = unwrap!(Arc::make_mut(&mut self.{{name}}).get_mut(&{{>args}}));
    let old = *ptr;
    {{#*inline "op_name"~}}
        {{#ifeq compute_counter.op "+"}}add{{else}}mul{{/ifeq~}}
    {{/inline~}}
    ptr.sub_{{>op_name}}(old_incr);
    ptr.add_{{>op_name}}(new_incr);
    if old != *ptr {
        debug!("update {{name}}{:?} to {:?}", {{>args}}, *ptr);
        diff.{{name}}.entry({{>args}}).or_insert((old, *ptr)).1 = *ptr;
    }
    if ptr.is_failed() { Err(()) } else { Ok(()) }
}
{{/if}}
