use super::ir;
#[allow(unused_imports)]
use super::ir::prelude::*;
#[allow(unused_imports)]
use std;
use std::sync::Arc;
use num;
use itertools::Itertools;
#[allow(unused_imports)]
use utils::*;

{{>store}}

{{#each choices}}
    {{>choice_def this}}
{{/each}}

{{enums}}

{{>choices this}}

{{>actions this}}

/// Propagate the changes stored in `diff`.
pub fn propagate_changes(diff: &mut DomainDiff, ir_instance: &mut Arc<ir::Function>,
                     store: &mut DomainStore) -> Result<(), ()> {
    {{~#each choices}}
        while let Some((({{#each arguments}}{{this.[0]}}, {{/each}}), old, new)) =
                diff.pop_{{name}}_diff() {
            debug!("propagating {{name}}{:?} {:?} -> {:?}",
                   ({{>choice.arg_names this}}), old, new);
            {{name}}::on_change(old, new,
                {{~>choice.arg_names this}}ir_instance, store, diff)?;
        }
    {{~/each}}
    Ok(())
}

/// Applies a set of decisions to the domain and propagate the changes.
pub fn apply_decisions(actions: Vec<Action>, ir_instance: &mut Arc<ir::Function>,
                   domain: &mut DomainStore) -> Result<(), ()> {
    let mut diff = DomainDiff::default();
    for action in actions { apply_action(action, domain, &mut diff)?; }
    while !diff.is_empty() { propagate_changes(&mut diff, ir_instance, domain)?; }
    Ok(())
}

/// Update the domain after a lowering.
#[cfg(feature="gen_applicators")]
fn process_lowering(ir_instance: &mut ir::Function,
                    domain: &mut DomainStore,
                    new_objs: &ir::NewObjs,
                    diff: &mut DomainDiff) -> Result<Vec<Action>, ()> {
    let mut actions = Vec::new();
    domain.alloc(ir_instance, &new_objs);
    actions.extend(init_domain_partial(domain, ir_instance, &new_objs, diff)?);
    Ok(actions)
}
#[cfg(not(feature="gen_applicators"))]
use super::process_lowering;



/// Initializes the `DomainStore` with available choices for each decision.
#[allow(unused_variables, unused_mut)]
pub fn init_domain(store: &mut DomainStore,
                   ir_instance: &mut ir::Function) -> Result<Vec<Action>, ()> {
    trace!("called init_domain from file {}", file!());
    // Run all the filters once.
    let ref mut diff = DomainDiff::default(); // Pass an empty diff to propagate and triggers.
    let mut unused_diff = DomainDiff::default();
    {{#each choices~}}
        {{#>loop_nest iteration_space~}}
            {{>run_filters this}}
        {{/loop_nest~}}
    {{/each~}}
    {{store.filter_all}}
    // Propagate the filters where necessary.
    let mut actions: Vec<Action> = Vec::new();
    {{#each triggers~}}
        let mut trigger_{{id}} = Vec::new();
        {{#>loop_nest loop_nest}}
            if check_trigger_{{id}}({{>choice.arg_names}}ir_instance, store, diff) {
                trigger_{{id}}.push(({{>choice.arg_ids}}));
            }
        {{/loop_nest}}
    {{/each}}
    {{#each choices~}}
        {{#>loop_nest iteration_space}}{{>propagate this}}{{/loop_nest~}}
    {{/each~}}
    // Propagate triggers.
    {{#each triggers}}{{>trigger_call call_id=id delayed=true}}{{/each~}}
    Ok(actions)
}

/// Initializes the part of the `DomainStore` allocated for the given objects with available
/// choices for each decision.
#[allow(unused_variables, unused_mut)]
pub fn init_domain_partial(store: &mut DomainStore,
                       ir_instance: &mut ir::Function,
                       new_objs: &ir::NewObjs,
                       diff: &mut DomainDiff) -> Result<Vec<Action>, ()> {
    let mut unused_diff = DomainDiff::default();
    // Disable new increments of existing counters.
    {{#each incr_iterators~}}
        {{#>iter_new_objects iter~}}
            {{>disable_increment}}
        {{/iter_new_objects~}}
    {{/each~}}
    // Call filters.
    {{#each partial_iterators~}}
        {{#>iter_new_objects this.[0]~}}
            {{>run_filters this.[1].choice arg_names=this.[1].arg_names}}
        {{/iter_new_objects~}}
    {{/each~}}
    // Propagate decisions that are not already propagted.
    let mut actions: Vec<Action> = Vec::new();
    {{#each partial_iterators~}}
        {{#>iter_new_objects this.[0]~}}
                {{>propagate this.[1].choice arg_names=this.[1].arg_names}}
        {{/iter_new_objects~}}
    {{/each~}}
    // Take new increment on existing counters into account.
    {{#each incr_iterators~}}
        {{#>iter_new_objects iter~}}
            {{>account_new_incrs}}
        {{/iter_new_objects~}}
    {{/each~}}
    // Check if we should call the new triggers.
    {{#each triggers~}}
        let mut trigger_{{id}} = Vec::new();
        {{#each partial_iterators}}
            {{#>iter_new_objects this.[0]}}
    if check_trigger_{{../id}}(
        {{~>choice.arg_names ../../this arg_names=this.[1]~}}ir_instance, store, diff)
    {
        trigger_{{../id}}.push(({{>choice.arg_ids ../../this arg_names=this.[1]}}));
    }
            {{/iter_new_objects~}}
        {{/each~}}
    {{/each~}}
    // Propagate triggers.
    {{#each triggers}}{{>trigger_call call_id=id delayed=true}}{{/each~}}
    Ok(actions)
}

{{#each triggers~}}
    {{>trigger_check}}
{{/each~}}

// TODO(cleanup): generate (IrInstance, Domain) pair here.
{{runtime}}
