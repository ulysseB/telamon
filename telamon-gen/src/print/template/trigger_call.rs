for ({{>choice.arg_names}}) in trigger_{{call_id}} {
    #[allow(dead_code)]
    const DELAYED: bool = {{#if delayed}}true{{else}}false{{/if}};
    {{#if make_mut}}let ir_instance = Arc::make_mut(ir_instance);{{/if}}
    trace!("running trigger file {} line {}", file!(), line!());
    let (new_objs, new_actions): (NewObjs, Vec<Action>) = {{code}}?;
    actions.extend(new_actions);
    actions.extend(process_lowering(ir_instance, store, &new_objs, diff)?);
}
