//! Prints the definition and manipulation of choices.
use crate::ir::{self, SetRef};
use crate::print;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens, quote};
use serde_derive::Serialize;

/// Prints the ids of the variables of a `ChoiceInstance`.
pub fn ids(choice_instance: &ir::ChoiceInstance, ctx: &print::Context) -> TokenStream {
    let choice = ctx.ir_desc.get_choice(&choice_instance.choice);
    let arg_sets = choice.arguments().sets();
    let ids = choice_instance
        .vars
        .iter()
        .zip_eq(arg_sets)
        .map(|(&var, set)| {
            // TODO(cleanup): use idents instead of variables in the context.
            let var = Ident::new(&ctx.var_name(var).to_string(), Span::call_site());
            print::set::ObjectId::from_object(&var.into(), set.def())
        });
    quote!(#(#ids,)*)
}

/// Restricts a choice to `value`. If `delayed` is true, actions are put in the
/// `actions` vector instead of being directly applied.
pub fn restrict(
    choice_instance: &ir::ChoiceInstance,
    value: &print::Value,
    delayed: bool,
    ctx: &print::Context,
) -> TokenStream {
    assert_eq!(
        &choice_instance.value_type(ctx.ir_desc).full_type(),
        value.value_type()
    );
    // TODO(span): keep the real span.
    let name = Ident::new(&choice_instance.choice, Span::call_site());
    let ids = ids(choice_instance, ctx);
    if delayed {
        quote!(actions.extend(#name::restrict_delayed(#ids ir_instance, store, #value)?);)
    } else {
        quote!(#name::restrict(#ids ir_instance, store, #value, diff)?;)
    }
}

// TODO(cleanup): use TokenStream insted of templates
use crate::ir::Adaptable;
use itertools::Itertools;
use crate::print::{ast, filter, value_set};

#[derive(Serialize)]
pub struct Ast<'a> {
    /// The name of the choice.
    name: &'a str,
    /// The documentation attached to the choice.
    doc: Option<&'a str>,
    /// The arguments for wich the choice is instantiated.
    arguments: Vec<(&'a str, ast::Set<'a>)>,
    /// The type of the values the choice can take.
    value_type: ast::ValueType,
    /// The type of the full value corresponding to 'value_type'
    full_value_type: ast::ValueType,
    /// The definition of the choice.
    choice_def: ChoiceDef,
    /// Indicates if hte choice is symmetric.
    is_symmetric: bool,
    /// Indicates if hte choice is antisymmetric.
    is_antisymmetric: bool,
    /// AST of the triggers.
    trigger_calls: Vec<TriggerCall<'a>>,
    /// Loop nest that iterates over the arguments of the choice.
    iteration_space: ast::LoopNest<'a>,
    /// AST of the restrict function on counters.
    restrict_counter: Option<RestrictCounter<'a>>,
    /// Actions to perform when the choice value is modified.
    on_change: Vec<OnChangeAction<'a>>,
    /// Actions to perform when initializing counters.
    filter_actions: Vec<FilterAction<'a>>,
    /// Ast for filters.
    filters: Vec<filter::Filter<'a>>,
    /// Compute Counter function AST.
    compute_counter: Option<ComputeCounter<'a>>,
}

impl<'a> Ast<'a> {
    pub fn new(choice: &'a ir::Choice, ir_desc: &'a ir::IrDesc) -> Self {
        let (is_symmetric, is_antisymmetric) = match *choice.arguments() {
            ir::ChoiceArguments::Plain { .. } => (false, false),
            ir::ChoiceArguments::Symmetric { inverse, .. } => (true, inverse),
        };
        let filter_actions = choice
            .filter_actions()
            .map(move |action| FilterAction::new(action, choice, ir_desc))
            .collect();
        let mut trigger_calls = Vec::new();
        let on_change = choice
            .on_change()
            .map(|action| {
                OnChangeAction::new(action, choice, ir_desc, &mut trigger_calls)
            })
            .collect();
        let filters = choice
            .filters()
            .enumerate()
            .map(|(id, f)| filter::Filter::new(f, id, choice, ir_desc))
            .collect();
        let ref ctx = ast::Context::new(ir_desc, choice, &[], &[]);
        let arguments = choice
            .arguments()
            .iter()
            .map(|(n, s)| (n as &str, ast::Set::new(s, ctx)))
            .collect();
        Ast {
            name: choice.name(),
            doc: choice.doc(),
            value_type: ast::ValueType::new(choice.value_type(), ctx),
            full_value_type: ast::ValueType::new(choice.value_type().full_type(), ctx),
            choice_def: ChoiceDef::new(choice.choice_def()),
            restrict_counter: RestrictCounter::new(choice, ir_desc),
            iteration_space: self::iteration_space(choice, ctx),
            compute_counter: ComputeCounter::new(choice, ir_desc),
            arguments,
            trigger_calls,
            is_symmetric,
            is_antisymmetric,
            on_change,
            filter_actions,
            filters,
        }
    }

    pub fn name(&self) -> &str {
        self.name
    }
}

/// Returns the iteration space associated to `choice`.
fn iteration_space<'a>(choice: &ir::Choice, ctx: &ast::Context<'a>) -> ast::LoopNest<'a> {
    match *choice.arguments() {
        ref args @ ir::ChoiceArguments::Plain { .. } => {
            let args = (0..args.len()).map(ir::Variable::Arg);
            ast::LoopNest::new(args, ctx, &mut vec![], false)
        }
        ir::ChoiceArguments::Symmetric { .. } => {
            ast::LoopNest::triangular(ir::Variable::Arg(0), ir::Variable::Arg(1), ctx)
        }
    }
}

#[derive(Serialize)]
enum ChoiceDef {
    Enum,
    Counter { kind: ir::CounterKind },
    Integer,
}

impl ChoiceDef {
    fn new(def: &ir::ChoiceDef) -> Self {
        match *def {
            ir::ChoiceDef::Enum(..) => ChoiceDef::Enum,
            ir::ChoiceDef::Counter { kind, .. } => ChoiceDef::Counter { kind },
            ir::ChoiceDef::Number { .. } => ChoiceDef::Integer,
        }
    }
}

#[derive(Serialize)]
struct TriggerCall<'a> {
    id: usize,
    arguments: Vec<(ast::Variable<'a>, ast::Set<'a>)>,
    code: String,
}

/// An action to run at filter initialization.
/// [template]: printed by the `filter_action` template.
#[derive(Serialize)]
struct FilterAction<'a> {
    constraints: Vec<ast::SetConstraint<'a>>,
    body: FilterCall<'a>,
}

impl<'a> FilterAction<'a> {
    fn new(
        action: &'a ir::FilterAction,
        choice: &'a ir::Choice,
        ir_desc: &'a ir::IrDesc,
    ) -> Self {
        let set = ast::Variable::with_name("values");
        let forall_vars = &action.filter.forall_vars;
        let ref ctx = ast::Context::new(ir_desc, choice, forall_vars, &[]);
        let conflicts = ast::Conflict::choice_args(choice, &ctx);
        let body = FilterCall::new(&action.filter, set, conflicts, 0, ctx);
        let constraints = ast::SetConstraint::new(&action.set_constraints, ctx);
        FilterAction { constraints, body }
    }
}

#[derive(Serialize)]
struct ComputeCounter<'a> {
    base: String,
    half: bool,
    nest: ast::LoopNest<'a>,
    body: String,
    op: String,
}

impl<'a> ComputeCounter<'a> {
    fn new(choice: &'a ir::Choice, ir_desc: &'a ir::IrDesc) -> Option<Self> {
        use crate::ir::ChoiceDef::Counter;
        let def = choice.choice_def();
        if let Counter {
            ref incr_iter,
            kind,
            ref value,
            ref incr,
            ref incr_condition,
            visibility,
            ref base,
        } = *def
        {
            let ctx = ast::Context::new(ir_desc, choice, incr_iter, &[]);
            let mut conflicts = ast::Conflict::choice_args(choice, &ctx);
            let forall_vars = (0..incr_iter.len()).map(ir::Variable::Forall);
            let nest_builder =
                ast::LoopNest::new(forall_vars, &ctx, &mut conflicts, false);
            let body = print::counter::compute_counter_body(
                value,
                incr,
                incr_condition,
                kind,
                visibility,
                &ctx,
            );
            Some(ComputeCounter {
                base: ast::code(base, &ctx),
                half: visibility == ir::CounterVisibility::NoMax,
                nest: nest_builder,
                body: body.to_string(),
                op: kind.to_string(),
            })
        } else {
            None
        }
    }
}

/// AST for an `OnChange` action.
/// [template]: printed by the `on_change` template.
#[derive(Serialize)]
struct OnChangeAction<'a> {
    constraints: Vec<ast::SetConstraint<'a>>,
    loop_nest: ast::LoopNest<'a>,
    action: ChoiceAction<'a>,
}

impl<'a> OnChangeAction<'a> {
    /// Generates the AST for an action to perform when `choice` is restricted.
    fn new(
        action: &'a ir::OnChangeAction,
        choice: &'a ir::Choice,
        ir_desc: &'a ir::IrDesc,
        trigger_calls: &mut Vec<TriggerCall<'a>>,
    ) -> Self {
        // TODO(cc_perf): if the other is symmetric -> need to iterate only on part of it.
        // Setup the context and variable names.
        let forall_vars = action.forall_vars.iter().chain(action.action.variables());
        let inputs = action.action.inputs();
        let ref ctx = ast::Context::new(ir_desc, choice, forall_vars, inputs);
        let mut conflicts = ast::Conflict::choice_args(choice, ctx);
        // Declare loop nests.
        let loop_nest = {
            let action_foralls = (0..action.forall_vars.len()).map(ir::Variable::Forall);
            ast::LoopNest::new(action_foralls, ctx, &mut conflicts, false)
        };
        // Build the body.
        let constraints = ast::SetConstraint::new(&action.set_constraints, ctx);
        let forall_offset = action.forall_vars.len();
        let action = ChoiceAction::new(
            &action.action,
            forall_offset,
            &action.set_constraints,
            conflicts,
            ctx,
            choice,
            trigger_calls,
        );
        OnChangeAction {
            constraints,
            loop_nest,
            action,
        }
    }
}

/// Ast for an `ir::ChoiceAction`.
#[derive(Serialize)]
enum ChoiceAction<'a> {
    FilterSelf,
    FilterRemote(RemoteFilterCall<'a>),
    IncrCounter {
        counter_name: &'a str,
        arguments: Vec<(ast::Variable<'a>, ast::Set<'a>)>,
        incr_condition: String,
        value_getter: String,
        counter_type: ast::ValueType,
        is_half: bool,
        zero: u32,
        min: String,
        max: String,
    },
    UpdateCounter {
        name: &'a str,
        incr_name: &'a str,
        incr_args: Vec<(ast::Variable<'a>, ast::Set<'a>)>,
        incr_condition: String,
        arguments: Vec<(ast::Variable<'a>, ast::Set<'a>)>,
        counter_type: ast::ValueType,
        incr_type: ast::ValueType,
        is_half: bool,
        zero: u32,
    },
    Trigger {
        call_id: usize,
        arguments: Vec<(ast::Variable<'a>, ast::Set<'a>)>,
        inputs: Vec<(ast::Variable<'a>, ast::ChoiceInstance<'a>)>,
        others_conditions: Vec<String>,
        self_condition: String,
    },
}

/// Prints an action performed by a choice.
impl<'a> ChoiceAction<'a> {
    fn new(
        action: &'a ir::ChoiceAction,
        forall_offset: usize,
        set_constraints: &'a ir::SetConstraints,
        conflicts: Vec<ast::Conflict<'a>>,
        ctx: &ast::Context<'a>,
        current_choice: &ir::Choice,
        trigger_calls: &mut Vec<TriggerCall<'a>>,
    ) -> Self {
        match action {
            ir::ChoiceAction::FilterSelf => ChoiceAction::FilterSelf,
            ir::ChoiceAction::RemoteFilter(remote_call) => {
                let call =
                    RemoteFilterCall::new(remote_call, conflicts, forall_offset, ctx);
                ChoiceAction::FilterRemote(call)
            }
            ir::ChoiceAction::IncrCounter {
                counter,
                value,
                incr_condition,
            } => {
                let counter_choice = ctx.ir_desc.get_choice(&counter.choice);
                let counter_def = counter_choice.choice_def();
                let arguments = ast::vars_with_sets(counter_choice, &counter.vars, ctx);
                let adaptator = ir::Adaptator::from_arguments(&counter.vars);
                let counter_type = counter_def.value_type().adapt(&adaptator);
                let value_getter = print::counter::increment_amount(value, true, ctx);
                let value: print::Value = value_getter.create_ident("value").into();
                let min = value.get_min(ctx);
                let max = value.get_max(ctx);
                if let ir::ChoiceDef::Counter {
                    kind, visibility, ..
                } = *counter_def
                {
                    ChoiceAction::IncrCounter {
                        counter_name: &counter.choice,
                        incr_condition: value_set::print(incr_condition, ctx).to_string(),
                        counter_type: ast::ValueType::new(counter_type, ctx),
                        arguments,
                        value_getter: value_getter.into_token_stream().to_string(),
                        is_half: visibility == ir::CounterVisibility::NoMax,
                        zero: kind.zero(),
                        min: min.into_token_stream().to_string(),
                        max: max.into_token_stream().to_string(),
                    }
                } else {
                    panic!()
                }
            }
            ir::ChoiceAction::UpdateCounter {
                counter,
                incr,
                incr_condition,
            } => {
                let counter_choice = ctx.ir_desc.get_choice(&counter.choice);
                let counter_def = counter_choice.choice_def();
                let incr_choice = ctx.ir_desc.get_choice(&incr.choice);
                let arguments = ast::vars_with_sets(counter_choice, &counter.vars, ctx);
                let incr_args = ast::vars_with_sets(incr_choice, &incr.vars, ctx);
                let adaptator = ir::Adaptator::from_arguments(&counter.vars);
                let counter_type = counter_def.value_type().adapt(&adaptator);
                if let ir::ChoiceDef::Counter {
                    kind, visibility, ..
                } = *counter_def
                {
                    ChoiceAction::UpdateCounter {
                        name: &counter.choice,
                        incr_name: &incr.choice,
                        incr_condition: value_set::print(incr_condition, ctx).to_string(),
                        is_half: visibility == ir::CounterVisibility::NoMax,
                        zero: kind.zero(),
                        counter_type: ast::ValueType::new(counter_type, ctx),
                        incr_type: ast::ValueType::new(current_choice.value_type(), ctx),
                        incr_args,
                        arguments,
                    }
                } else {
                    panic!()
                }
            }
            ir::ChoiceAction::Trigger {
                id,
                condition,
                code,
                inverse_self_cond,
            } => {
                let others_conditions = condition
                    .others_conditions
                    .iter()
                    .map(|c| filter::condition(c, ctx).to_string())
                    .collect();
                let inputs = condition
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(pos, input)| {
                        (ctx.input_name(pos), ast::ChoiceInstance::new(input, ctx))
                    })
                    .collect();
                let arguments = (0..current_choice.arguments().len())
                    .map(ir::Variable::Arg)
                    .chain((0..forall_offset).map(ir::Variable::Forall))
                    .map(|v| {
                        let (name, set) = ctx.var_def(v);
                        (
                            name,
                            ast::Set::new(
                                set_constraints.find_set(v).unwrap_or(set),
                                ctx,
                            ),
                        )
                    })
                    .collect_vec();
                let code = ast::code(code, ctx);
                let call_id = trigger_calls.len();
                let mut self_condition = condition.self_condition.clone();
                if *inverse_self_cond {
                    self_condition.inverse(ctx.ir_desc);
                }
                let call = TriggerCall {
                    id: *id,
                    code,
                    arguments: arguments.clone(),
                };
                trigger_calls.push(call);
                ChoiceAction::Trigger {
                    call_id,
                    others_conditions,
                    inputs,
                    arguments,
                    self_condition: value_set::print(&self_condition, ctx).to_string(),
                }
            }
        }
    }
}

/// AST for an `ir::RemoteFilterCall`.
#[derive(Serialize)]
pub struct RemoteFilterCall<'a> {
    choice: &'a str,
    is_symmetric: bool,
    filter_call: FilterCall<'a>,
    choice_full_type: ast::ValueType,
    arguments: Vec<(ast::Variable<'a>, ast::Set<'a>)>,
}

impl<'a> RemoteFilterCall<'a> {
    pub fn new(
        remote_call: &'a ir::RemoteFilterCall,
        conflicts: Vec<ast::Conflict<'a>>,
        forall_offset: usize,
        ctx: &ast::Context<'a>,
    ) -> Self {
        let set = ast::Variable::with_name("values");
        let choice = ctx.ir_desc.get_choice(&remote_call.choice.choice);
        let filter_call =
            FilterCall::new(&remote_call.filter, set, conflicts, forall_offset, ctx);
        let arguments = ast::vars_with_sets(choice, &remote_call.choice.vars, ctx);
        let adaptator = ir::Adaptator::from_arguments(&remote_call.choice.vars);
        let full_type = choice.value_type().full_type().adapt(&adaptator);
        RemoteFilterCall {
            is_symmetric: choice.arguments().is_symmetric(),
            choice: choice.name(),
            choice_full_type: ast::ValueType::new(full_type, ctx),
            filter_call,
            arguments,
        }
    }
}

/// AST for an `ir::FilterCall`.
#[derive(Serialize)]
struct FilterCall<'a> {
    loop_nest: ast::LoopNest<'a>,
    filter_ref: FilterRef<'a>,
}

impl<'a> FilterCall<'a> {
    fn new(
        filter_call: &'a ir::FilterCall,
        value_var: ast::Variable<'a>,
        mut conflicts: Vec<ast::Conflict<'a>>,
        forall_offset: usize,
        ctx: &ast::Context<'a>,
    ) -> Self {
        let num_foralls = filter_call.forall_vars.len();
        let foralls = (0..num_foralls).map(|i| ir::Variable::Forall(i + forall_offset));
        let filter_ref = FilterRef::new(&filter_call.filter_ref, value_var.clone(), ctx);
        let loop_nest = ast::LoopNest::new(foralls, ctx, &mut conflicts, false);
        FilterCall {
            loop_nest,
            filter_ref,
        }
    }
}

/// AST for an `ir::FilterRef`.
#[derive(Serialize)]
enum FilterRef<'a> {
    Inline {
        rules: Vec<filter::Rule<'a>>,
    },
    Call {
        choice: &'a str,
        id: usize,
        arguments: Vec<(ast::Variable<'a>,)>,
        value_var: ast::Variable<'a>,
    },
}

impl<'a> FilterRef<'a> {
    fn new(
        filter_ref: &'a ir::FilterRef,
        value_var: ast::Variable<'a>,
        ctx: &ast::Context<'a>,
    ) -> Self {
        match filter_ref {
            ir::FilterRef::Inline(rules) => {
                let rules = rules
                    .iter()
                    .map(|r| filter::Rule::new(value_var.clone(), r, ctx))
                    .collect();
                FilterRef::Inline { rules }
            }
            ir::FilterRef::Function { choice, id, args } => {
                let arguments = args.iter().map(|&v| (ctx.var_name(v),)).collect();
                FilterRef::Call {
                    choice,
                    id: *id,
                    arguments,
                    value_var,
                }
            }
        }
    }
}

#[derive(Serialize)]
struct RestrictCounter<'a> {
    incr: ast::ChoiceInstance<'a>,
    incr_amount: String,
    incr_iter: ast::LoopNest<'a>,
    incr_type: ast::ValueType,
    incr_condition: String,
    is_half: bool,
    op: String,
    neg_op: &'static str,
    min: String,
    max: String,
    restrict_amount: String,
    restrict_amount_delayed: String,
}

impl<'a> RestrictCounter<'a> {
    fn new(choice: &'a ir::Choice, ir_desc: &'a ir::IrDesc) -> Option<Self> {
        use crate::ir::ChoiceDef::Counter;
        let def = choice.choice_def();
        if let Counter {
            ref incr_iter,
            kind,
            ref value,
            ref incr,
            ref incr_condition,
            visibility,
            ..
        } = *def
        {
            let ctx = ast::Context::new(ir_desc, choice, incr_iter, &[]);
            let mut conflicts = ast::Conflict::choice_args(choice, &ctx);
            let forall_vars = (0..incr_iter.len()).map(ir::Variable::Forall);
            let incr_iter = ast::LoopNest::new(forall_vars, &ctx, &mut conflicts, false);
            let incr = ast::ChoiceInstance::new(incr, &ctx);
            let incr_amount_getter = print::counter::increment_amount(value, false, &ctx);
            let incr_amount: print::Value =
                incr_amount_getter.create_ident("incr_amount").into();
            let min_incr_amount = incr_amount.get_min(&ctx);
            let max_incr_amount = incr_amount.get_max(&ctx);
            let restrict_amount = if let ir::CounterVal::Choice(choice) = value {
                Some(print::counter::restrict_incr_amount(
                    choice,
                    &incr_amount,
                    kind,
                    false,
                    &ctx,
                ))
            } else {
                None
            };
            let restrict_amount_delayed = if let ir::CounterVal::Choice(choice) = value {
                Some(print::counter::restrict_incr_amount(
                    choice,
                    &incr_amount,
                    kind,
                    true,
                    &ctx,
                ))
            } else {
                None
            };
            Some(RestrictCounter {
                incr,
                incr_iter,
                incr_amount: incr_amount_getter.into_token_stream().to_string(),
                incr_type: ast::ValueType::new(incr_condition.t(), &ctx),
                incr_condition: value_set::print(&incr_condition, &ctx).to_string(),
                is_half: visibility == ir::CounterVisibility::NoMax,
                op: kind.to_string(),
                neg_op: match kind {
                    ir::CounterKind::Add => "-",
                    ir::CounterKind::Mul => "/",
                },
                min: min_incr_amount.into_token_stream().to_string(),
                max: max_incr_amount.into_token_stream().to_string(),
                restrict_amount: restrict_amount.into_token_stream().to_string(),
                restrict_amount_delayed: restrict_amount_delayed
                    .into_token_stream()
                    .to_string(),
            })
        } else {
            None
        }
    }
}
