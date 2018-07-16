//! Prints the definition of a choice.
use ir;
use ir::Adaptable;
use itertools::Itertools;
use print::{ast, filter, value_set};

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
        let filter_actions = choice.filter_actions()
            .map(move |action| FilterAction::new(action, choice, ir_desc)).collect();
        let mut trigger_calls = Vec::new();
        let on_change = choice.on_change().map(|action| {
            OnChangeAction::new(action, choice, ir_desc, &mut trigger_calls)
        }).collect();
        let filters = choice.filters().enumerate()
            .map(|(id, f)| filter::Filter::new(f, id, choice, ir_desc)).collect();
        let ref ctx = ast::Context::new(ir_desc, choice, &[], &[]);
        let arguments = choice.arguments().iter()
            .map(|(n, s)| (n as &str, ast::Set::new(s, ctx))).collect();
        Ast {
            name: choice.name(),
            doc: choice.doc(),
            value_type: ast::ValueType::new(choice.value_type(), ctx),
            full_value_type: ast::ValueType::new(choice.value_type().full_type(), ctx),
            choice_def: ChoiceDef::new(choice.choice_def()),
            restrict_counter: RestrictCounter::new(choice, ir_desc),
            iteration_space: self::iteration_space(ctx),
            compute_counter: ComputeCounter::new(choice, ir_desc),
            arguments, trigger_calls, is_symmetric, is_antisymmetric,
            on_change, filter_actions, filters
        }
    }

    pub fn name(&self) -> &str { self.name }
}

/// Returns the iteration space associated to a choice.
fn iteration_space<'a>(ctx: &ast::Context<'a>) -> ast::LoopNest<'a> {
    match *ctx.choice.arguments() {
        ref args @ ir::ChoiceArguments::Plain { .. } => {
            let args = (0..args.len()).map(ir::Variable::Arg);
            ast::LoopNest::new(args, ctx, &mut vec![], false)
        },
        ir::ChoiceArguments::Symmetric { .. } => {
            ast::LoopNest::triangular(ir::Variable::Arg(0), ir::Variable::Arg(1), ctx)
        }
    }
}

#[derive(Serialize)]
enum ChoiceDef { Enum, Counter { kind: ir::CounterKind }, Integer }

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
    fn new(action: &'a ir::FilterAction, choice: &'a ir::Choice,
           ir_desc: &'a ir::IrDesc) -> Self {
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
    nest: ast::LoopNest<'a>,
    incr: ast::ChoiceInstance<'a>,
    incr_condition: String,
    value: CounterValue<'a>,
    op: String,
    half: bool,
}

impl<'a> ComputeCounter<'a> {
    fn new(choice: &'a ir::Choice, ir_desc: &'a ir::IrDesc) -> Option<Self> {
        use ir::ChoiceDef::Counter;
        let def = choice.choice_def();
        if let Counter {
            ref incr_iter, kind, ref value, ref incr, ref incr_condition, visibility,
            ref base
        } = *def {
            let ctx = ast::Context::new(ir_desc, choice, incr_iter, &[]);
            let mut conflicts = ast::Conflict::choice_args(choice, &ctx);
            let forall_vars = (0..incr_iter.len()).map(ir::Variable::Forall);
            let nest_builder = ast::LoopNest::new(
                forall_vars, &ctx, &mut conflicts, false);
            Some(ComputeCounter {
                base: ast::code(base, &ctx),
                incr: ast::ChoiceInstance::new(incr, &ctx),
                incr_condition: value_set::print(incr_condition, &ctx),
                nest: nest_builder,
                value: CounterValue::new(value, &ctx),
                op: kind.to_string(),
                half: visibility == ir::CounterVisibility::NoMax,
            })
        } else { None }
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
    fn new(action: &'a ir::OnChangeAction, choice: &'a ir::Choice,
           ir_desc: &'a ir::IrDesc, trigger_calls: &mut Vec<TriggerCall<'a>>) -> Self {
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
        let action = ChoiceAction::new(&action.action, forall_offset,
            &action.set_constraints, conflicts, ctx, trigger_calls);
        OnChangeAction { constraints, loop_nest, action }
    }
}

/// Ast for an `ir::ChoiceAction`.
#[derive(Serialize)]
enum ChoiceAction<'a> {
    FilterSelf,
    Filter {
        choice: &'a str,
        is_symmetric: bool,
        filter_call: FilterCall<'a>,
        choice_full_type: ast::ValueType,
        arguments: Vec<(ast::Variable<'a>, ast::Set<'a>)>,
    },
    IncrCounter {
        counter_name: &'a str,
        arguments: Vec<(ast::Variable<'a>, ast::Set<'a>)>,
        incr_condition: String,
        value: CounterValue<'a>,
        counter_type: ast::ValueType,
        is_half: bool,
        zero: u32,
    },
    UpdateCounter {
        name: &'a str,
        incr_name: &'a str,
        incr_args: Vec<(ast::Variable<'a>, ast::Set<'a>)>,
        incr_condition: String,
        arguments: Vec<(ast::Variable<'a>, ast::Set<'a>)>,
        value_type: ast::ValueType,
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
    fn new(action: &'a ir::ChoiceAction,
           forall_offset: usize,
           set_constraints: &'a ir::SetConstraints,
           conflicts: Vec<ast::Conflict<'a>>,
           ctx: &ast::Context<'a>,
           trigger_calls: &mut Vec<TriggerCall<'a>>) -> Self {
        match action {
            ir::ChoiceAction::FilterSelf => ChoiceAction::FilterSelf,
            ir::ChoiceAction::Filter { choice: choice_instance, filter } => {
                let set = ast::Variable::with_name("values");
                let choice = ctx.ir_desc.get_choice(&choice_instance.choice);
                let filter_call = FilterCall::new(filter, set, conflicts, forall_offset, ctx);
                let arguments = ast::vars_with_sets(choice, &choice_instance.vars, ctx);
                let adaptator = ir::Adaptator::from_arguments(&choice_instance.vars);
                let full_type = choice.value_type().full_type().adapt(&adaptator);
                ChoiceAction::Filter {
                    is_symmetric: choice.arguments().is_symmetric(),
                    choice: choice.name(),
                    choice_full_type: ast::ValueType::new(full_type, ctx),
                    filter_call, arguments,
                }
            },
            ir::ChoiceAction::IncrCounter { counter, value, incr_condition } => {
                let counter_choice = ctx.ir_desc.get_choice(&counter.choice);
                let counter_def = counter_choice.choice_def();
                let arguments = ast::vars_with_sets(counter_choice, &counter.vars, ctx);
                let adaptator = ir::Adaptator::from_arguments(&counter.vars);
                let counter_type = counter_def.value_type().adapt(&adaptator);
                if let ir::ChoiceDef::Counter { kind, visibility, .. } = *counter_def {
                    ChoiceAction::IncrCounter {
                        counter_name: &counter.choice,
                        incr_condition: value_set::print(incr_condition, ctx),
                        counter_type: ast::ValueType::new(counter_type, ctx),
                        arguments,
                        value: CounterValue::new(value, ctx),
                        is_half: visibility == ir::CounterVisibility::NoMax,
                        zero: kind.zero(),
                    }
                } else { panic!() }
            },
            ir::ChoiceAction::UpdateCounter { counter, incr, incr_condition } => {
                let counter_choice = ctx.ir_desc.get_choice(&counter.choice);
                let counter_def = counter_choice.choice_def();
                let incr_choice = ctx.ir_desc.get_choice(&incr.choice);
                let arguments = ast::vars_with_sets(counter_choice, &counter.vars, ctx);
                let incr_args = ast::vars_with_sets(incr_choice, &incr.vars, ctx);
                let adaptator = ir::Adaptator::from_arguments(&counter.vars);
                let value_type = counter_def.value_type().adapt(&adaptator);
                if let ir::ChoiceDef::Counter { kind, visibility, .. } = *counter_def {
                    ChoiceAction::UpdateCounter {
                        name: &counter.choice,
                        incr_name: &incr.choice,
                        incr_condition: value_set::print(incr_condition, ctx),
                        is_half: visibility == ir::CounterVisibility::NoMax,
                        zero: kind.zero(),
                        value_type: ast::ValueType::new(value_type, ctx),
                        incr_args, arguments,
                    }
                } else { panic!() }
            },
            ir::ChoiceAction::Trigger { id, condition, code, inverse_self_cond } => {
                let others_conditions = condition.others_conditions.iter()
                    .map(|c| filter::condition(c, ctx)).collect();
                let inputs = condition.inputs.iter().enumerate().map(|(pos, input)| {
                    (ctx.input_name(pos), ast::ChoiceInstance::new(input, ctx))
                }).collect();
                let arguments = (0..ctx.choice.arguments().len()).map(ir::Variable::Arg)
                    .chain((0..forall_offset).map(ir::Variable::Forall))
                    .map(|v| {
                        let (name, set) = ctx.var_def(v);
                        (name, ast::Set::new(set_constraints.find_set(v).unwrap_or(set), ctx))
                    }).collect_vec();
                let code = ast::code(code, ctx);
                let call_id = trigger_calls.len();
                let mut self_condition = condition.self_condition.clone();
                if *inverse_self_cond { self_condition.inverse(ctx.ir_desc); }
                let call = TriggerCall { id: *id, code, arguments: arguments.clone() };
                trigger_calls.push(call);
                ChoiceAction::Trigger {
                    call_id, others_conditions, inputs, arguments,
                    self_condition: value_set::print(&self_condition, ctx),
                }
            },
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
    fn new(filter_call: &'a ir::FilterCall,
           value_var: ast::Variable<'a>,
           mut conflicts: Vec<ast::Conflict<'a>>,
           forall_offset: usize,
           ctx: &ast::Context<'a>) -> Self {
        let num_foralls = filter_call.forall_vars.len();
        let foralls = (0..num_foralls).map(|i| ir::Variable::Forall(i+forall_offset));
        let filter_ref = FilterRef::new(&filter_call.filter_ref, value_var.clone(), ctx);
        let loop_nest = ast::LoopNest::new(foralls, ctx, &mut conflicts, false);
        FilterCall { loop_nest, filter_ref }
    }
}

/// AST for an `ir::FilterRef`.
#[derive(Serialize)]
enum FilterRef<'a> {
    Inline { rules: Vec<filter::Rule<'a>> },
    Call {
        choice: &'a str,
        id: usize,
        arguments: Vec<(ast::Variable<'a>,)>,
        value_var: ast::Variable<'a>
    },
}

impl<'a> FilterRef<'a> {
    fn new(filter_ref: &'a ir::FilterRef, value_var: ast::Variable<'a>,
                  ctx: &ast::Context<'a>) -> Self {
        match *filter_ref {
            ir::FilterRef::Inline(ref rules) => {
                let rules = rules.iter().map(|r| {
                    filter::Rule::new(value_var.clone(), r, ctx)
                }).collect();
                FilterRef::Inline { rules }
            },
            ir::FilterRef::Local { id, ref args } => {
                let arguments = args.iter().map(|&v| (ctx.var_name(v),)).collect();
                FilterRef::Call { choice: "self", id, arguments, value_var }
            },
            ir::FilterRef::Remote { ref choice, id, ref args } => {
                let arguments = args.iter().map(|&v| (ctx.var_name(v),)).collect();
                FilterRef::Call { choice, id, arguments, value_var }
            },
        }
    }
}

#[derive(Serialize)]
struct RestrictCounter<'a> {
    amount: CounterValue<'a>,
    incr: ast::ChoiceInstance<'a>,
    incr_iter: ast::LoopNest<'a>,
    incr_type: ast::ValueType,
    incr_condition: String,
    is_half: bool,
    op: String,
    neg_op: &'static str,
}

impl<'a> RestrictCounter<'a> {
    fn new(choice: &'a ir::Choice, ir_desc: &'a ir::IrDesc) -> Option<Self> {
        use ir::ChoiceDef::Counter;
        let def = choice.choice_def();
        if let Counter {
            ref incr_iter, kind, ref value, ref incr, ref incr_condition, visibility, ..
        } = *def {
            let ctx = ast::Context::new(ir_desc, choice, incr_iter, &[]);
            let mut conflicts = ast::Conflict::choice_args(choice, &ctx);
            let forall_vars = (0..incr_iter.len()).map(ir::Variable::Forall);
            let incr_iter = ast::LoopNest::new(forall_vars, &ctx, &mut conflicts, false);
            let incr = ast::ChoiceInstance::new(incr, &ctx);
            Some(RestrictCounter {
                amount: CounterValue::new(value, &ctx),
                incr, incr_iter,
                incr_type: ast::ValueType::new(incr_condition.t(), &ctx),
                incr_condition: value_set::print(&incr_condition, &ctx),
                is_half: visibility == ir::CounterVisibility::NoMax,
                op: kind.to_string(),
                neg_op: match kind {
                    ir::CounterKind::Add => "-",
                    ir::CounterKind::Mul => "/",
                },
            })
        } else { None }
    }
}

#[derive(Serialize)]
pub enum CounterValue<'a> {
    Code(String),
    Choice {
        name: &'a str,
        arguments:  Vec<(ast::Variable<'a>, ast::Set<'a>)>,
        full_type: ast::ValueType,
    }
}

impl<'a> CounterValue<'a> {
    pub fn new(value: &'a ir::CounterVal, ctx: &ast::Context<'a>) -> Self {
        match *value {
            ir::CounterVal::Code(ref code) => CounterValue::Code(ast::code(code, ctx)),
            ir::CounterVal::Choice(ref counter) => {
                let choice = ctx.ir_desc.get_choice(&counter.choice);
                let arguments = ast::vars_with_sets(choice, &counter.vars, ctx);
                let full_type = choice.choice_def().value_type().full_type();
                let adaptator = ir::Adaptator::from_arguments(&counter.vars);
                let full_type = ast::ValueType::new(full_type.adapt(&adaptator), ctx);
                CounterValue::Choice { arguments, full_type, name: choice.name() }
            },
        }
    }
}
