//! Prints the domain store definition.
use proc_macro2::{Ident, Span};
use print;
use quote::ToTokens;

/// Returns the name of the getter method for `choice`. If `get_old` is true, the method
/// will only take into account decisions that have been propagated.
pub fn getter_name(choice: &str, get_old: bool) -> Ident {
    let name = if get_old {
        format!("get_old_{}", choice)
    } else {
        format!("get_{}", choice)
    };
    Ident::new(&name, Span::call_site())
}

//TODO(cleanup): use TokenStream instead of templates
use ir;
use ir::SetRef;
use itertools::Itertools;
use print::value_set;
use print::ast::{self, Variable, LoopNest};
use print::choice::Ast as ChoiceAst;
use std::iter;
use utils::*;


/// Returns the partial iterators over the choices.
// TODO(cleanup): do not use ChoiceAst so we can directly iterate on IR choices.
pub fn partial_iterators<'a>(choices: &'a [ChoiceAst<'a>], ir_desc: &'a ir::IrDesc)
        -> Vec<(PartialIterator<'a>, NewChoice<'a>)> {
    choices.iter().flat_map(|choice_ast| {
        let choice = ir_desc.get_choice(choice_ast.name());
        let ref ctx = ast::Context::new(ir_desc, choice, &[], &[]);
        let args = choice.arguments().sets().enumerate()
            .map(|(i, set)| (ir::Variable::Arg(i), set)).collect_vec();
        let is_symmetric = choice.arguments().is_symmetric();
        let iters = PartialIterator::generate(&args, is_symmetric, ir_desc, ctx);
        iters.into_iter().map(move |(iter, ctx)| {
            let arg_names = args.iter().map(|&(v, _)| ctx.var_name(v)).collect();
            let value_type = ctx.choice.choice_def().value_type();
            let value_type = ast::ValueType::new(value_type, &ctx);
            (iter, NewChoice { choice: choice_ast, arg_names, value_type })
        })
    }).collect()
}

/// A newly allocated decision.
#[derive(Serialize)]
pub struct NewChoice<'a> {
    pub arg_names: Vec<Variable<'a>>,
    pub choice: &'a ChoiceAst<'a>,
    pub value_type: ast::ValueType,
}

/// Returns the partitial iterators for increment on existing counters.
pub fn incr_iterators<'a>(ir_desc: &'a ir::IrDesc) -> Vec<IncrIterator<'a>> {
    let mut out = Vec::new();
    for choice in ir_desc.choices() {
        if let ir::ChoiceDef::Counter {
                ref incr_iter, ref incr, ref incr_condition, visibility, kind, ref value, ..
        } = *choice.choice_def() {
            let ref ctx = ast::Context::new(ir_desc, choice, incr_iter, &[]);
            let ref counter = ir::ChoiceInstance {
                choice: choice.name().clone(),
                vars: (0..choice.arguments().len()).map(ir::Variable::Arg).collect(),
            };
            let counter_type = choice.choice_def().value_type();
            for (pos, set) in incr_iter.iter().enumerate() {
                // Setup the context.
                let obj = ast::Variable::with_name("obj");
                let ref mut ctx = ctx.set_var_name(ir::Variable::Forall(pos), obj);
                if let Some(arg) = set.arg() {
                    ctx.mut_var_name(arg, Variable::with_name("obj_var"));
                }
                // Setup the variables to loop on.
                let num_args = choice.arguments().len();
                let num_foralls = incr_iter.len();
                let counter_loops = if let Some(ir::Variable::Arg(arg)) = set.arg() {
                    (0..arg).chain(arg+1..num_args).map(ir::Variable::Arg).collect_vec()
                } else { (0..num_args).map(ir::Variable::Arg).collect_vec() };
                let incr_loops = if let Some(ir::Variable::Forall(arg)) = set.arg() {
                    assert!(arg < pos);
                    (0..arg).chain(arg+1..pos).chain(pos+1..num_foralls)
                        .map(ir::Variable::Forall).collect_vec()
                } else {
                    (0..pos).chain(pos+1..num_foralls)
                        .map(ir::Variable::Forall).collect_vec()
                };
                // Setup the conflicts of the new object argument.
                let arg_conflicts = set.arg().map(|arg| {
                    let arg_set = set.def().arg().unwrap();
                    match arg {
                        ir::Variable::Arg(_) => {
                            let list = ast::new_objs_list(arg_set.def(), "new_objs");
                            vec![ast::Conflict::NewObjs { list, set: arg_set.def() }]
                        },
                        ir::Variable::Forall(_) =>
                            PartialIterator::new_objs_conflicts(ir_desc, set).collect(),
                    }.into_iter().flat_map(|conflict| conflict.generate_ast(arg_set, ctx))
                    .collect()
                }).unwrap_or(vec![]);
                // Create the loop nest.
                let ref mut conflicts = PartialIterator::current_new_obj_conflicts(set)
                    .collect();
                let mut loop_nest = ast::LoopNest::new(counter_loops, ctx, conflicts, true);
                conflicts.extend(PartialIterator::new_objs_conflicts(ir_desc, set));
                loop_nest.extend(incr_loops, ctx, conflicts, false);
                let incr_condition = RcStr::new(value_set::print(incr_condition, ctx).to_string());
                let incr = ast::ChoiceInstance::new(incr, ctx);
                let set_ast = ast::SetDef::new(set.def());
                let incr_amount_getter = print::counter::increment_amount(value, true, ctx);
                let incr_amount: print::Value = incr_amount_getter
                    .create_ident("incr_value").into();
                let min_incr_amount = incr_amount.get_min(ctx);
                let max_incr_amount = incr_amount.get_max(ctx);
                let incr_iterator = IncrIterator {
                    iter: PartialIterator { loop_nest, set: set_ast, arg_conflicts },
                    incr, incr_condition, visibility, kind,
                    incr_amount: incr_amount_getter.into_token_stream().to_string(),
                    zero: kind.zero(),
                    counter: ast::ChoiceInstance::new(counter, ctx),
                    counter_type: ast::ValueType::new(counter_type.clone(), ctx),
                    min_incr_amount: min_incr_amount.into_token_stream().to_string(),
                    max_incr_amount: max_incr_amount.into_token_stream().to_string(),
                };
                out.push(incr_iterator);
            }
        }
    }
    out
}

/// AST for iteration over an the increment of a counter.
#[derive(Serialize)]
pub struct IncrIterator<'a> {
    iter: PartialIterator<'a>,
    incr_amount: String,
    min_incr_amount: String,
    max_incr_amount: String,
    counter: ast::ChoiceInstance<'a>,
    kind: ir::CounterKind,
    zero: u32,
    incr: ast::ChoiceInstance<'a>,
    visibility: ir::CounterVisibility,
    counter_type: ast::ValueType,
    incr_condition: RcStr,
}

/// AST for iteration over part of a choice instantiation space.
#[derive(Debug, Serialize)]
pub struct PartialIterator<'a> {
    set: ast::SetDef<'a>,
    arg_conflicts: Vec<ast::ConflictAst<'a>>,
    loop_nest: ast::LoopNest<'a>,
}

impl<'a> PartialIterator<'a> {
    /// Generates the list of sets of new objects and sets of objects to iterate on to
    /// visit all the new combinations of elements in the given combination of sets.
    pub fn generate(args: &[(ir::Variable, &'a ir::Set)], is_symmetric: bool,
                    ir_desc: &'a ir::IrDesc, ctx: &ast::Context<'a>)
        -> Vec<(Self, ast::Context<'a>)>
    {
        let mut output = Vec::new();
        for (pos, &(var, set)) in args.iter().enumerate() {
            let mut ctx = ctx.set_var_name(var, Variable::with_name("obj"));
            let mut loop_args = (0..pos).chain(pos+1..args.len())
                .map(|i| args[i].0).collect_vec();
            if let Some(set_arg) = set.arg() {
                loop_args.retain(|&v| v != set_arg);
                ctx.mut_var_name(set_arg, Variable::with_name("obj_var"));
            }
            let arg_conflicts = {
                let ctx = &ctx;
                set.def().arg().into_iter().flat_map(move |arg| {
                    PartialIterator::new_objs_conflicts(ir_desc, set)
                        .flat_map(move |c| c.generate_ast(arg, ctx))
                }).collect()
            };
            let mut conflicts = Self::new_objs_conflicts(ir_desc, set)
                .chain(Self::current_new_obj_conflicts(set)).collect();
            let loop_nest = LoopNest::new(loop_args, &ctx, &mut conflicts, false);
            let set_ast = ast::SetDef::new(set.def());
            let iter = PartialIterator { set: set_ast, loop_nest, arg_conflicts };
            output.push((iter, ctx));
            if is_symmetric { break }
        }
        output
    }

    /// Returns the list of sets of new objects to conflict with a given set.
    fn new_objs_conflicts(ir_desc: &'a ir::IrDesc, set: &'a ir::Set)
        -> impl Iterator<Item=ast::Conflict<'a>> + 'a
    {
        let self_conflict_name =
            format!("{}[pos+1..]", ast::new_objs_list(set.def(), "new_objs"));
        let new_objs = Variable::with_string(self_conflict_name);
        let self_conflict = ast::Conflict::NewObjs { list: new_objs, set: set.def() };
        ir_desc.set_defs().take_while(move |s| s.name() != set.def().name()).map(|set| {
            let list = ast::new_objs_list(set, "new_objs");
            ast::Conflict::NewObjs { list, set }
        }).chain(iter::once(self_conflict))
    }

    /// Returns the conflicts with the new object and its argument.
    fn current_new_obj_conflicts(set: &'a ir::Set)
        -> impl Iterator<Item=ast::Conflict<'a>> + 'a
    {
        let obj_conflict = ast::Conflict::new(ast::Variable::with_name("obj"), set);
        let arg_conflict = set.def().arg().map(|arg| {
            ast::Conflict::new(ast::Variable::with_name("obj_var"), arg)
        });
        iter::once(obj_conflict).chain(arg_conflict)
    }
}
