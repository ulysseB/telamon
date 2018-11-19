///! Filter code generation.
use ir;
use print;
use print::ast::{self, Context};
use print::value_set;
use proc_macro2::{Ident, Span, TokenStream};
use quote;

/// Ast for a filtering funtion.
#[derive(Serialize)]
pub struct Filter<'a> {
    id: usize,
    arguments: Vec<(ast::Variable<'a>, ast::Set<'a>)>,
    type_name: ast::ValueType,
    bindings: Vec<(ast::Variable<'a>, ast::ChoiceInstance<'a>)>,
    body: PositiveFilter<'a>,
}

impl<'a> Filter<'a> {
    pub fn new(
        filter: &'a ir::Filter,
        id: usize,
        choice: &'a ir::Choice,
        ir_desc: &'a ir::IrDesc,
    ) -> Self {
        trace!("filter {}_{}", choice.name(), id);
        let arguments = &filter.arguments[choice.arguments().len()..];
        let ref ctx = Context::new(ir_desc, choice, arguments, &filter.inputs);
        let arguments = filter
            .arguments
            .iter()
            .enumerate()
            .map(|(pos, t)| {
                let var = if pos < choice.arguments().len() {
                    ir::Variable::Arg(pos)
                } else {
                    ir::Variable::Forall(pos - choice.arguments().len())
                };
                (ctx.var_name(var), ast::Set::new(t, ctx))
            }).collect();
        let bindings = filter
            .inputs
            .iter()
            .enumerate()
            .map(|(id, input)| {
                (ctx.input_name(id), ast::ChoiceInstance::new(input, &ctx))
            }).collect();
        let values_var = ast::Variable::with_name("values");
        let body = PositiveFilter::new(&filter.rules, choice, &ctx, values_var.clone());
        let type_name = ast::ValueType::new(choice.value_type().full_type(), ctx);
        Filter {
            id,
            arguments,
            body,
            bindings,
            type_name,
        }
    }
}

/// A filter that enables values.
#[derive(Serialize)]
enum PositiveFilter<'a> {
    Switch {
        var: ast::Variable<'a>,
        cases: Vec<(String, PositiveFilter<'a>)>,
    },
    AllowValues {
        var: ast::Variable<'a>,
        values: String,
    },
    AllowAll {
        var: ast::Variable<'a>,
        value_type: ast::ValueType,
    },
    Rules {
        value_type: ast::ValueType,
        old_var: ast::Variable<'a>,
        new_var: ast::Variable<'a>,
        rules: Vec<Rule<'a>>,
    },
    Empty,
}

impl<'a> PositiveFilter<'a> {
    /// Creates a `PositiveFilter` enforcing a set of rules.
    fn rules(
        rules: &'a [ir::Rule],
        choice: &'a ir::Choice,
        ctx: &Context<'a>,
        var: ast::Variable<'a>,
    ) -> Self {
        let value_type = ast::ValueType::new(choice.value_type().full_type(), ctx);
        match *rules {
            [] => PositiveFilter::AllowAll { var, value_type },
            [ref r] if r.conditions.is_empty() && r.set_constraints.is_empty() => {
                if r.alternatives.is_empty() {
                    PositiveFilter::Empty
                } else {
                    let values = value_set::print(&r.alternatives, ctx).to_string();
                    PositiveFilter::AllowValues { var, values }
                }
            }
            ref rules => {
                let new_var = ast::Variable::with_prefix("values");
                let rules = rules
                    .iter()
                    .map(|r| Rule::new(new_var.clone(), r, ctx))
                    .collect();
                PositiveFilter::Rules {
                    value_type,
                    old_var: var,
                    new_var,
                    rules,
                }
            }
        }
    }

    /// Create a PositiveFilter from an ir::SubFilter.
    fn new(
        filter: &'a ir::SubFilter,
        choice: &'a ir::Choice,
        ctx: &Context<'a>,
        set: ast::Variable<'a>,
    ) -> Self {
        match *filter {
            ir::SubFilter::Rules(ref rules) => {
                PositiveFilter::rules(rules, choice, ctx, set)
            }
            ir::SubFilter::Switch { switch, ref cases } => {
                let cases = cases
                    .iter()
                    .map(|&(ref values, ref sub_filter)| {
                        let sub_filter =
                            PositiveFilter::new(sub_filter, choice, ctx, set.clone());
                        (value_set::print(values, ctx).to_string(), sub_filter)
                    }).collect();
                PositiveFilter::Switch {
                    var: ctx.input_name(switch),
                    cases,
                }
            }
        }
    }
}

#[derive(Serialize)]
pub struct Rule<'a> {
    var: ast::Variable<'a>,
    conditions: Vec<String>,
    set_conditions: Vec<ast::SetConstraint<'a>>,
    values: String,
}

impl<'a> Rule<'a> {
    pub fn new(
        var: ast::Variable<'a>,
        rule: &'a ir::Rule,
        ctx: &Context<'a>,
    ) -> Rule<'a> {
        let values = value_set::print(&rule.alternatives, ctx).to_string();
        let conditions = rule
            .conditions
            .iter()
            .map(|c| condition(c, ctx).to_string())
            .collect();
        let set_conditions = ast::SetConstraint::new(&rule.set_constraints, ctx);
        Rule {
            var,
            conditions,
            set_conditions,
            values,
        }
    }
}

/// Prints an `ir::Condition`.
pub fn condition(cond: &ir::Condition, ctx: &Context) -> TokenStream {
    match cond {
        ir::Condition::Bool(b) => quote!(#b),
        ir::Condition::Code { code, negate } => {
            let mut code = print::Value::new_const(code, ctx);
            if *negate {
                code.negate();
            }
            quote!(#code)
        }
        ir::Condition::Enum {
            input,
            values,
            negate,
            inverse,
        } => {
            let input = ctx.input(*input);
            let enum_def = ctx.ir_desc.get_enum(unwrap!(input.value_type().as_enum()));
            let set = ir::normalized_enum_set(values, !*negate, *inverse, enum_def);
            let value_set = value_set::print(&set, ctx);
            quote!(!#input.intersects(#value_set))
        }
        ir::Condition::CmpCode { lhs, rhs, op } => {
            let rhs = print::Value::new_const(rhs, ctx);
            comparison(*op, &ctx.input(*lhs).clone().into(), &rhs, ctx)
        }
        ir::Condition::CmpInput {
            lhs,
            rhs,
            op,
            inverse,
        } => {
            let mut rhs: print::Value = ctx.input(*rhs).clone().into();
            if *inverse {
                rhs.inverse()
            }
            comparison(*op, &ctx.input(*lhs).clone().into(), &rhs, ctx)
        }
    }
}

/// Produces code that compares to domains.
fn comparison(
    op: ir::CmpOp,
    lhs: &print::Value,
    rhs: &print::Value,
    ctx: &Context,
) -> TokenStream {
    if let ir::ValueType::Enum(..) = lhs.value_type() {
        match op {
            ir::CmpOp::Eq => quote!((#lhs.is_constrained() && #lhs.intersects(#rhs))),
            ir::CmpOp::Neq => quote!(!#lhs.intersects(#rhs)),
            _ => panic!("unsupported operator on enums"),
        }
    } else {
        let lhs_universe = print::value::universe(lhs.value_type(), ctx);
        let rhs_universe = print::value::universe(rhs.value_type(), ctx);
        quote!(#lhs.#op(#lhs_universe, #rhs, #rhs_universe))
    }
}

impl quote::ToTokens for ir::CmpOp {
    fn to_tokens(&self, stream: &mut TokenStream) {
        let name = match self {
            ir::CmpOp::Eq => "eq",
            ir::CmpOp::Neq => "neq",
            ir::CmpOp::Lt => "lt",
            ir::CmpOp::Gt => "gt",
            ir::CmpOp::Leq => "leq",
            ir::CmpOp::Geq => "geq",
        };
        // TODO(span): get the real span from the lexer
        Ident::new(name, Span::call_site()).to_tokens(stream);
    }
}
