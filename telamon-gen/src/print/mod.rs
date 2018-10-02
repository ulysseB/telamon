//! Print the IR description in rust.
use handlebars::{self, Handlebars, Helper, RenderContext, RenderError, Renderable};
use ir;
use itertools::Itertools;
use serde_json::value::Value as JsonValue;
use std::fmt::{self, Display, Formatter};
//use std::io::prelude::*;
use pathfinding::prelude::topological_sort;
use utils::*;

// TODO(cleanup): use handlebars instead of printer macros and static templates

/// Create a '$sep' separated list of strings.
macro_rules! concat_sep {
    ($sep: expr, $front: expr) => { $front };
    ($sep: expr, $front: expr $(,$back:expr)+) => {
        concat!($front, $sep, concat_sep!($sep, $($back),*))
    };
}

macro_rules! register_template {
    ($engine:expr, $($name:ident)/+) => {
        let template = include_str!(concat!("template", $("/", stringify!($name)),*, ".rs"));
        // TODO(cleanup): use a relative path
        /*let mut file = ::std::io::BufReader::new(::std::fs::File::open(
            concat!("/home/ulysse/telamon/telamon-gen/src/print/template",
                    $("/", stringify!($name)),*, ".rs")).unwrap());
        let mut template = String::new();
        file.read_to_string(&mut template).unwrap();*/

        let name = concat_sep!(".", $(stringify!($name)),*);
        $engine.register_template_string(name, template.trim_right()).unwrap();
    }
}

macro_rules! render {
    ($($tmpl:ident)/+, $(< $($lifetime: tt),* >, )*
     $($name:ident: $ty: ty = $val: expr),*) => {
        {
            #[derive(Serialize)]
            struct Data<$($($lifetime),*),*> { $($name: $ty),* };
            let data = Data { $($name: $val),* };
            let template = concat_sep!(".", $(stringify!($tmpl)),*);
            ::print::ENGINE.render(template, &data).unwrap()
        }
    };
    ($($tmpl:ident)/+, $data:expr) => {
        {
            let template = concat_sep!(".", $(stringify!($tmpl)),*);
            ::print::ENGINE.render(template, &$data).unwrap()
        }
    }
}

lazy_static! {
    static ref ENGINE: Handlebars = {
        let mut engine = Handlebars::new();
        engine.register_escape_fn(handlebars::no_escape);
        engine.register_helper("replace", Box::new(replace));
        engine.register_helper("to_type_name", Box::new(to_type_name));
        engine.register_helper("ifeq", Box::new(ifeq));
        engine.register_helper("debug_context", Box::new(debug));
        register_template!(engine, add_to_quotient);
        register_template!(engine, alloc);
        register_template!(engine, account_new_incrs);
        register_template!(engine, actions);
        register_template!(engine, choice_def);
        register_template!(engine, choice_filter);
        register_template!(engine, choice / arg_names);
        register_template!(engine, choice / arg_defs);
        register_template!(engine, choice / arg_ids);
        register_template!(engine, choice / complement);
        register_template!(engine, choice / getter);
        register_template!(engine, compute_counter);
        register_template!(engine, conflict);
        register_template!(engine, counter_value);
        register_template!(engine, disable_increment);
        register_template!(engine, filter);
        register_template!(engine, filter_action);
        register_template!(engine, filter_call);
        register_template!(engine, filter_self);
        register_template!(engine, value_type / full_domain);
        register_template!(engine, value_type / name);
        register_template!(engine, value_type / num_constructor);
        register_template!(engine, value_type / univers);
        register_template!(engine, getter);
        register_template!(engine, incr_counter);
        register_template!(engine, iter_new_objects);
        register_template!(engine, loop_nest);
        register_template!(engine, main);
        register_template!(engine, on_change);
        register_template!(engine, positive_filter);
        register_template!(engine, propagate);
        register_template!(engine, restrict_counter);
        register_template!(engine, rule);
        register_template!(engine, run_filters);
        register_template!(engine, set_constraints);
        register_template!(engine, set / from_superset);
        register_template!(engine, set / id_getter);
        register_template!(engine, set / item_getter);
        register_template!(engine, set / iterator);
        register_template!(engine, set / new_objs);
        register_template!(engine, store);
        register_template!(engine, trigger_call);
        register_template!(engine, trigger_check);
        register_template!(engine, trigger_on_change);
        register_template!(engine, update_counter);
        engine
    };
}

type RenderResult = Result<(), RenderError>;

/// Performs string substitution.
fn replace(h: &Helper, _: &Handlebars, rc: &mut RenderContext) -> RenderResult {
    if h.is_block() {
        return Err(RenderError::new("replace is not valid on blocks"));
    }
    let mut string = match h.param(0).map(|p| p.value()) {
        None => return Err(RenderError::new("missing argument for replace")),
        Some(&JsonValue::String(ref string)) => string.clone(),
        Some(x) => {
            debug!("replace argument = {}", x);
            debug!("in context {:?}", rc.context());
            return Err(RenderError::new("replace expects string arguments"));
        }
    };
    for (key, value) in h.hash() {
        let value = value.value().as_str().ok_or_else(|| {
            let err = format!(
                "replace maps strings to strings (got {:?}), in context {:?}",
                value,
                rc.context()
            );
            RenderError::new(err)
        })?;
        string = string.replace(key, value);
    }
    rc.writer.write_all(string.into_bytes().as_ref())?;
    Ok(())
}

/// Prints a variable of the context.
fn debug(h: &Helper, _: &Handlebars, rc: &mut RenderContext) -> RenderResult {
    if h.is_block() {
        return Err(RenderError::new("debug is not valid on blocks"));
    }
    match h.param(0) {
        None => debug!("context {:?}", rc.context()),
        Some(x) => debug!("value {:?}", x.value()),
    };
    Ok(())
}

/// Converts a choice name to a rust type name.
fn to_type_name(h: &Helper, _: &Handlebars, rc: &mut RenderContext) -> RenderResult {
    if h.is_block() {
        return Err(RenderError::new("to_type_name is not valid on blocks"));
    }
    let string = match h.param(0).map(|p| p.value()) {
        None => return Err(RenderError::new("missing argument for to_type_name")),
        Some(&JsonValue::String(ref string)) => string.clone(),
        Some(x) => {
            debug!("replace argument = {}", x);
            return Err(RenderError::new("to_type_name expects a string argument"));
        }
    };
    rc.writer
        .write_all(::to_type_name(&string).into_bytes().as_ref())?;
    Ok(())
}

/// Prints the template if the a value is equal to another.
fn ifeq(h: &Helper, r: &Handlebars, rc: &mut RenderContext) -> RenderResult {
    let param = h
        .param(0)
        .ok_or_else(|| RenderError::new("Param 0 not found for helper \"ifeq\""))?;
    let value = h
        .param(1)
        .ok_or_else(|| RenderError::new("Param 1 not found for helper \"ifeq\""))?;
    let template = if param.value() == value.value() {
        h.template()
    } else {
        h.inverse()
    };
    template.map(|t| t.render(r, rc)).unwrap_or(Ok(()))
}

/// Creates a printer form an iterator.
// TODO(cleanup): remove printing macros
macro_rules! iter_printer {
    ($iter: expr, $item: pat, $($format_args: tt)*) => {
        ::print::Printer(move |f: &mut Formatter| {
            for $item in $iter { write!(f, $($format_args)*)?; }
            Ok(())
        })
    };
}

/// A closure around a printing function.
struct Printer<T: Fn(&mut Formatter) -> fmt::Result>(T);

impl<T: Fn(&mut Formatter) -> fmt::Result> Display for Printer<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result { self.0(f) }
}

mod ast;
mod choice;
mod filter;
mod store;
mod value_set;

use self::store::PartialIterator;

/// Generate the trigger code to add a representant to a quotient set.
pub fn add_to_quotient(
    set: &ir::SetDef,
    repr_name: &str,
    counter_name: &str,
    item: &str,
    var: &Option<RcStr>,
) -> String
{
    let mut add_to_set = set.attributes()[&ir::SetDefKey::AddToSet]
        .replace("$item", &format!("${}", item));
    if let Some(ref var) = *var {
        add_to_set = add_to_set.replace("$var", &format!("${}", var));
    }
    render!(add_to_quotient, <'a>,
        repr_name: &'a str = repr_name,
        counter_name: &'a str = counter_name,
        add_to_set: &'a str = &add_to_set,
        item: &'a str = item,
        var: &'a Option<RcStr> = var)
}

/// Generates code for the domain description.
pub fn print(ir_desc: &ir::IrDesc) -> String {
    let dummy_choice = &ir::dummy_choice();
    let choices = order_choices(ir_desc)
        .map(|c| choice::Ast::new(c, ir_desc))
        .collect_vec();
    let partials = store::partial_iterators(&choices, ir_desc);
    let incr_iterators = store::incr_iterators(ir_desc);
    let triggers = ir_desc
        .triggers()
        .enumerate()
        .map(|(id, t)| Trigger::new(id, t, dummy_choice, ir_desc))
        .collect();
    type PartialIters<'a> = Vec<(store::PartialIterator<'a>, store::NewChoice<'a>)>;
    render!(main, <'a>,
        choices: &'a [choice::Ast<'a>] = &choices,
        enums: String = ir_desc.enums().format("\n\n").to_string(),
        partial_iterators: PartialIters<'a> = partials,
        incr_iterators: Vec<store::IncrIterator<'a>> = incr_iterators,
        triggers: Vec<Trigger<'a>> = triggers
    )
}

/// Order the choices so that they are computed in the right order to avoid
/// overflows.
fn order_choices<'a>(
    ir_desc: &'a ir::IrDesc,
) -> impl Iterator<Item = &'a ir::Choice> + 'a {
    let names = ir_desc.choices().map(|c| c.name()).collect_vec();
    let sorted = topological_sort(&names, |choice| {
        let def = ir_desc.get_choice(choice).choice_def();
        if let ir::ChoiceDef::Counter { ref value, .. } = *def {
            if let ir::CounterVal::Choice(ref counter) = *value {
                return Some(&counter.choice);
            }
        }
        None
    });
    unwrap!(sorted)
        .into_iter()
        .rev()
        .map(move |c| ir_desc.get_choice(c))
}

#[derive(Debug, Serialize)]
struct Trigger<'a> {
    id: usize,
    loop_nest: ast::LoopNest<'a>,
    partial_iterators: Vec<(PartialIterator<'a>, Vec<ast::Variable<'a>>)>,
    inputs: Vec<(ast::Variable<'a>, ast::ChoiceInstance<'a>)>,
    arguments: Vec<(ast::Variable<'a>, ast::Set<'a>)>,
    conditions: Vec<String>,
    code: String,
}

impl<'a> Trigger<'a> {
    fn new(
        id: usize,
        trigger: &'a ir::Trigger,
        dummy_choice: &'a ir::Choice,
        ir_desc: &'a ir::IrDesc,
    ) -> Self
    {
        let inputs = &trigger.inputs;
        let ctx = &ast::Context::new(ir_desc, dummy_choice, &trigger.foralls, inputs);
        let inputs = trigger
            .inputs
            .iter()
            .enumerate()
            .map(|(pos, input)| {
                (ctx.input_name(pos), ast::ChoiceInstance::new(input, ctx))
            })
            .collect();
        let foralls = (0..trigger.foralls.len()).map(ir::Variable::Forall);
        let loop_nest = ast::LoopNest::new(foralls.clone(), ctx, &mut vec![], false);
        let arguments = foralls
            .clone()
            .map(|v| ctx.var_def(v))
            .map(|(v, set)| (v, ast::Set::new(set, ctx)))
            .collect();
        let conditions = trigger
            .conditions
            .iter()
            .map(|c| filter::condition(c, ctx))
            .collect();
        let code = ast::code(&trigger.code, ctx);
        let vars = foralls.map(|v| (v, ctx.var_def(v).1)).collect_vec();
        let partial_iters = PartialIterator::generate(&vars, false, ir_desc, ctx);
        let partial_iterators = partial_iters
            .into_iter()
            .map(|(iter, ctx)| {
                (iter, vars.iter().map(|&(v, _)| ctx.var_name(v)).collect())
            })
            .collect();
        Trigger {
            id,
            loop_nest,
            partial_iterators,
            arguments,
            inputs,
            conditions,
            code,
        }
    }
}

impl Display for ir::Enum {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let n_values = self.values().len();
        let value_defs = iter_printer!(
            value_def_order(self).enumerate(),
            (pos, (name, doc)),
            "{doc}pub const {name}: {t_name} = {t_name} {{ bits: 0b1{pos_zero} }};\n\n",
            doc = iter_printer!(doc, doc, "///{}\n", doc),
            name = name,
            t_name = self.name(),
            pos_zero = (0..pos).map(|_| "0").collect::<String>()
        );
        let alias_defs = iter_printer!(
            self.aliases(),
            (name, &(ref values, ref doc)),
            "{doc}pub const {name}: {t_name} = {t_name} {{ bits: {values} }};\n\n",
            doc = doc
                .iter()
                .format_with("", |doc, f| f(&format_args!("///{}\n", doc))),
            name = name,
            t_name = self.name(),
            values = values.iter().format_with(" | ", |v, f| f(&format_args!(
                "{}::{}.bits",
                self.name(),
                v
            )))
        );
        let printers = iter_printer!(
            self.values(),
            (v, _),
            "if self.intersects({t}::{v}) {{ values.push(\"{v}\"); }}",
            t = self.name(),
            v = v
        );
        write!(
            f,
            include_str!("template/enum_def.rs"),
            type_name = self.name(),
            doc_comment = iter_printer!(self.doc(), doc, "///{}\n", doc),
            value_defs = value_defs,
            alias_defs = alias_defs,
            num_values = n_values,
            bits_type = bits_type(n_values),
            all_bits = "0b".to_string() + &(0..n_values).map(|_| "1").collect::<String>(),
            inverse = Printer(|f| inverse(self, f)),
            printers = printers,
        )
    }
}

/// Prints the inverse function if needed.
fn inverse(enum_: &ir::Enum, f: &mut Formatter) -> fmt::Result {
    if let Some(mapping) = enum_.inverse_mapping() {
        let mut values: HashSet<_> = enum_.values().keys().collect();
        let mut low = Vec::new();
        let mut high = Vec::new();
        for &(ref lhs, ref rhs) in mapping {
            values.remove(lhs);
            values.remove(rhs);
            low.push(lhs);
            high.push(rhs);
        }
        let n = enum_.name();
        let same_bits = if values.is_empty() {
            "0".to_string()
        } else {
            values
                .iter()
                .map(|v| format!("{}::{}.bits", n, v))
                .format(" | ")
                .to_string()
        };
        write!(
            f,
            include_str!("template/inverse.rs"),
            type_name = enum_.name(),
            high_bits = high
                .iter()
                .map(|v| format!("{}::{}.bits", n, v))
                .format(" | "),
            low_bits = low
                .iter()
                .map(|v| format!("{}::{}.bits", n, v))
                .format(" | "),
            same_bits = same_bits
        )?;
    }
    Ok(())
}

fn value_def_order<'a>(
    enum_: &'a ir::Enum,
) -> impl Iterator<Item = (&'a RcStr, &'a Option<String>)> + 'a {
    if let Some(mapping) = enum_.inverse_mapping() {
        let mut values: HashMap<_, _> = enum_.values().iter().collect();
        Box::new(
            mapping
                .iter()
                .flat_map(|&(ref x, ref y)| vec![x, y])
                .map(|x| (x, values.remove(x).unwrap()))
                .collect_vec()
                .into_iter()
                .chain(values),
        ) as Box<Iterator<Item = _>>
    } else {
        Box::new(enum_.values().iter())
    }
}

/// Returns the type to use to implement a bitfiled.
fn bits_type(num_values: usize) -> &'static str {
    if num_values <= 8 {
        "u8"
    } else if num_values <= 16 {
        "u16"
    } else if num_values <= 32 {
        "u32"
    } else if num_values <= 64 {
        "u64"
    } else {
        panic!("too many variants")
    }
}

#[cfg(test)]
/// Printing for test structures.
mod test {
    use super::ENGINE;
    use ir::test::{EvalContext, StaticCond};
    use itertools::Itertools;
    use print;
    use std;

    impl<'a> std::fmt::Display for EvalContext<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            for (input, values) in self.inputs_def.iter().zip_eq(&self.input_values) {
                let choice = self.ir_desc.get_choice(&input.choice);
                let ctx =
                    print::ast::Context::new(self.ir_desc, choice, &[], self.inputs_def);
                let values = print::value_set::print(values, &ctx);
                write!(f, " {} = {},", input.choice, values)?;
            }
            for (cond, value) in &self.static_conds {
                write!(f, " {} = {},", cond, value)?;
            }
            Ok(())
        }
    }

    impl std::fmt::Display for StaticCond {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match *self {
                StaticCond::Code(ref code) => write!(f, "{}", code.code),
            }
        }
    }

    /// Test the replace helper.
    #[test]
    fn replace() {
        let _ = ::env_logger::try_init();
        let out = ENGINE
            .template_render("{{replace \"foobar\" foo=\"bar\"}}", &())
            .unwrap();
        assert_eq!(out, "barbar");
    }
}
