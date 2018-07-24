//! Print the IR description in rust.
use handlebars::{self, Handlebars, Helper, Renderable, RenderContext, RenderError};
use ir;
use itertools::Itertools;
use serde_json::value::Value as JsonValue;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt::{self, Display, Formatter};
use std::hash::Hash;
use std::iter::FromIterator;
//use std::io::prelude::*;
use indexmap::IndexMap;
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
        register_template!(engine, choice/arg_names);
        register_template!(engine, choice/arg_defs);
        register_template!(engine, choice/arg_ids);
        register_template!(engine, choice/complement);
        register_template!(engine, choice/getter);
        register_template!(engine, compute_counter);
        register_template!(engine, conflict);
        register_template!(engine, counter_value);
        register_template!(engine, disable_increment);
        register_template!(engine, filter);
        register_template!(engine, filter_action);
        register_template!(engine, filter_call);
        register_template!(engine, filter_self);
        register_template!(engine, value_type/full_domain);
        register_template!(engine, value_type/name);
        register_template!(engine, value_type/num_constructor);
        register_template!(engine, value_type/univers);
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
        register_template!(engine, set/from_superset);
        register_template!(engine, set/id_getter);
        register_template!(engine, set/item_getter);
        register_template!(engine, set/iterator);
        register_template!(engine, set/new_objs);
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
    if h.is_block() { return Err(RenderError::new("replace is not valid on blocks")); }
    let mut string = match h.param(0).map(|p| p.value()) {
        None => return Err(RenderError::new("missing argument for replace")),
        Some(&JsonValue::String(ref string)) => string.clone(),
        Some(x) => {
            debug!("replace argument = {}", x);
            debug!("in context {:?}", rc.context());
            return Err(RenderError::new("replace expects string arguments"))
        },
    };
    for (key, value) in h.hash() {
        let value = value.value().as_str().ok_or_else(|| {
            let err = format!("replace maps strings to strings (got {:?}), in context {:?}",
                value, rc.context());
            RenderError::new(err)
        })?;
        string = string.replace(key, value);
    }
    rc.writer.write_all(string.into_bytes().as_ref())?;
    Ok(())
}

/// Prints a variable of the context.
fn debug(h: &Helper, _: &Handlebars, rc: &mut RenderContext) -> RenderResult {
    if h.is_block() { return Err(RenderError::new("debug is not valid on blocks")); }
    match h.param(0) {
        None => debug!("context {:?}", rc.context()),
        Some(x) => debug!("value {:?}", x.value()),
    };
    Ok(())
}

/// Converts a choice name to a rust type name.
fn to_type_name(h: &Helper, _: &Handlebars, rc: &mut RenderContext) -> RenderResult {
    if h.is_block() { return Err(RenderError::new("to_type_name is not valid on blocks")); }
    let string = match h.param(0).map(|p| p.value()) {
        None => return Err(RenderError::new("missing argument for to_type_name")),
        Some(&JsonValue::String(ref string)) => string.clone(),
        Some(x) => {
            debug!("replace argument = {}", x);
            return Err(RenderError::new("to_type_name expects a string argument"))
        },
    };
    rc.writer.write_all(::to_type_name(&string).into_bytes().as_ref())?;
    Ok(())
}

/// Prints the template if the a value is equal to another.
fn ifeq(h: &Helper, r: &Handlebars, rc: &mut RenderContext) -> RenderResult {
    let param = h.param(0).ok_or_else(|| {
        RenderError::new("Param 0 not found for helper \"ifeq\"")
    })?;
    let value = h.param(1).ok_or_else(|| {
        RenderError::new("Param 1 not found for helper \"ifeq\"")
    })?;
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
#[cfg(test)]
pub(crate) use self::ast::Variable;

/// Generate the trigger code to add a representant to a quotient set.
pub fn add_to_quotient(set: &ir::SetDef,
                       repr_name: &str,
                       counter_name: &str,
                       item: &str,
                       var: &Option<RcStr>) -> String {
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
        .map(|c| choice::Ast::new(c, ir_desc)).collect_vec();
    let partials = store::partial_iterators(&choices, ir_desc);
    let incr_iterators = store::incr_iterators(ir_desc);
    let triggers = ir_desc.triggers().enumerate()
        .map(|(id, t)| Trigger::new(id, t, dummy_choice, ir_desc)).collect();
    type PartialIters<'a> = Vec<(store::PartialIterator<'a>, store::NewChoice<'a>)>;
    render!(main, <'a>,
        choices: &'a [choice::Ast<'a>] = &choices,
        enums: String = ir_desc.enums().format("\n\n").to_string(),
        partial_iterators: PartialIters<'a> = partials,
        incr_iterators: Vec<store::IncrIterator<'a>> = incr_iterators,
        triggers: Vec<Trigger<'a>> = triggers
    )
}

/// Find a topological order in a directed graph.
///
/// The topological order is guaranteed to be stable, i.e. the
/// relative position of elements which are not (transitively) ordered
/// is preserved.
///
/// # Return value
///
/// This function will either return `Ok` with a stable topological
/// order or, if there are cycles in the ordering defined by
/// `predecessors`, `Err` with a pair `(sorted, cycles)` where
/// `cycles` contains all the nodes which are part of a cycle, and
/// `sorted` contains a stable topological order of the remaining
/// nodes.
///
/// # Arguments
///
/// * `nodes` - A slice that holds the nodes to sort. Must not contain
///   duplicates.
/// * `predecessors` - A function returning an iterable of
///   predecessors or dependencies for each node. Will be called
///   exactly once for each node in `nodes`. Each element of the
///   returned iterator must be in `nodes.
///
/// # Panics
///
/// This function will panic if there are duplicates in `nodes` or if
/// `predecessors` return a value which is not in `nodes`.
///
/// # Examples
///
/// See the `stable_topological_sort_tests` module.
fn stable_topological_sort<'a, N, P, I>(
    nodes: &[N],
    predecessors: P,
) -> Result<Vec<N>, (Vec<N>, Vec<N>)>
where
    N: Eq + Hash + Clone,
    P: Fn(&N) -> I,
    I: IntoIterator<Item = N>,
{
    /// Data about a node used by the topological sort algorithm.
    #[derive(Eq)]
    struct NodeData {
        /// The index of the node in `nodes`
        index: usize,
        /// The number of predecessors. Nodes that have already been
        /// sorted are no longer considered as predecessors.
        num_predecessors: usize,
        /// The indices in `nodes` of this node's successors.
        successors: Vec<usize>,
    }

    /// `PartialEq` is implemented manually since it needs to be
    /// compatible with `Ord` and `PartialOrd`. Note that technically
    /// we will only ever create a single `NodeData` with a given
    /// `index` value, so the default implementation would only be
    /// wrong conceptually (although slower than the manual version).
    impl PartialEq for NodeData {
        fn eq(&self, other: &NodeData) -> bool {
            self.index == other.index
        }
    }

    /// Custom `Ord` implementation to only take the `index` into
    /// account. Note that we want the `BinaryHeap` to act as a
    /// min-heap below, so we reverse the ordering here.
    impl Ord for NodeData {
        fn cmp(&self, other: &NodeData) -> Ordering {
            self.index.cmp(&other.index).reverse()
        }
    }

    impl PartialOrd for NodeData {
        fn partial_cmp(&self, other: &NodeData) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl NodeData {
        fn new(index: usize) -> NodeData {
            NodeData {
                index,
                num_predecessors: 0,
                successors: Vec::new(),
            }
        }
    }

    // We need an index map to map back from the predecessors into their indices.
    let mut node_data: Vec<_> = (0..nodes.len()).map(NodeData::new).map(Some).collect();
    let index_map: HashMap<_, _> =
        HashMap::from_iter(nodes.iter().enumerate().map(|(index, node)| (node, index)));
    if index_map.len() != nodes.len() {
        panic!("duplicate nodes found");
    }

    // Properly set the predecessors and successors for each node.
    for (index, node) in nodes.iter().enumerate() {
        let num_predecessors = predecessors(node)
            .into_iter()
            .map(|pred| {
                let &pred_index =
                    index_map.get(&pred).expect("predecessor is not in graph");
                node_data[pred_index]
                    .as_mut()
                    .unwrap()
                    .successors
                    .push(index)
            })
            .count();

        node_data[index].as_mut().unwrap().num_predecessors = num_predecessors
    }

    // Build the priority queue containing the nodes which have no
    // predecessors. Note that we must loop again to create this
    // instead of `take`ing directly from `node_data` when counting
    // the predecessors because otherwise we would throw off the
    // predecessor count for the succesors, and decrementing that
    // count for the nodes we have already moved to the queue would
    // violate the stability guarantee.
    let mut queue = BinaryHeap::with_capacity(nodes.len());
    for data in node_data.iter_mut() {
        if data.as_mut().unwrap().num_predecessors == 0 {
            queue.push(data.take().unwrap())
        }
    }

    // At each iteration of this loop we maintain the invariant that
    // the priority queue contains exactly the nodes for which all
    // predecessors are in sorted, but have not been sorted
    // themselves. Using a priority queue ensures that we are always
    // processing nodes in the order they appear in the initial
    // `nodes` slice.
    //
    // Whenever we move a node from the queue to the `sorted` array,
    // we update all its successors (if any) by decrementing its
    // number of predecessors. Whenever the number of predecessors
    // reaches `0`, this node was the last predecessor not already
    // sorted, and we add the successor to the priority queue.
    let mut sorted = Vec::with_capacity(nodes.len());
    while let Some(data) = queue.pop() {
        sorted.push(nodes[data.index].clone());

        for &successor_index in data.successors.iter() {
            let mut successor_data = &mut node_data[successor_index];
            successor_data.as_mut().unwrap().num_predecessors -= 1;
            if successor_data.as_mut().unwrap().num_predecessors == 0 {
                queue.push(successor_data.take().unwrap())
            }
        }
    }

    // If there is a cycle, we will never put it in the queue because
    // there always will be another element of the cycle as a
    // successor, and so `sorted` will be smaller than `nodes.We can
    // get back all the elements which belong to a cycle by looking
    // for the non-empty indices in `node_data`.
    if sorted.len() != nodes.len() {
        let mut cycles = Vec::with_capacity(nodes.len() - sorted.len());
        for (index, data) in node_data.into_iter().enumerate() {
            if data.is_some() {
                cycles.push(nodes[index].clone())
            }
        }
        Err((sorted, cycles))
    } else {
        Ok(sorted)
    }
}

/// Note that the code in the `stable_topological_sort_tests` omdule
/// should be in an Examples sections in the `stable_topological_sort`
/// function, but we can't run rustdoc tests on private functions for
/// some reason.
///
/// For the sake of concision and readability, the examples use
/// letters to represent nodes and `:` to separate a node from its
/// dependencies. For instance, `"d"` is a node called `"d"` without
/// dependencies, `"e:d"` is a node called `"e"` with a single
/// dependency on the node `"d"`, and `"f:e,q"` is a node called `"f"`
/// with two dependencies on nodes `"e"` and `"q"`.
#[cfg(test)]
mod stable_topological_sort_tests {
    use super::stable_topological_sort;

    /// Sorting an array with no dependencies does not modify the
    /// order.
    #[test]
    fn stable_no_deps() {
        assert_eq!(
            stable_topological_sort(&vec!["a", "b", "c", "d", "e"], |_| vec![]),
            Ok(vec!["a", "b", "c", "d", "e"]))
    }

    /// Sorting an array which is already in topological order, even in
    /// the presence of dependencies, does not modify the order either.
    #[test]
    fn stable_deps() {
        assert_eq!(
            stable_topological_sort(&vec!["a", "b", "c:b", "d"], |&node| {
                match node {
                    "c:b" => vec!["b"],
                    _ => vec![],
                }
            }),
            Ok(vec!["a", "b", "c:b", "d"]))
    }

    /// Elements which have a dependency are sorted immediately after
    /// their last dependency.
    #[test]
    fn right_after_dep() {
        assert_eq!(
            stable_topological_sort(&vec!["a", "b:d", "c", "d", "e"], |&node| {
                match node {
                    "b:d" => vec!["d"],
                    _ => vec![],
                }
            }),
            Ok(vec!["a", "c", "d", "b:d", "e"]))
    }

    /// If multiple independent elements share a dependency, their
    /// initial order is maintained.
    #[test]
    fn indep_after_dep() {
        assert_eq!(
            stable_topological_sort(&vec!["a", "b:d", "c:d", "d", "e"], |&node| {
                match node {
                    "b:d" => vec!["d"],
                    "c:d" => vec!["d"],
                    _ => vec![],
                }
            }),
            Ok(vec!["a", "d", "b:d", "c:d", "e"]))
    }

    /// If multiple elements share a dependency, their own dependencies
    /// are still satisfied.
    #[test]
    fn dep_after_dep() {
        assert_eq!(
            stable_topological_sort(&vec!["a", "b:d,c", "c:d", "d", "e"], |&node| {
                match node {
                    "b:d,c" => vec!["d", "c:d"],
                    "c:d" => vec!["d"],
                    _ => vec![],
                }
            }),
            Ok(vec!["a", "d", "c:d", "b:d,c", "e"]))
    }
}

/// Order the choices so that they are computed in the right order to avoid overflows.
fn order_choices<'a>(ir_desc: &'a ir::IrDesc) -> impl Iterator<Item=&'a ir::Choice> +'a {
    let names = ir_desc.choices().map(|c| c.name()).collect_vec();
    let sorted = stable_topological_sort(&names, |choice| {
        let def = ir_desc.get_choice(choice).choice_def();
        if let ir::ChoiceDef::Counter { ref value, .. } = *def {
            if let ir::CounterVal::Choice(ref counter) = *value {
                return Some(&counter.choice);
            }
        }
        None
    });
    unwrap!(sorted).into_iter().map(move |c| ir_desc.get_choice(c))
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
    fn new(id: usize, trigger: &'a ir::Trigger, dummy_choice: &'a ir::Choice,
           ir_desc: &'a ir::IrDesc) -> Self {
        let inputs = &trigger.inputs;
        let ctx = &ast::Context::new(ir_desc, dummy_choice, &trigger.foralls, inputs);
        let inputs = trigger.inputs.iter().enumerate().map(|(pos, input)| {
            (ctx.input_name(pos), ast::ChoiceInstance::new(input, ctx))
        }).collect();
        let foralls = (0..trigger.foralls.len()).map(ir::Variable::Forall);
        let loop_nest = ast::LoopNest::new(foralls.clone(), ctx, &mut vec![], false);
        let arguments = foralls.clone().map(|v| ctx.var_def(v))
            .map(|(v, set)| (v, ast::Set::new(set, ctx))).collect();
        let conditions = trigger.conditions.iter()
            .map(|c| filter::condition(c, ctx)).collect();
        let code = ast::code(&trigger.code, ctx);
        let vars = foralls.map(|v| (v, ctx.var_def(v).1)).collect_vec();
        let partial_iters = PartialIterator::generate(&vars, false, ir_desc, ctx);
        let partial_iterators = partial_iters.into_iter().map(|(iter, ctx)| {
            (iter, vars.iter().map(|&(v, _)| ctx.var_name(v)).collect())
        }).collect();
        Trigger { id, loop_nest, partial_iterators, arguments, inputs, conditions, code }
    }
}


impl Display for ir::Enum {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let n_values = self.values().len();
        let value_defs = iter_printer!(value_def_order(self).enumerate(),
            (pos, (name, doc)),
            "{doc}pub const {name}: {t_name} = {t_name} {{ bits: 0b1{pos_zero} }};\n\n",
            doc = iter_printer!(doc, doc, "///{}\n", doc),
            name = name,
            t_name = self.name(),
            pos_zero = (0..pos).map(|_| "0").collect::<String>());
        let alias_defs = iter_printer!(self.aliases(), (name, &(ref values, ref doc)),
            "{doc}pub const {name}: {t_name} = {t_name} {{ bits: {values} }};\n\n",
            doc = doc.iter().format_with("", |doc, f| f(&format_args!("///{}\n", doc))),
            name = name,
            t_name = self.name(),
            values = values.iter().format_with(" | ", |v, f| {
                f(&format_args!("{}::{}.bits", self.name(), v))
            }));
        let printers = iter_printer!(self.values(), (v, _),
            "if self.intersects({t}::{v}) {{ values.push(\"{v}\"); }}",
            t=self.name(), v=v);
        write!(f, include_str!("template/enum_def.rs"),
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
        let same_bits = if values.is_empty() { "0".to_string() } else {
            values.iter().map(|v| format!("{}::{}.bits", n, v)).format(" | ").to_string()
        };
        write!(f, include_str!("template/inverse.rs"),
            type_name = enum_.name(),
            high_bits = high.iter().map(|v| format!("{}::{}.bits", n, v)).format(" | "),
            low_bits = low.iter().map(|v| format!("{}::{}.bits", n, v)).format(" | "),
            same_bits = same_bits)?;
    }
    Ok(())
}

fn value_def_order<'a>(enum_: &'a ir::Enum)
        -> impl Iterator<Item=(&'a RcStr, &'a Option<String>)> +'a {
    if let Some(mapping) = enum_.inverse_mapping() {
        let mut values: IndexMap<_, _> = enum_.values().iter().collect();
        Box::new(mapping.iter().flat_map(|&(ref x, ref y)| vec![x, y])
            .map(|x| (x, values.remove(x).unwrap())).collect_vec().into_iter()
            .chain(values)) as Box<Iterator<Item=_>>
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
    use ir::test::{EvalContext, StaticCond};
    use itertools::Itertools;
    use print;
    use std;
    use super::ENGINE;

    impl<'a> std::fmt::Display for EvalContext<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            for (input, values) in self.inputs_def.iter().zip_eq(&self.input_values) {
                let choice = self.ir_desc.get_choice(&input.choice);
                let ctx = print::ast::Context::new(self.ir_desc, choice, &[],
                                                   self.inputs_def);
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
        let out = ENGINE.template_render("{{replace \"foobar\" foo=\"bar\"}}", &()).unwrap();
        assert_eq!(out, "barbar");
    }
}
