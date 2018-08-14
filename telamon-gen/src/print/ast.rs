//! AST building blocks for the generated code.
use ir;
use ir::SetRef;
use itertools::Itertools;
use serde::{Serialize, Serializer};
use std::fmt::{self, Display, Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use utils::*;
use indexmap::IndexMap;

/// A named variable.
#[derive(Clone, Debug)]
pub enum Variable<'a> { Ref(&'a str), Rc(RcStr) }

lazy_static! { static ref NEXT_VAR_ID: AtomicUsize = AtomicUsize::new(0); }

impl<'a> Variable<'a> {
    /// Reset the prefix counter. This is meant for use in tests only.
    #[cfg(test)]
    #[doc(hidden)]
    pub(crate) fn reset_prefix() {
        NEXT_VAR_ID.store(0, Ordering::SeqCst);
    }

    /// Creates a new variable with the given prefix.
    pub fn with_prefix(prefix: &str) -> Self {
        let id = NEXT_VAR_ID.fetch_add(1, Ordering::SeqCst);
        Variable::Rc(RcStr::new(format!("{}_{}", prefix, id)))
    }

    /// Create a new variable with a prefix matchingthe given set.
    pub fn with_set(set: &ir::Set) -> Self { Self::with_prefix(set.def().prefix()) }

    /// Creates a variable from an existing name.
    pub fn with_name(name: &'a str) -> Self { Variable::Ref(name) }

    /// Creates a variable from an existing name.
    pub fn with_string(name: String) -> Self { Variable::Rc(RcStr::new(name)) }

    pub fn name(&self) -> &str {
        match *self {
            Variable::Ref(name) => name,
            Variable::Rc(ref name) => &**name as &str,
        }
    }
}

hash_from_key!(Variable<'a>, Variable::name, 'a);

impl<'a> Serialize for Variable<'a> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

/// The printing context.
#[derive(Clone)]
pub struct Context<'a> {
    pub ir_desc: &'a ir::IrDesc,
    pub choice: &'a ir::Choice,
    vars: HashMap<ir::Variable, (Variable<'a>, &'a ir::Set)>,
    input_names: Vec<Variable<'a>>,
    input_defs: &'a [ir::ChoiceInstance],
}

impl<'a> Context<'a> {
    /// Creates a new context.
    pub fn new<IT>(ir_desc: &'a ir::IrDesc, choice: &'a ir::Choice, foralls: IT,
                   input_defs: &'a [ir::ChoiceInstance]) -> Self
        where IT: IntoIterator<Item=&'a ir::Set>
    {
        let mut vars = HashMap::default();
        for (id, (name, set)) in choice.arguments().iter().enumerate() {
            vars.insert(ir::Variable::Arg(id), (Variable::with_name(name), set));
        }
        for (id, set) in foralls.into_iter().enumerate() {
            vars.insert(ir::Variable::Forall(id), (Variable::with_set(set), set));
        }
        let input_names = input_defs.iter()
            .map(|i| Variable::with_prefix(&i.choice)).collect();
        Context { ir_desc, choice, vars, input_names, input_defs }
    }

    /// Sets the name of a variable.
    pub fn set_var_name(&self, var: ir::Variable, name: Variable<'a>) -> Self {
        let mut ctx = self.clone();
        ctx.vars.get_mut(&var).unwrap().0 = name;
        ctx
    }

    pub fn mut_var_name(&mut self, var: ir::Variable, name: Variable<'a>) {
        self.vars.get_mut(&var).unwrap().0 = name;
    }

    /// Returns the name of a variable.
    pub fn var_name(&self, var: ir::Variable) -> Variable<'a> {
        self.vars[&var].0.clone()
    }

    /// Returns the name of the variable and the set it iterates on.
    pub fn var_def(&self, var: ir::Variable) -> (Variable<'a>, &'a ir::Set) {
        let ref entry = self.vars[&var];
        (entry.0.clone(), &entry.1)
    }

    /// Returns the name of an input.
    pub fn input_name(&self, id: usize) -> Variable<'a> {
        self.input_names[id].clone()
    }

    /// Returns the choice definition of an input.
    pub fn input_choice_def(&self, id: usize) -> &'a ir::ChoiceDef {
        self.ir_desc.get_choice(&self.input_defs[id].choice).choice_def()
    }
}

/// An instance of a choice.
///
/// Associated templates:
/// * [choice/getter]: retrives the choice value from the store.
#[derive(Debug, Serialize)]
pub struct ChoiceInstance<'a> {
    name: &'a str,
    arguments: Vec<(Variable<'a>, Set<'a>)>,
}

impl<'a> ChoiceInstance<'a> {
    /// Creates the choice instance from an `ir::ChoiceInstance` and a context.
    pub fn new(instance: &ir::ChoiceInstance, ctx: &Context<'a>) -> Self {
        let choice = ctx.ir_desc.get_choice(&instance.choice);
        let vars = instance.vars.iter().map(|&v| ctx.var_name(v));
        let sets = choice.arguments().sets().map(|s| Set::new(s, ctx));
        ChoiceInstance { name: choice.name(), arguments: vars.zip_eq(sets).collect() }
    }
}

#[derive(Debug, Serialize)]
pub struct SetConstraint<'a> {
    var: Variable<'a>,
    sets: Vec<Set<'a>>,
}

impl<'a> SetConstraint<'a> {
    pub fn new(constraints: &'a ir::SetConstraints, ctx: &Context<'a>) -> Vec<Self> {
        constraints.constraints().iter().flat_map(|&(var, ref expected)| {
            let (ref var, ref current) = ctx.vars[&var];
            let sets = expected.path_to_superset(current);
            if sets.is_empty() { None } else {
                let sets = sets.into_iter().rev().map(|s| Set::new(s, ctx)).collect();
                Some(SetConstraint { var: var.clone(), sets })
            }
        }).collect()
    }
}

/// A conflict that must be skipped in a loop nest.
#[derive(Clone, Debug)]
pub enum Conflict<'a> {
    /// Ensure the loop does not iterates on a previously declared variable.
    Variable { var: Variable<'a>, set: &'a ir::Set },
    /// Ensures the loop does not iterates on an object newly added to the set.
    NewObjs { list: Variable<'a>, set: &'a ir::SetDef },
}

#[derive(Clone, Debug, Serialize)]
pub enum ConflictAst<'a> {
    Variable { conflict_var: Variable<'a>, set: SetDef<'a> },
    NewObjs { list: Variable<'a>, set: Set<'a> },
}

impl<'a> Conflict<'a> {
    /// Creates a new conflict.
    pub fn new(var: Variable<'a>, set: &'a ir::Set) -> Self {
        Conflict::Variable { var, set }
    }

    /// Generates the list of conflicts with all the aruments of a choice.
    pub fn choice_args(choice: &'a ir::Choice, ctx: &Context<'a>) -> Vec<Self> {
        choice.arguments().sets().enumerate().map(|(pos, t)| {
            Conflict::new(ctx.var_name(ir::Variable::Arg(pos)), t)
        }).collect()
    }

    /// Generates the an AST if `self` conflicts with `set`..
    pub fn generate_ast(&self, set: &'a ir::Set, ctx: &Context<'a>)
        -> Option<ConflictAst<'a>>
    {
        match *self {
            Conflict::Variable { ref var, set: conflict_set } => {
                conflict_set.get_collision_level(set).map(|set| {
                    let set = SetDef::new(set);
                    ConflictAst::Variable { conflict_var: var.clone(), set }
                })
            },
            Conflict::NewObjs { ref list, set: conflict_set }
                if conflict_set.name() == set.def().name() =>
            {
                Some(ConflictAst::NewObjs { list: list.clone(), set: Set::new(set, ctx) })
            },
            Conflict::NewObjs { .. } => None,
        }
    }
}

/// Builds a loop nest given a body.
#[derive(Clone, Debug, Serialize)]
pub struct LoopNest<'a> {
    levels: Vec<(Variable<'a>, Set<'a>, Vec<ConflictAst<'a>>)>,
    triangular: bool,
}

impl<'a> LoopNest<'a> {
    /// Creates a new loop nest build.
    pub fn new<IT>(it: IT, ctx: &Context<'a>,
                   outer_vars: &mut Vec<Conflict<'a>>,
                   skip_new_objs: bool) -> Self
            where IT: IntoIterator<Item=ir::Variable> {
        let levels = it.into_iter().map(|var| {
            Self::build_level(var, ctx, outer_vars, skip_new_objs)
        }).collect();
        LoopNest { levels: levels, triangular: false }
    }

    /// Extends the loop with new levels.
    pub fn extend<IT>(&mut self, it: IT,
                      ctx: &Context<'a>,
                      outer_vars: &mut Vec<Conflict<'a>>,
                      skip_new_objs: bool) where IT: IntoIterator<Item=ir::Variable> {
        self.levels.extend(it.into_iter().map(|v| {
            Self::build_level(v, ctx, outer_vars, skip_new_objs)
        }));
    }

    /// Creates a triangular loop nesti build.
    pub fn triangular(lhs: ir::Variable, rhs: ir::Variable, ctx: &Context<'a>) -> Self {
        let (lhs, lhs_set) = ctx.var_def(lhs);
        let (rhs, rhs_set) = ctx.var_def(rhs);
        assert_eq!(lhs_set, rhs_set);
        let set = Set::new(lhs_set, ctx);
        let conflict_var = lhs.clone();
        let conflict = ConflictAst::Variable { conflict_var, set: set.def.clone() };
        let levels = vec![(lhs, set.clone(), vec![]),(rhs, set, vec![conflict])];
        LoopNest { levels: levels, triangular: true }
    }

    /// Builds a level of the loop nest.
    fn build_level(var: ir::Variable, ctx: &Context<'a>,
                       outer_vars: &mut Vec<Conflict<'a>>,
                       skip_new_objs: bool)
        -> (Variable<'a>, Set<'a>, Vec<ConflictAst<'a>>)
    {
        let (name, set) = ctx.var_def(var);
        let mut conflicts = outer_vars.iter().flat_map(|conflict| {
            conflict.generate_ast(set, ctx)
        }).collect_vec();
        if skip_new_objs {
            let list = new_objs_list(set.def(), "new_objs");
            conflicts.push(ConflictAst::NewObjs { list, set: Set::new(set, ctx) });
        }
        outer_vars.push(Conflict::new(name.clone(), set));
        (name, Set::new(set, ctx), conflicts)
    }

}

/// Performs variable substitution in a piece of rust code.
pub fn code<'a>(code: &ir::Code, ctx: &Context) -> String {
    let mut s = code.code.to_string();
    s = s.replace("$fun", "ir_instance");
    for &(sub, ref var) in &code.vars {
        let sub = ctx.var_name(sub);
        s = s.replace(&("$".to_string() + var), sub.name());
    }
    s
}

impl<'a> Display for Variable<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result { self.name().fmt(f) }
}

/// The type of a value.
#[derive(Serialize)]
pub enum ValueType { Enum(RcStr), Range, HalfRange, NumericSet(String) }

impl ValueType {
    pub fn new(t: ir::ValueType, ctx: &Context) -> Self {
        match t {
            ir::ValueType::Enum(name) => ValueType::Enum(name.clone()),
            ir::ValueType::Range => ValueType::Range,
            ir::ValueType::HalfRange => ValueType::HalfRange,
            ir::ValueType::NumericSet(univers) =>
                ValueType::NumericSet(code(&univers, ctx)),
        }
    }
}

impl Display for ValueType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            ValueType::Enum(ref name) => name,
            ValueType::Range => "Range",
            ValueType::HalfRange => "HalfRange",
            ValueType::NumericSet(..) => "NumericSet",
        }.fmt(f)
    }
}

impl Display for ir::CounterKind {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            ir::CounterKind::Add => "+",
            ir::CounterKind::Mul => "*",
        }.fmt(f)
    }
}

/// Creates a variable containing the list of newly created objects of the given set.
pub fn new_objs_list(set: &ir::SetDef, new_objs: &str) -> Variable<'static> {
    let name = render!(set/new_objs, <'a>,
                       def: SetDef<'a> = SetDef::new(set),
                       objs: &'a str = new_objs);
    Variable::with_string(name)
}

/// AST to print the reference to a set.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize)]
pub struct Set<'a> {
    def: SetDef<'a>,
    var: Option<Variable<'a>>,
    constraints: Vec<Set<'a>>,
}

impl<'a> Set<'a> {
    pub fn new<S: ir::SetRef<'a>>(set: S, ctx: &Context<'a>) -> Self {
        if let Some(rev_set) = set.reverse_constraint() {
            let mut rev_set_ast = Set::new(rev_set.clone(), ctx);
            rev_set_ast.constraints = set.without_reverse_constraints()
                .path_to_superset(&rev_set.superset().unwrap())
                .into_iter().rev().map(|s| Set::new(s, ctx)).collect();
            rev_set_ast
        } else {
            let var = set.arg().map(|v| ctx.var_name(v));
            Set { def: SetDef::new(set.def()), var, constraints: vec![] }
        }
    }
}

/// AST for the set definition.
#[derive(Debug, Clone, Serialize)]
pub struct SetDef<'a> {
    name: &'a str,
    keys: &'a IndexMap<ir::SetDefKey, String>,
    arg: Option<Box<SetDef<'a>>>,
}

impl<'a> SetDef<'a> {
    pub fn new(def: &'a ir::SetDef) -> Self {
        SetDef {
            name: def.name(),
            keys: def.attributes(),
            arg: def.arg().map(|arg| Box::new(SetDef::new(arg.def()))),
        }
    }

    fn name(&self) -> &str { self.name }
}

hash_from_key!(SetDef<'a>, SetDef::name, 'a);

/// Fetches the variables names and zips them with the set they must be on to be passed
/// as argument to the given choice.
pub fn vars_with_sets<'a>(choice: &'a ir::Choice, vars: &[ir::Variable],
                          ctx: &Context<'a>) -> Vec<(Variable<'a>, Set<'a>)> {
    vars.iter().map(|&v| ctx.var_name(v))
        .zip_eq(choice.arguments().sets().map(|s| Set::new(s, ctx))).collect()
}
