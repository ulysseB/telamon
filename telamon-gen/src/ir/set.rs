use ir::{self, Adaptable};
use std;
use std::fmt;
use std::borrow::Borrow;
use utils::*;
use indexmap::IndexMap;

/// Generic trait for sets.
pub trait SetRef<'a> {
    /// Returns the set definition.
    fn def(&self) -> &'a SetDef;

    /// Returns the argument of the set, if any.
    fn arg(&self) -> Option<ir::Variable>;

    /// A constraint on the variables to iterate on, issued from a set reversal.
    fn reverse_constraint(&self) -> Option<ir::SetRefImpl<'a>>;

    /// Returns the same set but without reverse constraints.
    fn without_reverse_constraints(&self) -> ir::SetRefImpl<'a> {
        SetRefImpl { reverse_constraint: None, .. self.as_ref() }
    }

    /// Returns the direct superset of this set, if any.
    fn superset(&self) -> Option<SetRefImpl<'a>> {
        self.def().superset.as_ref().map(|&Set { ref def, var, .. }| {
            let var = var.map(|v| {
                assert_eq!(v, ir::Variable::Arg(0));
                self.arg().unwrap()
            });
            SetRefImpl { def, var, reverse_constraint: None }
        })
    }

    /// Returns the path of sets to access a super-set.
    fn path_to_superset(&self, superset: &SetRef) -> Vec<SetRefImpl<'a>> {
        let mut out = Vec::new();
        let mut current = self.as_ref();
        while current != superset.as_ref() {
            out.push(current.clone());
            current = current.superset().unwrap();
        }
        out
    }

    /// Indicates if the first set is a sub-set of the second.
    fn is_subset_of(&self, other: &Set) -> bool {
        let is_subset = self.as_ref() == other.as_ref()
            || self.superset().map(|s| s.is_subset_of(other)).unwrap_or(false)
            || self.reverse_constraint().map_or(false, |s| s.is_subset_of(other));
        is_subset && other.reverse_constraint.is_none()
    }

    /// Returns the `SetRefImpl` corresponding to this set.
    fn as_ref(&self) -> SetRefImpl<'a> {
        let reverse_constraint = self.reverse_constraint().map(Box::new);
        SetRefImpl { def: self.def(), var: self.arg(), reverse_constraint }
    }
}

/// References a set of objects.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Set {
    def: std::rc::Rc<SetDef>,
    reverse_constraint: Option<Box<Set>>,
    var: Option<ir::Variable>,
}

impl Set {
    /// Create a new set instance.
    pub fn new(def: &std::rc::Rc<SetDef>, var: Option<ir::Variable>) -> Self {
        assert_eq!(var.is_some(), def.arg().is_some());
        Set { def: def.clone(), var, reverse_constraint: None }
    }

    /// Indicates if the first set is a sub-set of the second,
    /// without matching argument names.
    pub fn is_subset_of_def(&self, other: &Set) -> bool {
        self.reverse_constraint.as_ref().map_or(false, |s| s.is_subset_of_def(other))
            || self.def.is_subset_of_def(other.def())
    }

    /// Returns the common superset where the two set might have an object in common.
    pub fn get_collision_level<'a>(&'a self, rhs: &Set) -> Option<&'a SetDef> {
        // Find a path to a common ancestor
        let mut lhs = (&self.def, self.var);
        let mut rhs = (&rhs.def, rhs.var);
        let mut lhs_to_superset = Vec::new();
        let mut rhs_to_superset = Vec::new();
        while lhs != rhs {
            if lhs.0.depth > rhs.0.depth {
                lhs_to_superset.push(lhs.0);
                let superset = lhs.0.superset.as_ref().unwrap();
                let var = superset.var.map(|v| {
                    assert_eq!(v, ir::Variable::Arg(0));
                    lhs.1.unwrap()
                });
                lhs = (&superset.def, var);
            } else if let Some(ref superset) = rhs.0.superset {
                rhs_to_superset.push(rhs.0);
                let var = superset.var.map(|v| {
                    assert_eq!(v, ir::Variable::Arg(0));
                    rhs.1.unwrap()
                });
                rhs = (&superset.def, var);
            } else { return None; }
        }
        // Check the paths to the superset does not contains disjoint sets.
        for on_lhs_path in lhs_to_superset {
            for &on_rhs_path in &rhs_to_superset {
                for lhs_disjoint in &on_lhs_path.disjoints {
                    if on_rhs_path.name() as &str == lhs_disjoint { return None; }
                }
                for rhs_disjoint in &on_rhs_path.disjoints {
                    if on_lhs_path.name() as &str == rhs_disjoint { return None; }
                }
            }
        }
        // Return the common superset, that sis both in lhs and rhs.
        Some(lhs.0)
    }

    /// Returns a superset of this set and a set parametrized by elements of the superset
    /// that iterates on the possible parameters of this set given a variable of the
    /// superset.
    pub fn reverse(&self, self_var: ir::Variable, arg: &Set) -> Option<(Set, Set)> {
        if let Some(reverse_def) = self.def.reverse() {
            let superset = self.reverse_constraint.as_ref().map(|b| b.borrow())
                .unwrap_or(reverse_def.arg().unwrap()).clone();
            assert!((&superset).arg().is_none());
            let mut reverse = Set::new(&reverse_def, Some(self_var));
            if !(&reverse).is_subset_of(arg) {
                reverse.reverse_constraint = Some(Box::new(arg.clone()));
            }
            Some((superset, reverse))
        } else { None }
    }
}

impl Adaptable for Set {
    fn adapt(&self, adaptator: &ir::Adaptator) -> Self {
        let reverse_constraint = self.reverse_constraint.as_ref()
            .map(|set| Box::new(set.adapt(adaptator)));
        Set {
            def: self.def.clone(),
            var: self.var.map(|v| adaptator.variable(v)),
            reverse_constraint,
        }
    }
}

impl<'a> SetRef<'a> for &'a Set {
    fn def(&self) -> &'a SetDef { &self.def }

    fn arg(&self) -> Option<ir::Variable> { self.var }

    fn reverse_constraint(&self) -> Option<ir::SetRefImpl<'a>> {
        self.reverse_constraint.as_ref().map(|s| SetRef::as_ref(&s.as_ref()))
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct SetRefImpl<'a> {
    def: &'a SetDef,
    var: Option<ir::Variable>,
    reverse_constraint: Option<Box<ir::SetRefImpl<'a>>>,
}

impl<'a> SetRef<'a> for SetRefImpl<'a> {
    fn def(&self) -> &'a SetDef { self.def }

    fn arg(&self) -> Option<ir::Variable> { self.var }

    fn reverse_constraint(&self) -> Option<ir::SetRefImpl<'a>> {
        self.reverse_constraint.as_ref().map(|s| s.as_ref().clone())
    }
}

/// Defines a set of objects.
#[derive(Clone)]
pub struct SetDef {
    name: RcStr,
    arg: Option<ir::Set>,
    superset: Option<Set>,
    reverse: ReverseSet,
    keys: IndexMap<SetDefKey, String>,
    depth: usize,
    def_order: usize,
    disjoints: Vec<String>,
}

impl SetDef {
    /// Creates a new set definition.
    pub fn new(name: String,
               arg: Option<ir::Set>,
               superset: Option<Set>,
               reverse: Option<(Set, String)>,
               keys: IndexMap<SetDefKey, String>,
               disjoints: Vec<String>) -> std::rc::Rc<Self> {
        let name = RcStr::new(name);
        let reverse = if let Some((set, iter)) = reverse {
            let name = RcStr::default();
            let reverse = ReverseSet::None;
            let superset = arg.as_ref().unwrap().def();
            let from_superset = keys[&SetDefKey::FromSuperset].replace("$item", "$tmp")
                .replace("$var", "$item").replace("$tmp", "$var") + ".map(|_| $item)";
            let reverse_keys = vec![
                (SetDefKey::ItemType, superset.keys[&SetDefKey::ItemType].clone()),
                (SetDefKey::IdType, superset.keys[&SetDefKey::IdType].clone()),
                (SetDefKey::ItemGetter, superset.keys[&SetDefKey::ItemGetter].clone()),
                (SetDefKey::IdGetter, superset.keys[&SetDefKey::IdGetter].clone()),
                (SetDefKey::Iter, iter),
                (SetDefKey::FromSuperset, from_superset),
            ].into_iter().collect();
            for key in &[SetDefKey::ItemType, SetDefKey::IdType] {
                assert_eq!(set.def.keys[key], keys[key],
                   "reverse supersets must use the same id and item types than the set");
            }
            let def = SetDef::build(
                name, Some(set), arg.clone(), reverse, reverse_keys, vec![]);
            ReverseSet::Explicit(std::cell::RefCell::new(std::rc::Rc::new(def)))
        } else { ReverseSet::None };
        let def = SetDef::build(name, arg, superset, reverse, keys, disjoints);
        let def = std::rc::Rc::new(def);
        if let ReverseSet::Explicit(ref cell) = def.reverse {
            let mut rc = cell.borrow_mut();
            let reverse = std::rc::Rc::get_mut(&mut *rc).unwrap();
            reverse.reverse = ReverseSet::Implicit(std::rc::Rc::downgrade(&def));
        }
        def
    }

    /// Creates a new set definition.
    fn build(name: RcStr,
             arg: Option<ir::Set>,
             superset: Option<Set>,
             reverse: ReverseSet,
             keys: IndexMap<SetDefKey, String>,
             disjoints: Vec<String>) -> Self {
        let depth = superset.as_ref().map(|s| s.def.depth + 1).unwrap_or(0);
        let def_order = arg.as_ref().map(|s| s.def.def_order + 1).unwrap_or(0);
        SetDef { name, arg, superset, keys, disjoints, depth, def_order, reverse }
    }

    /// The name of the set.
    pub fn name(&self) -> &RcStr { &self.name }

    /// Returns the argument of the set, if any.
    pub fn arg(&self) -> Option<&ir::Set> { self.arg.as_ref() }

    /// Returns the superset of the set, if any.
    pub fn superset(&self) -> Option<&ir::Set> { self.superset.as_ref() }

    /// The attributes of the set.
    pub fn attributes(&self) -> &IndexMap<SetDefKey, String> { &self.keys }

    /// Suggest a prefix for variables in the set.
    pub fn prefix(&self) -> &str {
        self.keys.get(&SetDefKey::Prefix).map(|s| s as &str).unwrap_or("obj")
    }

    /// Returns an integer that indicates an order in which variables can be defined
    /// to always be defined before any argument of the set they belong into.
    pub fn def_order(&self) -> usize { self.def_order }

    /// Returns the reverse set, for sets that have both a parameter and a superset.
    fn reverse(&self) -> Option<std::rc::Rc<SetDef>> {
        match self.reverse {
            ReverseSet::None => None,
            ReverseSet::Explicit(ref cell) => Some(cell.borrow().clone()),
            ReverseSet::Implicit(ref rc) => Some(std::rc::Weak::upgrade(rc).unwrap()),
        }
    }

    /// Indicates if the first set is a sub-set of the second.
    pub fn is_subset_of_def(&self, other: &SetDef) -> bool {
        self == other || self.superset.as_ref().map(|s| {
            s.def.is_subset_of_def(other)
        }).unwrap_or(false)
    }

}

impl std::fmt::Debug for SetDef {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

hash_from_key!(SetDef, SetDef::name);

impl PartialOrd for SetDef {
    fn partial_cmp(&self, other: &SetDef) -> Option<std::cmp::Ordering> {
        self.name.partial_cmp(&other.name)
    }
}

impl Ord for SetDef {
    fn cmp(&self, other: &SetDef) -> std::cmp::Ordering { self.name.cmp(&other.name) }
}

/// A set that lists the arguments of another set.
#[derive(Clone)]
enum ReverseSet {
    /// The reverse relation was defined in the source code.
    Explicit(std::cell::RefCell<std::rc::Rc<SetDef>>),
    /// The reverse relation was infered fromthe inverse relation.
    Implicit(std::rc::Weak<SetDef>),
    /// Their is no reverse set.
    None,
}

#[derive(Debug, Hash, PartialEq, Eq, Serialize, Copy, Clone)]
#[repr(C)]
pub enum SetDefKey {
    ItemType,
    IdType,
    ItemGetter,
    IdGetter,
    Iter,
    FromSuperset,
    Prefix,
    NewObjs,
    Reverse,
    AddToSet,
}

impl SetDefKey {
    /// Returns the variables defined for the key.
    pub fn env(&self) -> Vec<&'static str> {
        match *self {
            SetDefKey::ItemType | SetDefKey::IdType | SetDefKey::Prefix => vec![],
            SetDefKey::ItemGetter => vec!["fun", "id"],
            SetDefKey::IdGetter => vec!["fun", "item"],
            SetDefKey::Iter => vec!["fun"],
            SetDefKey::FromSuperset => vec!["fun", "item"],
            SetDefKey::NewObjs => vec!["objs"],
            SetDefKey::Reverse => vec!["fun"],
            SetDefKey::AddToSet => vec!["fun", "item"],
        }
    }

    /// Indicates if the environement contains the set argument.
    pub fn is_arg_in_env(&self) -> bool {
        match *self {
            SetDefKey::ItemGetter |
            SetDefKey::Iter |
            SetDefKey::FromSuperset |
            SetDefKey::AddToSet => true,
            _ => false,
        }
    }

    /// The list of required keys.
    pub const REQUIRED: [SetDefKey; 6] = [
        SetDefKey::ItemType,
        SetDefKey::IdType,
        SetDefKey::ItemGetter,
        SetDefKey::IdGetter,
        SetDefKey::Iter,
        SetDefKey::NewObjs,
    ];
}

impl fmt::Display for SetDefKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
            
