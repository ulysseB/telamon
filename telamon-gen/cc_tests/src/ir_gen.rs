//! Defines a fake IR.

/// Defines an IR with the given sets.
macro_rules! define_ir {
    // Define the entry point.
    (type $($rest:tt)*) => { define_ir!(@stack [], type $($rest)*); };
    (trait $($rest:tt)*) => { define_ir!(@stack [], trait $($rest)*); };
    (struct $($rest:tt)*) => { define_ir!(@stack [], struct $($rest)*); };
    () => { define_ir!(@stack [],); };
    // Build the stack
    (@stack [$($stack:tt)*], $kind:tt $name:ident; $($rest:tt)*) => {
        define_ir!(@stack [[$kind $name,@,@,no,@], $($stack)*], $($rest)*);
    };
    (@stack [$($stack:tt)*], $kind:tt $name:ident[$param:ident]; $($rest:tt)*) => {
        define_ir!(@stack [[$kind $name,$param,@,no,@], $($stack)*], $($rest)*);
    };
    (@stack [$($stack:tt)*], $kind:tt $name:ident : $super:ident; $($rest:tt)*) => {
        define_ir!(@stack [[$kind $name,@,$super,no,@], $($stack)*], $($rest)*);
    };
    (@stack [$($stack:tt)*], $kind:tt $name:ident[$param:ident reverse $reverse:ident]:
        $super:ident; $($rest:tt)*) =>
    {
        define_ir!(@stack [[$kind $name,$param,$super,no,$reverse], $($stack)*], $($rest)*);
    };
    (@stack [$($stack:tt)*],
     $kind:tt $name:ident[$param:ident reverse $reverse:ident]:
        $super:ident[$x:ident]; $($rest:tt)*) =>
    {
        define_ir!(@stack [[$kind $name,$param,$super,yes,$reverse], $($stack)*], $($rest)*);
    };
    // Print the ir from the stack.
    (@stack [$([$kind:tt $name:ident, $param:tt,
               $super:tt, $super_param:tt, $reverse:tt],)*],) => {
        mod ir {
            #[allow(unused_imports)]
            use utils::*;

            $(pub mod $name {
                define_ir!(@id_def $kind $name, $super);
                define_ir!(@obj_def $kind $name, $super);
                define_ir!(@impl_super $kind $super);
                define_ir!(@get_iter_def $name, $param);
                define_ir!(@from_superset $name, $param, $super);
                define_ir!(@create_def $kind $name, $param, $super, $super_param);
                define_ir!(@reverse_def $name, $param, $super, $reverse);
            })*

            #[derive(Default, Clone)]
            pub struct Function {
                next_id: usize,
                $($name: HashMap<define_ir!(@full_id $name, $param),
                                 ::std::rc::Rc<$name::Obj>>,)*
            }

            #[derive(Default)]
            pub struct NewObjs {
                $(pub $name: Vec<define_ir!(@full_id $name, $param)>,)*
            }

            pub mod prelude {
                pub use super::NewObjs;
                $(define_ir!(@prelude $kind $name);)*
            }
        }
    };
    // Returns the Id of (param::Id, Id) if the set has a parameter.
    (@full_id $name:ident, @) => { $name::Id };
    (@full_id $name:ident, $param:ident) => { ($param::Id, $name::Id) };
    // Defines the set ID.
    (@id_def trait $name:ident, $x:tt) => { define_ir!(@id_def struct $name, $x); };
    (@id_def struct $name:ident, $x:tt) => {
        #[derive(Clone, Copy, Hash, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct Id(pub usize);
    };
    (@id_def type $name:ident, $super:ident) => { pub use super::$super::Id as Id; };
    (@id_def type $name:ident, @) => { compile_error!("type sets must have a superset"); };
    // Defines the set item type.
    (@obj_def struct $name:ident, $super:tt) => {
        #[derive(Clone)]
        pub struct Obj { pub(super) id: Id, pub(super) cond: ::std::cell::Cell<bool> }

        impl Obj {
            pub fn id(&self) -> Id { self.id }
            pub fn condition(&self) -> bool { self.cond.get() }
            pub fn set_condition(&self) { self.cond.set(true) }
        }
    };
    (@obj_def trait $name:ident, $super:tt) => {
        pub trait Obj {
            fn id(&self) -> Id;
            fn condition(&self) -> bool;
            fn set_condition(&self);
        }
    };
    (@obj_def type $name:ident, $super:ident) => { pub use super::$super::Obj; };
    // Defines the iterator and the getter for a set.
    (@get_iter_def $name:ident, @) => {
        pub fn get(fun: &super::Function, id: Id) -> &Obj { &*fun.$name[&id] }

        pub fn iter(fun: &super::Function) -> impl Iterator<Item=&Obj> {
            fun.$name.values().map(|x| &**x)
        }
    };
    (@get_iter_def $name:ident, $param:ident) => {
        pub fn get<'a>(fun: &'a super::Function,
                       param: &super::$param::Obj, id: Id) -> &'a Obj {
            &*fun.$name[&(param.id(), id)]
        }

        pub fn iter(fun: &super::Function, param: super::$param::Id)
            -> impl Iterator<Item=&Obj>
        {
            fun.$name.iter().filter(move |&(id, _)| id.0 == param).map(|x| &**x.1)
        }
    };
    // Implements the superset trait for a set object.
    (@impl_super struct $super:ident) => {
        impl super::$super::Obj for Obj {
            define_ir!(@impl_super_methods $super);
        }
        define_ir!(@from_super_id $super);
    };
    (@impl_super trait $super:ident) => {
        impl<T: Obj + ?Sized> super::$superset::Obj for T {
            define_ir!(@impl_super_methods $super);
        }
        define_ir!(@from_super_id $super);
    };
    (@impl_super type $super:tt) => { };
    (@impl_super $kind:tt @) => { };
    (@impl_super_methods $super:ident) => {
        fn id(&self) -> super::$super::Id { Obj::id(self).into() }
        fn condition(&self) -> bool { Obj::condition(self) }
        fn set_condition(&self) { Obj::set_condition(self) }
    };
    (@from_super_id $super:ident) => {
        impl From<Id> for super::$super::Id {
            fn from(id: Id) -> super::$super::Id { super::$super::Id(id.0) }
        }
    };
    // Implements the filter from the superset.
    (@from_superset $name:ident, $param:tt, @) => { };
    (@from_superset $name:ident, @, $super:ident) => {
        pub fn from_superset<'a>(fun: &'a super::Function,
                                 obj: &super::$super::Obj) -> Option<&'a Obj> {
            fun.$name.get(&Id(obj.id().0)).map(|x| &**x)
        }
    };
    (@from_superset $name:ident, $param:ident, $super:ident) => {
        pub fn from_superset<'a>(fun: &'a super::Function,
                                 param: &super::$param::Obj,
                                 obj: &super::$super::Obj) -> Option<&'a Obj> {
            fun.$name.get(&(param.id(), Id(obj.id().0))).map(|x| &**x)
        }
    };
    // Imports traits into the prelude.
    (@prelude trait $name:ident) => { pub use super::$name::Obj as $name; };
    (@prelude struct $name:ident) => { };
    (@prelude type $name:ident) => { };
    // Defines the set object constructor.
    (@create_def struct $name:ident, @, $($rest:tt)*) => {
        pub fn create(fun: &mut super::Function, cond: bool) -> Id {
            let id = Id(fun.next_id);
            fun.next_id += 1;
            let obj = ::std::rc::Rc::new(Obj { id, cond: ::std::cell::Cell::new(cond) });
            define_ir!(@create_superset, fun, obj, id, @, $($rest)*);
            fun.$name.insert(id, obj);
            id
        }

        pub fn create_super(fun: &mut super::Function, id: Id,
                           obj: ::std::rc::Rc<Obj>) {
            define_ir!(@create_superset, fun, obj, id, @, $($rest)*);
            fun.$name.insert(id, obj);
        }
    };
    (@create_def struct $name:ident, $param:ident, $($rest:tt)*) => {
        pub fn create(fun: &mut super::Function, param: super::$param::Id,
                      cond: bool) -> Id {
            let id = Id(fun.next_id);
            fun.next_id += 1;
            let obj = ::std::rc::Rc::new(Obj { id, cond: ::std::cell::Cell::new(cond) });
            define_ir!(@create_superset, fun, obj, id, param, $($rest)*);
            fun.$name.insert((param, id), obj);
            id
        }

        pub fn create_super(fun: &mut super::Function, param: super::$param::Id, id: Id,
                            obj: ::std::rc::Rc<Obj>) {
            define_ir!(@create_superset, fun, obj, id, param, $($rest)*);
            fun.$name.insert((param, id), obj);
        }
    };
    (@create_def trait $name:ident, @, $($rest:tt)*) => {
        pub fn create_super<T>(fun: &mut super::Function, id: Id,
                         obj: ::std::rc::Rc<T>) where T: Obj + 'static {
            define_ir!(@create_superset, fun, obj, id, @, $($rest)*);
            fun.$name.insert(id, obj);
        }
    };
    (@create_def trait $name:ident, $param:ident, $($rest:tt)*) => {
        pub fn create_super<T>(fun: &mut super::Function, param: super::$param::Id, id: Id,
                               obj: ::std::rc::Rc<T>) where T: Obj + 'static {
            define_ir!(@create_superset, fun, obj, id, param, $($rest)*);
            fun.$name.insert((param, id), obj);
        }
    };
    (@create_def type $name:ident, @, $super:ident, no) => {
        pub fn add_to_subset(fun: &mut super::Function, id: Id) -> bool {
            let obj = fun.$super[&id].clone();
            fun.$name.insert(id, obj).is_none()
        }

        define_ir!(@create_def struct $name, @, $super, no);
    };
    (@create_def type $name:ident, $param:ident, $super:ident, no) => {
        pub fn add_to_subset(fun: &mut super::Function, param: super::$param::Id, id: Id) -> bool {
            let obj = fun.$super[&id].clone();
            fun.$name.insert((param, id), obj).is_none()
        }

        define_ir!(@create_def struct $name, $param, $super, no);
    };
    (@create_def type $name:ident, $param:ident, $super:ident, yes) => {
        pub fn add_to_subset(fun: &mut super::Function, param: super::$param::Id, id: Id) -> bool {
            let obj = fun.$super[&(param.into(), id)].clone();
            fun.$name.insert((param, id), obj).is_none()
        }

        define_ir!(@create_def struct $name, $param, $super, yes);
    };
    (@create_superset, $fun:ident, $obj:ident, $id:ident, $param:tt, @, no) => { };
    (@create_superset, $fun:ident, $obj:ident, $id:ident, $param:tt, $super:ident, no) => {
        super::$super::create_super($fun, $id.into(), $obj.clone());
    };
    (@create_superset, $fun:ident, $obj:ident, $id:ident, $param:tt, $super:ident, yes) => {
        super::$super::create_super($fun, $param.into(), $id.into(), $obj.clone());
    };
    // Defines the reverse iterator.
    (@reverse_def $name:ident, @, $($rest:tt)*) => { };
    (@reverse_def $name:ident, $param:ident, @, $($rest:tt)*) => { };
    (@reverse_def $name:ident, $param:ident, $super:ident, $reverse:ident) => {
        pub fn reverse(fun: &super::Function, reverse_param: super::$reverse::Id)
            -> impl Iterator<Item=&super::$param::Obj>
        {
            fun.$name.keys().filter(move |&&(_, id)| id.0 == reverse_param.0)
                .map(move |&(p_id, _)| &*fun.$param[&p_id])
        }
    };
}
