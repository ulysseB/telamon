use super::*;
use std::ops::Deref;

/// CheckContext is a type system.
#[derive(Debug, Default)]
pub struct CheckerContext {
    /// Map Name of unique identifiant.
    hash_set: HashMap<String, Spanned<Hint>>,
    hash_choice: HashMap<String, Spanned<Hint>>,
}

impl CheckerContext {

    /// This checks the redefinition of SetDef.
    pub fn declare_set(&mut self, set: &SetDef) -> Result<(), TypeError> {
        if let Some(pre) = self.hash_set.insert(
            set.name.data.to_owned(), set.name.with_data(Hint::Set)
        ) {
            Err(TypeError::Redefinition { object_kind: pre, object_name: set.name.to_owned() })
        } else {
            Ok(())
        }
    }

    /// This checks the redefinition of ChoiceDef (EnumDef and IntegerDef).
    pub fn declare_choice(&mut self, choice: &ChoiceDef) -> Result<(), TypeError> {
        if let Some(pre) = self.hash_choice.insert(
            choice.get_name().data.to_owned(), choice.get_name().with_data(Hint::from_choice(choice))
        ) {
            Err(TypeError::Redefinition { object_kind: pre, object_name: choice.get_name() })
        } else {
            Ok(())
        }
    }

    /// This checks the undefined of SetDef superset and arg.
    fn check_set_define(&self, statement: &SetDef) -> Result<(), TypeError> {
        match statement {
            SetDef { name: Spanned { beg, end, data: ref name },
            doc: _, arg, superset, disjoint: _, keys, ..  } => {
                if let Some(VarDef { name: _, set: SetRef { name, .. } }) = arg {
                    let name: &String = name.deref();
                    if !self.hash_set.contains_key(name) {
                        Err(TypeError::Undefined {
                            object_name: Spanned {
                                beg: beg.to_owned(), end: end.to_owned(),
                                data: name.to_owned(),
                            }
                        })?;
                    }
                }
                if let Some(SetRef { name: supername, .. }) = superset {
                    let name: &String = supername.deref();
                    if !self.hash_set.contains_key(name) {
                        Err(TypeError::Undefined {
                            object_name: Spanned {
                                beg: beg.to_owned(), end: end.to_owned(),
                                data: name.to_owned(),
                            }
                        })?;
                    }
                }
            },
        }
        Ok(())
    }

    /// This checks the undefined of EnumDef or IntegerDef.
    fn check_choice_define(&self, statement: ChoiceDef) -> Result<(), TypeError> {
        match statement {
            ChoiceDef::EnumDef(EnumDef {
                name: Spanned { ref beg, ref end, data: _ },
                doc: _, ref variables,
            .. }) |
            ChoiceDef::IntegerDef(IntegerDef {
                name: Spanned { ref beg, ref end, data: _ }, doc: _, ref variables,
            .. }) => {
                for VarDef { name: _, set: SetRef { name, .. } } in variables {
                    let name: &String = name.deref();
                    if !self.hash_set.contains_key(name) {
                        Err(TypeError::Undefined {
                            object_name: Spanned {
                                beg: beg.to_owned(), end: end.to_owned(),
                                data: name.to_owned(),
                            }
                        })?;
                    }
                }
            },
            _ => {},
        }
        Ok(())
    }
    
    /// Type checks the declare's condition.
    pub fn declare(&mut self, statement: &Statement) -> Result<(), TypeError> {
        match statement {
            Statement::SetDef(def) => self.declare_set(def),
            Statement::ChoiceDef(def) => self.declare_choice(def),
            _ => Ok(()),
        }
    }
    
    /// Type checks the define's condition.
    pub fn define(&mut self, statement: &Statement) -> Result<(), TypeError> {
        match statement {
            Statement::ChoiceDef(ChoiceDef::EnumDef(ref enumeration)) => {
                self.check_choice_define(
                    ChoiceDef::EnumDef(enumeration.clone()))?;
            },
            Statement::ChoiceDef(
                ChoiceDef::IntegerDef(ref integer)
            ) => {
                self.check_choice_define(
                    ChoiceDef::IntegerDef(integer.clone()))?;
            },
            Statement::SetDef(ref set) => {
                self.check_set_define(set)?;
            },
            _ => {},
        }
        Ok(())
    }
}
