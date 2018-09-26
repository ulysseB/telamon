use super::*;
use std::ops::Deref;

/// A toplevel integer
#[derive(Clone, Debug)]
pub struct IntegerDef {
    pub name: Spanned<String>,
    pub doc: Option<String>,
    pub variables: Vec<VarDef>,
    pub code: String, // varmap, type_check_code
}

impl IntegerDef {
    /// This checks if the variables are defined in the context.
    fn check_undefined_variables(
        &self,
        context: &CheckerContext,
    ) -> Result<(), TypeError> {
        for VarDef { name: _, ref set } in self.variables.iter() {
            if !context.check_set_define(set) {
                let name: &String = set.name.deref();

                Err(TypeError::Undefined {
                    object_name: self.name.with_data(name.to_owned()),
                })?;
            }
        }
        Ok(())
    }

    /// Type checks the declare's condition.
    pub fn declare(&self, context: &mut CheckerContext) -> Result<(), TypeError> {
        Ok(())
    }

    /// Defines an integer choice.
    fn define_integer(&self, tc: &mut TypingContext) {
        let choice_name = RcStr::new(self.name.data.to_owned());
        let doc = self.doc.to_owned().map(RcStr::new);
        let mut var_map = VarMap::default();
        let vars = self
            .variables
            .iter()
            .map(|v| {
                let name = v.name.clone();
                (name, var_map.decl_argument(&tc.ir_desc, v.to_owned()))
            }).collect::<Vec<_>>();
        let arguments = ir::ChoiceArguments::new(
            vars.into_iter()
                .map(|(n, s)| (n.data, s))
                .collect::<Vec<_>>(),
            false,
            false,
        );
        let universe = type_check_code(RcStr::new(self.code.to_owned()), &var_map);
        let choice_def = ir::ChoiceDef::Number { universe };
        tc.ir_desc
            .add_choice(ir::Choice::new(choice_name, doc, arguments, choice_def));
    }

    /// Type checks the define's condition.
    pub fn define(
        self,
        context: &mut CheckerContext,
        tc: &mut TypingContext,
    ) -> Result<(), TypeError> {
        self.check_undefined_variables(context)?;

        self.define_integer(tc);
        Ok(())
    }
}

impl PartialEq for IntegerDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
