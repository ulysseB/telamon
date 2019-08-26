use std::ops::Deref;

use crate::ast::context::CheckerContext;
use crate::ast::error::TypeError;
use crate::ast::{type_check_code, VarDef, VarMap};
use crate::ir;
use crate::lexer::Spanned;

use utils::RcStr;

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
        for VarDef { ref set, .. } in self.variables.iter() {
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
    fn define_integer(&self, ir_desc: &mut ir::IrDesc) {
        let choice_name = RcStr::new(self.name.data.to_owned());
        let doc = self.doc.to_owned().map(RcStr::new);
        let mut var_map = VarMap::default();
        let vars = self
            .variables
            .iter()
            .map(|v| {
                let name = v.name.clone();
                (name, var_map.decl_argument(&ir_desc, v.to_owned()))
            })
            .collect::<Vec<_>>();
        let arguments = ir::ChoiceArguments::new(
            vars.into_iter()
                .map(|(n, s)| (n.data, s))
                .collect::<Vec<_>>(),
            false,
            false,
        );
        let universe = type_check_code(RcStr::new(self.code.to_owned()), &var_map);
        let choice_def = ir::ChoiceDef::Number { universe };
        ir_desc.add_choice(ir::Choice::new(choice_name, doc, arguments, choice_def));
    }

    /// Type checks the define's condition.
    pub fn define(
        self,
        context: &mut CheckerContext,
        ir_desc: &mut ir::IrDesc,
    ) -> Result<(), TypeError> {
        self.check_undefined_variables(context)?;

        self.define_integer(ir_desc);
        Ok(())
    }
}

impl PartialEq for IntegerDef {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name
    }
}
