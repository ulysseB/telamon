use ir;
use utils::*;

/// Represent a transformation to apply to a rule to fir it in a new context.
#[derive(Default)]
pub struct Adaptator {
    variables: HashMap<ir::Variable, ir::Variable>,
    inputs: HashMap<usize, (usize, bool)>,
}

impl Adaptator {
    /// Creates an adaptator that maps the arguments to the given names.
    pub fn from_arguments(args: &[ir::Variable]) -> Self {
        let variables = args
            .iter()
            .enumerate()
            .map(|(id, &v)| (ir::Variable::Arg(id), v))
            .collect();
        Adaptator {
            variables,
            ..Default::default()
        }
    }

    /// Adapts a variable.
    pub fn variable(&self, var: ir::Variable) -> ir::Variable {
        self.variables.get(&var).cloned().unwrap_or(var)
    }

    /// Returns the new ID of an input and indicates if it is inversed.
    pub fn input(&self, input: usize) -> (usize, bool) {
        self.inputs.get(&input).cloned().unwrap_or((input, false))
    }

    /// Sets the mapping of a variable. Returns the previous mapping.
    pub fn set_variable(
        &mut self,
        old: ir::Variable,
        new: ir::Variable,
    ) -> Option<ir::Variable>
    {
        self.variables.insert(old, new)
    }

    /// Sets the mapping of an input.
    pub fn set_input(&mut self, old: usize, new: usize) {
        self.inputs.entry(old).or_insert((old, false)).0 = new;
    }

    /// Sets an input as inversed.
    pub fn set_inversed(&mut self, input: usize) {
        self.inputs.entry(input).or_insert((input, false)).1 = true;
    }
}

pub trait Adaptable {
    /// Adapts the object to the new environement.
    fn adapt(&self, adaptator: &Adaptator) -> Self;
}

impl<T> Adaptable for Vec<T>
where T: Adaptable
{
    fn adapt(&self, adaptator: &Adaptator) -> Self {
        self.iter().map(|x| x.adapt(adaptator)).collect()
    }
}
