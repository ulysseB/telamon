//! Code generation for `ir::Variable`.
use crate::codegen;
use indexmap::IndexMap;
use crate::ir;
use crate::search_space::*;
use telamon_utils::*;

/// Wraps an `ir::Variable` to expose specified decisions.
pub struct Variable<'a> {
    variable: &'a ir::Variable,
    t: ir::Type,
    instantiation_dims: HashMap<ir::DimId, usize>,
    alias: Option<Alias>,
}

impl<'a> Variable<'a> {
    /// Returns the ID of the variable.
    pub fn id(&self) -> ir::VarId {
        self.variable.id()
    }

    /// Returns the type of the variable.
    pub fn t(&self) -> ir::Type {
        self.t
    }

    /// Indicates if the variable aliases with another.
    ///
    /// This just indicates a single alias. In practice, aliases can be chained so the
    /// variable may alias we multiple other variables.
    pub fn alias(&self) -> Option<&Alias> {
        self.alias.as_ref()
    }

    /// Returns the dimensions (and their size) along wich the variable is instantiated.
    pub fn instantiation_dims(&self) -> impl Iterator<Item = (ir::DimId, usize)> + '_ {
        self.instantiation_dims
            .iter()
            .map(|(&key, &value)| (key, value))
    }
}

/// Indicates how a variable aliases with another.
pub struct Alias {
    other_variable: ir::VarId,
    dim_mapping: HashMap<ir::DimId, Option<ir::DimId>>,
    reverse_mapping: HashMap<ir::DimId, ir::DimId>,
}

impl Alias {
    /// Indicates the variable aliased with.
    pub fn other_variable(&self) -> ir::VarId {
        self.other_variable
    }

    /// Specifies the mapping of dimensions from `other_variable` to the alias. A mapping
    /// to `None` indicates the alias takes the last instance of the variable on the
    /// dimension.
    pub fn dim_mapping(&self) -> &HashMap<ir::DimId, Option<ir::DimId>> {
        &self.dim_mapping
    }

    /// Creates a new `Alias` that takes the last value of another variable.
    fn new_last(other_variable: ir::VarId, dims: &[ir::DimId]) -> Self {
        Alias {
            other_variable,
            dim_mapping: dims.iter().map(|&dim| (dim, None)).collect(),
            reverse_mapping: Default::default(),
        }
    }

    /// Creates a new alias that takes point-to-point the values of another variable.
    fn new_dim_map(
        other_variable: ir::VarId,
        mapping_ids: &[ir::DimMappingId],
        fun: &ir::Function,
    ) -> Self {
        let (dim_mapping, reverse_mapping) = mapping_ids
            .iter()
            .map(|&id| fun.dim_mapping(id).dims())
            .map(|[lhs, rhs]| ((lhs, Some(rhs)), (rhs, lhs)))
            .unzip();
        Alias {
            other_variable,
            dim_mapping,
            reverse_mapping,
        }
    }

    /// Finds dimensions on which the variable must be instantiated to implement the
    /// aliasing. Also returns their size.
    fn find_instantiation_dims(&self, space: &SearchSpace) -> HashMap<ir::DimId, usize> {
        self.dim_mapping
            .iter()
            .flat_map(|(&lhs, &rhs)| rhs.map(|rhs| (lhs, rhs)))
            .filter(|&(lhs, rhs)| {
                space.domain().get_order(lhs.into(), rhs.into()) != Order::MERGED
            })
            .map(|(_, rhs)| {
                let size = space.ir_instance().dim(rhs).size();
                let int_size = unwrap!(codegen::Size::from_ir(size, space).as_int());
                (rhs, int_size as usize)
            })
            .collect()
    }
}

/// Generates variables aliases.
fn generate_aliases(space: &SearchSpace) -> HashMap<ir::VarId, Option<Alias>> {
    space
        .ir_instance()
        .variables()
        .map(|var| {
            let alias = match var.def() {
                ir::VarDef::Inst(..) => None,
                ir::VarDef::Last(alias, dims) => Some(Alias::new_last(*alias, dims)),
                ir::VarDef::DimMap(alias, mappings) => {
                    Some(Alias::new_dim_map(*alias, mappings, space.ir_instance()))
                }
            };
            (var.id(), alias)
        })
        .collect()
}

/// Sort variables by aliasing order.
fn sort_variables<'a>(
    mut aliases: HashMap<ir::VarId, Option<Alias>>,
    space: &'a SearchSpace,
) -> impl Iterator<Item = (&'a ir::Variable, Option<Alias>)> {
    space.ir_instance().variables().flat_map(move |var| {
        let mut reverse_aliases = vec![];
        let mut current_var = Some(var.id());
        // Each variable depends at most on one other, so we we just walk the chain of
        // dependencies until we encountered an already sorted variable or a variable
        // with no dependency. Then we insert them in reverse order in the sorted array.
        while let Some((id, alias)) =
            { current_var.and_then(|id| aliases.remove(&id).map(|alias| (id, alias))) }
        {
            current_var = alias.as_ref().map(|alias| alias.other_variable);
            reverse_aliases.push((space.ir_instance().variable(id), alias));
        }
        reverse_aliases.reverse();
        reverse_aliases
    })
}

/// Add a wrapper around variables to expose specified decisions. Orders variables with an
/// alias after the variable they depend on.
pub fn wrap_variables<'a>(space: &'a SearchSpace) -> Vec<Variable<'a>> {
    let aliases = generate_aliases(space);
    // Generate the wrappers and record the dimensions along which variables are
    // instantiated. We use an `IndexMap` so that the insertion order is preserved.
    let mut wrapped_vars = IndexMap::new();
    for (variable, alias_opt) in sort_variables(aliases, space) {
        let instantiation_dims = alias_opt
            .as_ref()
            .map(|alias| alias.find_instantiation_dims(space))
            .unwrap_or_default();
        // Register instantiation dimensions in preceeding aliases.
        if let Some(ref alias) = alias_opt {
            for (&iter_dim, &size) in &instantiation_dims {
                let alias_iter_dim = alias.reverse_mapping[&iter_dim];
                let mut current = Some((alias.other_variable, alias_iter_dim));
                while let Some((var_id, iter_dim)) = current.take() {
                    let var: &mut Variable = &mut wrapped_vars[&var_id];
                    // Only continue if the dimension was not already inserted.
                    if var.instantiation_dims.insert(iter_dim, size).is_none() {
                        if let Some(ref alias) = var.alias {
                            let pred_dim = alias
                                .reverse_mapping
                                .get(&iter_dim)
                                .cloned()
                                .unwrap_or(iter_dim);
                            current = Some((alias.other_variable, pred_dim));
                        }
                    }
                }
            }
        }
        let wrapper = Variable {
            variable,
            t: unwrap!(space.ir_instance().device().lower_type(variable.t(), space)),
            instantiation_dims,
            alias: alias_opt,
        };
        wrapped_vars.insert(variable.id(), wrapper);
    }
    wrapped_vars.into_iter().map(|entry| entry.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::x86::Cpu;
    use crate::helper;
    use crate::ir;
    use std;

    fn mk_map<K, V>(content: &[(K, V)]) -> HashMap<K, V>
    where
        K: Copy + Eq + std::hash::Hash,
        V: Copy,
    {
        content.iter().cloned().collect()
    }

    /// Ensures chained aliases work correctly.
    #[test]
    fn chained_alias() {
        let _ = ::env_logger::try_init();
        let device = Cpu::dummy_cpu();
        let signature = ir::Signature::new("test".to_string());
        let mut builder = helper::Builder::new(&signature, &device);
        // This code builds the following function:
        // ```pseudocode
        // for i in 0..16:
        //   for j in 0..16:
        //      src[i] = 0;
        // for i in 0..16:
        //   dst = src[i]
        // ```
        // where all loops are unrolled.
        let dim0 = builder.open_dim_ex(ir::Size::new_const(4), DimKind::UNROLL);
        let dim1 = builder.open_dim_ex(ir::Size::new_const(8), DimKind::UNROLL);
        let src = builder.mov(&0i32);
        let src_var = builder.get_inst_variable(src);
        builder.close_dim(&dim1);
        let last_var = builder.create_last_variable(src_var, &[&dim1]);
        let dim2 = builder.open_mapped_dim(&dim0);
        builder.action(Action::DimKind(dim2[0], DimKind::UNROLL));
        builder.order(&dim0, &dim2, Order::BEFORE);
        let mapped_var = builder.create_dim_map_variable(last_var, &[(&dim0, &dim2)]);
        builder.mov(&mapped_var);
        let space = builder.get();
        let wrappers = wrap_variables(&space);
        // `wrap_variables` orders variables according to data dependencies.
        assert_eq!(wrappers[0].variable.id(), src_var);
        assert_eq!(wrappers[1].variable.id(), last_var);
        assert_eq!(wrappers[2].variable.id(), mapped_var);
        // Check `src_var`.
        assert_eq!(wrappers[0].instantiation_dims, mk_map(&[(dim0[0], 4)]));
        assert!(wrappers[0].alias.is_none());
        // Check `last_var`
        assert_eq!(wrappers[1].instantiation_dims, mk_map(&[(dim0[0], 4)]));
        if let Some(ref alias) = wrappers[1].alias {
            assert_eq!(alias.other_variable, src_var);
            assert_eq!(alias.dim_mapping, mk_map(&[(dim1[0], None)]));
        } else {
            panic!("expected an alias for last_var");
        }
        // Check `mapped_var`
        assert_eq!(wrappers[2].instantiation_dims, mk_map(&[(dim2[0], 4)]));
        if let Some(ref alias) = wrappers[2].alias {
            assert_eq!(alias.other_variable, last_var);
            assert_eq!(alias.dim_mapping, mk_map(&[(dim0[0], Some(dim2[0]))]));
        } else {
            panic!("expected an alias for mapped_var");
        }
    }
}
