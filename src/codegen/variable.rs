//! Code generation for `ir::Variable`.
use codegen;
use indexmap::IndexMap;
use ir;
use itertools::Itertools;
use search_space::*;
use std;
use utils::*;

/// Wraps an `ir::Variable` to expose specified decisions.
pub struct Variable<'a> {
    variable: &'a ir::Variable,
    t: ir::Type,
    instantiation_dims: HashMap<ir::DimId, usize>,
    stride_by_rank: std::rc::Rc<HashMap<u32, codegen::Size<'a>>>,
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

    /// Indicates the strides along each layout dimension, indexed per rank.
    pub fn rank_stride(&self, rank: u32) -> &codegen::Size<'a> {
        &self.stride_by_rank[&rank]
    }
}

/// Indicates how a variable aliases with another.
#[derive(Debug)]
pub struct Alias {
    other_variable: ir::VarId,
    dim_mapping: HashMap<ir::DimId, Option<ir::DimId>>,
    reverse_mapping: HashMap<ir::DimId, ir::DimId>,
}

impl Alias {
    /// Creates a new `Alias` to the given variable, with the given dimension mapping.
    pub fn new<IT: IntoIterator<Item = (ir::DimId, Option<ir::DimId>)>>(
        var: ir::VarId,
        dim_mapping: IT,
        space: &SearchSpace,
    ) -> Self {
        let mem_space = space.domain().get_memory_space(var);
        let dim_mapping = if mem_space.intersects(MemorySpace::MEMORY) {
            // In-memory variable have a single name: their pointer.
            HashMap::default()
        } else {
            let is_vector_reg = mem_space.intersects(MemorySpace::VECTOR_REGISTER);
            // Build a hashmap from `other_variable` dimensions to the current variable
            // dimensions. Remore vector dimensions if the variable is stored in vector
            // registers.
            dim_mapping
                .into_iter()
                .filter(|&(_, rhs)| {
                    rhs.map_or(true, |rhs| {
                        let dim_kind = space.domain().get_dim_kind(rhs);
                        !(is_vector_reg && dim_kind.intersects(DimKind::VECTOR))
                    })
                }).collect()
        };
        let reverse_mapping = dim_mapping
            .iter()
            .flat_map(|(&lhs, &rhs)| rhs.map(|rhs| (rhs, lhs)))
            .collect();
        Alias {
            other_variable: var,
            dim_mapping,
            reverse_mapping,
        }
    }

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
    fn new_last(var: ir::VarId, dims: &[ir::DimId], space: &SearchSpace) -> Self {
        Alias::new(var, dims.iter().map(|&dim| (dim, None)), space)
    }

    /// Creates a new alias that takes point-to-point the values of another variable.
    fn new_dim_map<IT>(var: ir::VarId, dim_mapping: IT, space: &SearchSpace) -> Self
    where
        IT: IntoIterator<Item = (ir::DimId, ir::DimId)>,
    {
        let dim_mapping = dim_mapping.into_iter().map(|(lhs, rhs)| (lhs, Some(rhs)));
        Alias::new(var, dim_mapping, space)
    }

    /// Finds dimensions on which the variable must be instantiated to implement the
    /// aliasing. Also returns their size.
    fn find_instantiation_dims(&self, space: &SearchSpace) -> HashMap<ir::DimId, usize> {
        self.dim_mapping
            .iter()
            .flat_map(|(&lhs, &rhs)| rhs.map(|rhs| (lhs, rhs)))
            .filter(|&(lhs, rhs)| {
                space.domain().get_order(lhs.into(), rhs.into()) != Order::MERGED
            }).map(|(_, rhs)| {
                let size = space.ir_instance().dim(rhs).size();
                let int_size = unwrap!(codegen::Size::from_ir(size, space).as_int());
                (rhs, int_size as usize)
            }).collect()
    }
}

/// Generates variables aliases.
fn generate_aliases(space: &SearchSpace) -> HashMap<ir::VarId, Option<Alias>> {
    let mut aliases: HashMap<_, _> = space
        .ir_instance()
        .variables()
        .map(|var| (var.id(), None))
        .collect();
    for var in space.ir_instance().variables() {
        let def_mode = space.domain().get_var_def_mode(var.id());
        let alias = match var.def() {
            ir::VarDef::Inst(..) => continue,
            _ if def_mode == VarDefMode::COPY => continue,
            ir::VarDef::Last(alias, dims) => Alias::new_last(*alias, dims, space),
            ir::VarDef::DimMap(alias, mappings) => {
                let mappings = mappings.iter().map(|(&lhs, &rhs)| (lhs, rhs));
                Alias::new_dim_map(*alias, mappings, space)
            }
            ir::VarDef::Fby { init, prev, .. } => {
                // FIXME: only alias with the copy instead of the instruction
                // Alias with the intruction that produce prev. We known it cannot have
                // another alias because loop-carried variables can only be used in a
                // single `fby`.
                let (prod_var, mapping) = find_actual_def(unwrap!(*prev), space);
                let mapping = mapping.iter().map(|(&lhs, &rhs)| (rhs, lhs));
                let alias = Some(Alias::new_dim_map(var.id(), mapping, space));
                let alias_entry = unwrap!(aliases.get_mut(&prod_var));
                assert!(std::mem::replace(alias_entry, alias).is_none());
                // Also alias with init.
                Alias::new_last(*init, &[], space)
            }
        };
        // Set the alias and check no alias was already set.
        let alias_entry = unwrap!(aliases.get_mut(&var.id()));
        assert!(std::mem::replace(alias_entry, Some(alias)).is_none());
    }
    aliases
}

/// Find the first variable that is not defined in-place from another in the definition
/// chain of `var`. Returns the mapping of dimensions from the actual definition to `var`.
fn find_actual_def(
    mut var: ir::VarId,
    space: &SearchSpace,
) -> (ir::VarId, HashMap<ir::DimId, ir::DimId>) {
    let mut dim_mapping = HashMap::default();
    // `var` take the value produced at the last iteration of `last_dims`.
    let mut last_dims = HashSet::default();
    loop {
        let def_mode = space.domain().get_var_def_mode(var);
        if def_mode == VarDefMode::COPY {
            return (var, dim_mapping);
        }
        match space.ir_instance().variable(var).def() {
            ir::VarDef::Inst(..) => return (var, dim_mapping),
            ir::VarDef::Fby { .. } => panic!("cannot chain fby operators"),
            ir::VarDef::Last(pred, dims) => {
                var = *pred;
                last_dims.extend(dims.iter().cloned());
            }
            ir::VarDef::DimMap(pred, local_mapping) => {
                var = *pred;
                for (&src, &dst) in local_mapping {
                    if last_dims.remove(&dst) {
                        last_dims.insert(src);
                    } else {
                        let dst = dim_mapping.remove(&dst).unwrap_or(dst);
                        dim_mapping.insert(src, dst);
                    }
                }
            }
        }
    }
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
pub fn wrap_variables<'a>(space: &'a SearchSpace) -> IndexMap<ir::VarId, Variable<'a>> {
    let aliases = generate_aliases(space);
    debug!("aliases: {:?}", aliases);
    // Generate the wrappers and record the dimensions along which variables are
    // instantiated. We use an `IndexMap` so that the insertion order is preserved.
    let mut wrapped_vars = IndexMap::new();
    for (variable, alias_opt) in sort_variables(aliases, space) {
        let instantiation_dims = alias_opt
            .as_ref()
            .map(|alias| alias.find_instantiation_dims(space))
            .unwrap_or_default();
        debug!(
            "instantiating {:?} on {:?} with alias {:?}",
            variable, instantiation_dims, alias_opt
        );
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
        // Compute the stride along instantiated dimensions if the variable is stored
        // in memory. Index strides by the rank of dimensions.
        let stride_per_rank = alias_opt.as_ref().map_or_else(
            || std::rc::Rc::new(ranks_strides(variable, space)),
            |alias| wrapped_vars[&alias.other_variable].rank_strides.clone(),
        );
        let wrapper = Variable {
            variable,
            t: unwrap!(space.ir_instance().device().lower_type(variable.t(), space)),
            instantiation_dims,
            stride_per_rank,
            alias: alias_opt,
        };
        wrapped_vars.insert(variable.id(), wrapper);
    }
    wrapped_vars
}

/// Compute the stride along `var` layout dimensions. Returns the strides indexed by rank
/// of the layout dimension.
fn ranks_strides<'a>(
    var: &ir::Variable,
    space: &SearchSpace<'a>,
) -> HashMap<u32, codegen::Size<'a>> {
    if !var.is_memory() {
        return Default::default();
    }
    let mut stride = codegen::Size::default();
    let mut stride_by_rank: HashMap<_, _> = var
        .layout()
        .iter()
        // Recover the layout dimension and its rank.
        .map(|&layout_dim| {
            let layout_dim = space.ir_instance().layout_dimension(layout_dim);
            let rank_universe = unwrap!(layout_dim.possible_ranks());
            let rank_domain = space.domain().get_rank(layout_dim.id());
            (
                layout_dim,
                unwrap!(
                    rank_domain.as_constrained(rank_universe),
                    "unconstrained rank {:?} for {:?}",
                    rank_domain,
                    layout_dim.id()
                ),
            )
        }).filter(|&(_, rank)| rank > 0)
        // Sort by rank.
        .sorted_by(|(_, lhs), (_, rhs)| lhs.cmp(rhs))
        .into_iter()
        // Compute strides from the innermost.
        .map(|(layout_dim, rank)| {
            let current_stride = stride.clone();
            let dim_size = space.ir_instance().dim(layout_dim.dim()).size();
            stride *= &codegen::Size::from_ir(dim_size, space);
            (rank, current_stride)
        }).collect();
    stride_by_rank.insert(0, codegen::Size::new(0, vec![], 1));
    stride_by_rank
}

/// Returns the stride along a layout dimension that accesses a variable.
pub fn layout_dim_stride<'a, 'b>(
    id: ir::LayoutDimId,
    variables: &'b IndexMap<ir::VarId, codegen::Variable<'a>>,
    space: &'b SearchSpace<'a>,
) -> &'b codegen::Size<'a> {
    let layout_dim = space.ir_instance().layout_dimension(id);
    let accessed_var = unwrap!(layout_dim.accessed_variable());
    let rank_universe = unwrap!(layout_dim.possible_ranks());
    let rank_domain = space.domain().get_rank(id);
    let rank = unwrap!(rank_domain.as_constrained(rank_universe));
    if rank == 0 {
        &codegen::size::ZERO
    } else {
        &variables[&accessed_var].rank_stride(rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use device::x86::Cpu;
    use helper;
    use ir;
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
        // for i in 0..4:
        //   for j in 0..8:
        //      src[i] = 0;
        // for i in 0..4:
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
        let def_mode = ir::VarDefMode::InPlace { allow_sync: false };
        let use_mode = ir::VarUseMode::FromRegisters;
        let mapped_var = builder.create_dim_map_variable(
            last_var,
            &[(&dim0, &dim2)],
            def_mode,
            use_mode,
        );
        builder.mov(&mapped_var);
        let space = builder.get();
        let wrappers = wrap_variables(&space);
        // `wrap_variables` orders variables according to data dependencies.
        let sorted_vars = wrappers.keys().cloned().collect_vec();
        assert_eq!(sorted_vars, [src_var, last_var, mapped_var]);
        // Check `src_var`.
        assert_eq!(
            wrappers[&src_var].instantiation_dims,
            mk_map(&[(dim0[0], 4)])
        );
        assert!(wrappers[&src_var].alias.is_none());
        // Check `last_var`
        assert_eq!(
            wrappers[&last_var].instantiation_dims,
            mk_map(&[(dim0[0], 4)])
        );
        if let Some(ref alias) = wrappers[&last_var].alias {
            assert_eq!(alias.other_variable, src_var);
            assert_eq!(alias.dim_mapping, mk_map(&[(dim1[0], None)]));
        } else {
            panic!("expected an alias for last_var");
        }
        // Check `mapped_var`
        assert_eq!(
            wrappers[&mapped_var].instantiation_dims,
            mk_map(&[(dim2[0], 4)])
        );
        if let Some(ref alias) = wrappers[&mapped_var].alias {
            assert_eq!(alias.other_variable, last_var);
            assert_eq!(alias.dim_mapping, mk_map(&[(dim0[0], Some(dim2[0]))]));
        } else {
            panic!("expected an alias for mapped_var");
        }
    }
}
