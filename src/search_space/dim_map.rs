//! DimMap and layout lowering.
use ir::{self, BasicBlock};
use search_space::{Action, DimKind, Domain, DomainStore, InstFlag, Order};
use search_space::operand;
use search_space::choices::{dim_kind, order};
use itertools::Itertools;

/// Lowers a layout
pub fn lower_layout(fun: &mut ir::Function, mem: ir::mem::InternalId,
                    st_dims: Vec<ir::dim::Id>, ld_dims: Vec<ir::dim::Id>,
                    domain: &DomainStore) -> Result<Vec<Action>, ()> {
    debug!("lower_layout({:?}) triggered", mem);
    fun.lower_layout(mem, st_dims, ld_dims);
    let mut actions = Vec::new();
    for &inst_id in fun.mem_block(mem.into()).uses() {
        let inst = fun.inst(inst_id);
        actions.extend(operand::inst_invariants(fun, inst));
        // TODO(automate): vectorization disabled -> express as an additional constraint
        for d in fun.dims() {
            if !inst.is_vectorizable(d) {
                let order = domain.get_order(inst.bb_id(), d.bb_id());
                if domain.get_dim_kind(d.id()) == DimKind::VECTOR {
                    actions.extend(order::restrict_delayed(
                            inst.bb_id(), d.bb_id(), fun, domain, !Order::INNER)?);
                } else if order.is(Order::INNER).is_true() {
                    actions.extend(dim_kind::restrict_delayed(
                            d.id(), fun, domain, !DimKind::VECTOR)?);
                }
            }
        };
    }
    Ok(actions)
}

/// Lowers a `DimMap`.
fn lower_dim_map(fun: &mut ir::Function, inst: ir::InstId, operand: usize,
                 new_objs: &mut ir::NewObjs) -> Result<Vec<Action>, ()> {
    debug!("lower_dim_map({:?}, {}) triggered", inst, operand);
    let (mem, st_dims, st, ld_dims, ld) = fun.lower_dim_map(inst, operand)?;
    let mut actions = Vec::new();
    // Order the store and load loop nests.
    for (&src, &dst) in st_dims.iter().zip_eq(&ld_dims) {
        actions.push(Action::Order(src.into(), dst.into(), Order::BEFORE | Order::MERGED));
    }
    // FIXME: allow global memory
    actions.push(Action::InstFlag(st, InstFlag::MEM_SHARED));
    actions.push(Action::InstFlag(ld, InstFlag::MEM_SHARED));
    //actions.push(Action::InstFlag(st, InstFlag::MEM_COHERENT));
    //actions.push(Action::InstFlag(ld, InstFlag::MEM_COHERENT));
    actions.push(Action::Order(st.into(), ld.into(), Order::BEFORE));
    let operand = fun.inst(inst).operands()[operand];
    actions.extend(operand::invariants(fun, operand, inst.into()));
    // Update the list of new objets
    for dim in st_dims.into_iter().chain(ld_dims) {
        new_objs.add_dimension(fun.dim(dim));
    }
    new_objs.add_mem_instruction(fun.inst(st));
    new_objs.add_mem_instruction(fun.inst(ld));
    new_objs.add_mem_block(mem);
    debug!("lower_dim_map actions: {:?}", actions);
    Ok(actions)
}

/// Trigger to call when two dimensions are not mapped.
pub fn dim_not_mapped(lhs: ir::dim::Id, rhs: ir::dim::Id, fun: &mut ir::Function)
    -> Result<(ir::NewObjs, Vec<Action>), ()>
{
    debug!("dim_not_mapped({:?}, {:?}) triggered", lhs, rhs);
    let to_lower = fun.insts().flat_map(|inst| {
        inst.dim_maps_to_lower(lhs, rhs).into_iter().map(move |op_id| (inst.id(), op_id))
    }).collect_vec();
    let mut new_objs = ir::NewObjs::default();
    let mut actions = Vec::new();
    for (inst, operand) in to_lower {
        actions.extend(lower_dim_map(fun, inst, operand, &mut new_objs)?);
    }
    Ok((new_objs, actions))
}

/// Trigger to call when two dimensions are not merged.
pub fn dim_not_merged(lhs: ir::dim::Id, rhs: ir::dim::Id, fun: &mut ir::Function)
    -> Result<(ir::NewObjs, Vec<Action>), ()>
{
    debug!("dim_not_merged({:?}, {:?}) triggered", lhs, rhs);
    fun.dim_not_merged(lhs, rhs);
    // TODO(cc_perf): avoid creating a 'NewObjs' object.
    Ok(Default::default())
}
