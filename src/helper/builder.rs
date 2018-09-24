//! Helper struct to build a `Function`.
use device::Device;
use helper::{AutoOperand, LogicalDim, MetaStatement};
use ir::{self, mem, op, Parameter, Type};
use ir::{
    AccessPattern, Function, InstId, Operand, Operator, Signature, ValueDef, ValueId,
};
use itertools::Itertools;
use search_space::{Action, DimKind, InstFlag, MemSpace, Order, SearchSpace};
use std::borrow::Borrow;
use utils::*;

/// Helper to build a `Function`.
pub struct Builder<'a> {
    function: Function<'a, ()>,
    open_dims: HashMap<ir::DimId, ir::DimId>,
    actions: Vec<Action>,
}

impl<'a> Builder<'a> {
    /// Creates a new `Builder` for a `Function` with the given signature.
    pub fn new(signature: &'a Signature, device: &'a Device) -> Builder<'a> {
        Builder {
            function: Function::new(signature, device),
            open_dims: HashMap::default(),
            actions: Vec::new(),
        }
    }

    /// Returns the function created by the builder
    pub fn get(self) -> SearchSpace<'a> {
        debug!("{:?}", self.actions);
        SearchSpace::new(self.function, self.actions).expect("invalid IR instance")
    }

    /// Returns the function created by the builder
    pub fn get_clone(&self) -> SearchSpace<'a> {
        let function = self.function.clone();
        SearchSpace::new(function, self.actions.clone()).expect("invalid IR instance")
    }

    /// Returns an operand from an `AutoOperand`.
    fn get_op<'b: 'a>(&mut self, op: &AutoOperand<'b>) -> Operand<'a, ()> {
        op.get(self)
    }

    /// Creates a binary operator.
    pub fn binop<'b: 'a>(
        &mut self,
        op: ir::BinOp,
        lhs: &AutoOperand<'b>,
        rhs: &AutoOperand<'b>,
    ) -> InstId {
        let lhs_op = self.get_op(lhs);
        let rhs_op = self.get_op(rhs);
        let rounding = default_rounding(&lhs_op.t());
        self.inst(op::BinOp(op, lhs_op, rhs_op, rounding))
    }

    /// Adds an `Add` instruction to the fuction.
    pub fn add<'b: 'a>(
        &mut self,
        lhs: &AutoOperand<'b>,
        rhs: &AutoOperand<'b>,
    ) -> InstId {
        self.binop(ir::BinOp::Add, lhs, rhs)
    }

    /// Adds a `Sub` instruction to the function.
    pub fn sub<'b: 'a>(
        &mut self,
        lhs: &AutoOperand<'b>,
        rhs: &AutoOperand<'b>,
    ) -> InstId {
        self.binop(ir::BinOp::Sub, lhs, rhs)
    }

    /// Adds a `Mul` instruction to the function. Defaults to low mode.
    pub fn mul<'b: 'a>(
        &mut self,
        lhs: &AutoOperand<'b>,
        rhs: &AutoOperand<'b>,
    ) -> InstId {
        let lhs_op = self.get_op(lhs);
        let rhs_op = self.get_op(rhs);
        let t = lhs_op.t();
        let rounding = default_rounding(&t);
        self.inst(op::Mul(lhs_op, rhs_op, rounding, t))
    }

    /// Adds a 'Mul` instruction with a wide mode to the function.
    pub fn mul_ex<'b: 'a>(
        &mut self,
        lhs: &AutoOperand<'b>,
        rhs: &AutoOperand<'b>,
        t: Type,
    ) -> InstId {
        let lhs_op = self.get_op(lhs);
        let rhs_op = self.get_op(rhs);
        let rounding = default_rounding(&t);
        let op = op::Mul(lhs_op, rhs_op, rounding, t);
        self.inst(op)
    }

    /// Adds a `Mad` or `Fma` instruction to the function. Defaults to low or wide mode
    /// depending on the operand types.
    pub fn mad<'b: 'a>(
        &mut self,
        mul_lhs: &AutoOperand<'b>,
        mul_rhs: &AutoOperand<'b>,
        add_rhs: &AutoOperand<'b>,
    ) -> InstId {
        let mul_lhs_op = self.get_op(mul_lhs);
        let mul_rhs_op = self.get_op(mul_rhs);
        let add_rhs_op = self.get_op(add_rhs);
        let rounding = default_rounding(&mul_lhs_op.t());
        let op = op::Mad(mul_lhs_op, mul_rhs_op, add_rhs_op, rounding);
        self.inst(op)
    }

    /// Adds a `Div` instruction to the fuction.
    pub fn div<'b: 'a>(
        &mut self,
        lhs: &AutoOperand<'b>,
        rhs: &AutoOperand<'b>,
    ) -> InstId {
        self.binop(ir::BinOp::Div, lhs, rhs)
    }

    /// Adds a `Mov` instruction to the function.
    pub fn mov<'b: 'a>(&mut self, arg: &AutoOperand<'b>) -> InstId {
        let arg_op = self.get_op(arg);
        self.inst(op::Mov(arg_op))
    }

    /// Adds a coherent load from global memory instruction to the function.
    pub fn ld<'b: 'a>(
        &mut self,
        ret_type: Type,
        addr: &AutoOperand<'b>,
        pattern: AccessPattern<'a>,
    ) -> InstId {
        self.ld_ex(ret_type, addr, pattern, InstFlag::MEM_COHERENT)
    }

    /// Adds a non-coherent load from global memory instruction to the function.
    pub fn ld_nc<'b: 'a>(
        &mut self,
        ret_type: Type,
        addr: &AutoOperand<'b>,
        pattern: AccessPattern<'a>,
    ) -> InstId {
        self.ld_ex(ret_type, addr, pattern, InstFlag::ALL)
    }

    /// Adds a load instruction with the given flags and memory block.
    pub fn ld_ex<'b: 'a>(
        &mut self,
        ret_type: Type,
        addr: &AutoOperand<'b>,
        pattern: AccessPattern<'a>,
        flags: InstFlag,
    ) -> InstId {
        let addr_op = self.get_op(addr);
        let inst_id = self.inst(op::Ld(ret_type, addr_op, pattern));
        self.actions.push(Action::InstFlag(inst_id, flags));
        inst_id
    }

    /// Adds a store instruction.
    pub fn st<'b: 'a>(
        &mut self,
        addr: &AutoOperand<'b>,
        val: &AutoOperand<'b>,
        pattern: AccessPattern<'a>,
    ) -> InstId {
        self.st_ex(addr, val, true, pattern, InstFlag::ALL)
    }

    /// Adds a store instruction with the given flags and memory block.
    pub fn st_ex<'b: 'a>(
        &mut self,
        addr: &AutoOperand<'b>,
        val: &AutoOperand<'b>,
        side_effect: bool,
        pattern: AccessPattern<'a>,
        flags: InstFlag,
    ) -> InstId {
        let addr_op = self.get_op(addr);
        let val_op = self.get_op(val);
        let inst_id = self.inst(op::St(addr_op, val_op, side_effect, pattern));
        self.actions.push(Action::InstFlag(inst_id, flags));
        inst_id
    }

    /// Adds a cast instruction to the given type.
    pub fn cast<'b: 'a>(&mut self, val: &AutoOperand<'b>, t: Type) -> InstId {
        let val_op = self.get_op(val);
        self.inst(op::Cast(val_op, t))
    }

    /// Restricts the order between two basic blocks. Does not restricts LINK and NPACK
    /// flags.
    pub fn order(&mut self, lhs: &MetaStatement, rhs: &MetaStatement, order: Order) {
        for lhs in lhs.borrow().ids() {
            for rhs in rhs.borrow().ids() {
                self.action(Action::Order(lhs, rhs, order));
            }
        }
    }

    /// Inserts an instruction in the function.
    fn inst(&mut self, op: Operator<'a, ()>) -> InstId {
        let open_dims = self.open_dims.iter().map(|(&x, _)| x).collect();
        unwrap!(self.function.add_inst(op, open_dims))
    }

    pub fn create_inst_value(&mut self, inst_id: InstId) -> ValueId {
        if let Some(val_id) = self.function.inst(inst_id).result_value() {
            val_id
        } else {
            let value_def = ValueDef::Inst(inst_id);
            let value_id = unwrap!(self.function.add_value(value_def));
            value_id
        }
    }

    /// Applies an action on the function.
    pub fn action(&mut self, action: Action) {
        self.actions.push(action)
    }

    /// Opens a new dimension.
    pub fn open_dim(&mut self, size: ir::Size<'a>) -> LogicalDim {
        self.open_tiled_dim(size, &[])
    }

    /// Opens a nest of new dimension with the given kinds and sizes.
    pub fn open_dim_ex(&mut self, size: ir::Size<'a>, kind: DimKind) -> LogicalDim {
        let dim = self.open_dim(size);
        self.actions.push(Action::DimKind(dim[0], kind));
        dim
    }

    /// Open multiple dimensions to represent a tiled dimension.
    pub fn open_tiled_dim(
        &mut self,
        size: ir::Size<'a>,
        // This is a reference to avoid breaking the interface. This parameter will be
        // removed when we allow specifying multiple tile sizes for each dimension so it
        // is no worth changing the code everywhere this function is used just yet.
        tile_sizes: &[u32],
    ) -> LogicalDim {
        // TODO(strip-mining): allow multiple tile size for each level.
        let tiling_factors = vec![tile_sizes.iter().product()];
        let (logical_id, real_ids) = unwrap!(self.function.add_logical_dim(
            size,
            tiling_factors,
            tile_sizes,
        ));
        self.open_dims.extend(real_ids.iter().map(|&id| (id, id)));
        LogicalDim {
            logical_id,
            real_ids,
            tile_sizes: tile_sizes.to_vec(),
        }
    }

    /// Opens a new dimension mapped to an existing one.
    ///
    /// The size of the new dim is inherited from the mapped dim.
    /// The dimension mapped to is closed if needed.
    pub fn open_mapped_dim(&mut self, old_dim: &LogicalDim) -> LogicalDim {
        let (size, tiling_factors) = {
            let old_dim = self.function.logical_dim(old_dim.id());
            (
                old_dim.total_size().clone(),
                old_dim.possible_tilings().to_vec(),
            )
        };
        let (new_id, new_dims) = unwrap!(self.function.add_logical_dim(
            size.clone(),
            tiling_factors.clone(),
            &old_dim.tile_sizes,
        ));
        for (old, &new) in old_dim.iter().zip_eq(&new_dims) {
            self.open_dims.remove(&old);
            self.open_dims.insert(new, old);
        }
        LogicalDim {
            logical_id: new_id,
            real_ids: new_dims,
            tile_sizes: old_dim.tile_sizes.clone(),
        }
    }

    /// Opens an existing dimension.
    pub fn reopen_dim(&mut self, dim: &LogicalDim) {
        for id in dim.iter() {
            self.open_dims.insert(id, id);
        }
    }

    /// Opens an existing dimension and maps it to another one.
    /// The dimension mapped to is closed if needed.
    pub fn reopen_mapped_dim(&mut self, dim: &LogicalDim, mapped_to: &LogicalDim) {
        for (dim, mapped_to) in dim.iter().zip_eq(mapped_to) {
            self.open_dims.remove(&mapped_to);
            self.open_dims.insert(dim, mapped_to);
        }
    }

    /// Closes a dimension.
    pub fn close_dim(&mut self, dims: &LogicalDim) {
        for dim in dims {
            assert!(self.open_dims.remove(&dim).is_some());
        }
    }

    /// Returns a constant size.
    pub fn cst_size(&self, size: u32) -> ir::Size<'a> {
        ir::Size::new_const(size)
    }

    /// Returns a parameter size.
    pub fn param_size(&self, param: &str, max_size: u32) -> ir::Size<'a> {
        ir::Size::new_param(self.find_param(param), max_size)
    }

    /// Allocates a memory block in shared memory.
    pub fn allocate_shared(&mut self, size: u32) -> mem::InternalId {
        let id = self.allocate(size, true);
        self.actions
            .push(Action::MemSpace(id.into(), MemSpace::SHARED));
        id
    }

    /// Allocates a memory block.
    pub fn allocate(&mut self, size: u32, private: bool) -> mem::InternalId {
        assert!(
            private,
            "allocating non-private memory is not yet supported"
        );
        self.function.add_mem_block(size)
    }

    /// Generates an access paterns with all the strides unknown on the opened dimensions.
    pub fn unknown_access_pattern(&self, mem: ir::MemId) -> AccessPattern<'static> {
        AccessPattern::Unknown { mem_id: mem }
    }

    /// Builds both an induction variable for a tensor memory access and the corresponding
    /// access pattern.
    pub fn tensor_access(
        &mut self,
        addr: &AutoOperand<'a>,
        mem_id: ir::MemId,
        t: ir::Type,
        dims: &[&LogicalDim],
    ) -> (ir::IndVarId, ir::AccessPattern<'a>) {
        let base = self.get_op(addr);
        let logical_increments = self.tensor_increments(t, dims);
        let increments = self.logical_to_real_increments(logical_increments);
        let dims = increments.iter().cloned().collect();
        let ind_var = unwrap!(ir::InductionVar::new(increments, base));
        let ind_var_id = self.function.add_ind_var(ind_var);
        (ind_var_id, AccessPattern::Tensor { mem_id, dims })
    }

    /// Generates the access pattern corresponding to accessing a tensor of the given
    /// type.
    pub fn tensor_access_pattern(
        &self,
        mem: ir::MemId,
        increments: Vec<(&LogicalDim, ir::Size<'a>)>,
    ) -> AccessPattern<'a> {
        let dims = self.logical_to_real_increments(increments);
        AccessPattern::Tensor {
            mem_id: mem,
            dims: dims.into_iter().collect(),
        }
    }

    /// Builds an induction variable.
    pub fn induction_var(
        &mut self,
        base: &AutoOperand<'a>,
        dims: Vec<(&LogicalDim, ir::Size<'a>)>,
    ) -> ir::IndVarId {
        let base = self.get_op(base);
        let dims = self.logical_to_real_increments(dims);
        self.function
            .add_ind_var(unwrap!(ir::InductionVar::new(dims, base)))
    }

    /// Converts increments on logical dimensions to increment on real dimensions.
    fn logical_to_real_increments(
        &self,
        increments: Vec<(&LogicalDim, ir::Size<'a>)>,
    ) -> Vec<(ir::DimId, ir::PartialSize<'a>)> {
        increments
            .into_iter()
            .flat_map(|(dim, size)| {
                let mut size: ir::PartialSize = size.into();
                self.function
                    .logical_dim(dim.id())
                    .dimensions()
                    .map(move |dim| {
                        let increment = size.clone();
                        size *= self.function.dim(dim).size();
                        (dim, increment)
                    })
            }).collect()
    }

    /// Returns the list of increment to access an n-dimensional tensor.
    fn tensor_increments<'b>(
        &self,
        t: ir::Type,
        dims: &[&'b LogicalDim],
    ) -> Vec<(&'b LogicalDim, ir::Size<'a>)> {
        let data_size = ir::Size::new_const(unwrap!(t.len_byte()));
        dims.into_iter()
            .rev()
            .scan(data_size, |size, &dim| {
                let increment = size.clone();
                *size *= self.function.logical_dim(dim.id()).total_size();
                Some((dim, increment))
            }).collect()
    }

    /// Creates a dim-map operand.
    pub fn dim_map(
        &self,
        base: ir::InstId,
        dim_map: &[(&LogicalDim, &LogicalDim)],
        scope: ir::DimMapScope<()>,
    ) -> ir::Operand<'a, ()> {
        let dim_map = dim_map
            .iter()
            .flat_map(|&(lhs, rhs)| lhs.iter().zip_eq(rhs.iter()));
        let inst = self.function.inst(base);
        ir::Operand::new_inst(inst, ir::DimMap::new(dim_map), scope)
    }

    /// Finds a paramter given its name.
    pub fn find_param(&self, param: &str) -> &'a Parameter {
        unwrap!(
            self.function
                .signature()
                .params
                .iter()
                .find(|p| p.name == param)
        )
    }

    /// Returns a reference to the function being built.
    pub(super) fn function(&self) -> &ir::Function<'a, ()> {
        &self.function
    }

    /// Returns the list of open dimensions with the dimensions they are mapped to.
    pub(super) fn open_dims(&self) -> impl Iterator<Item = (ir::DimId, ir::DimId)> + '_ {
        self.open_dims.iter().map(|(&new, &old)| (new, old))
    }
}

/// Returns the default rounding for a given operand type.
fn default_rounding(t: &Type) -> op::Rounding {
    if t.is_integer() {
        op::Rounding::Exact
    } else {
        op::Rounding::Nearest
    }
}
