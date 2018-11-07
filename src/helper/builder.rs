//! Helper struct to build a `Function`.
use device::Device;
use helper::{AutoOperand, LogicalDim, MetaStatement, TilingPattern, ToVariable};
use ir::{self, op, Parameter, Type};
use ir::{AccessPattern, Function, InstId, Operand, Operator, Signature};
use itertools::{flatten, Itertools};
use search_space::{Action, DimKind, InstFlag, MemSpace, Order, SearchSpace};
use std::borrow::Borrow;
use utils::*;

/// Helper to build a `Function`.
#[derive(Clone)]
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
        self.inst(op::UnaryOp(ir::UnaryOp::Mov, arg_op))
    }

    /// Adds a coherent load from global memory instruction to the function.
    pub fn ld<'b: 'a>(
        &mut self,
        ret_type: Type,
        addr: &AutoOperand<'b>,
        pattern: AccessPattern<'a>,
    ) -> InstId {
        self.ld_ex(ret_type, addr, pattern, InstFlag::COHERENT)
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
        self.inst(op::UnaryOp(ir::UnaryOp::Cast(t), val_op))
    }

    /// Create an instruction that initiates a DMA access.
    pub fn dma_start<'b: 'a>(
        &mut self,
        src_ptr: &AutoOperand<'b>,
        src_pattern: ir::AccessPattern<'a>,
        dst_ptr: &AutoOperand<'b>,
        has_visible_side_effects: bool,
    ) -> InstId {
        let src_ptr = self.get_op(src_ptr);
        let dst_ptr = self.get_op(dst_ptr);
        self.inst(op::DmaStart {
            src_ptr,
            src_pattern,
            dst_ptr,
            has_side_effects: has_visible_side_effects,
            dma_wait: None,
        })
    }

    /// Creates an instruction that waits for a DMA access to finish.
    pub fn dma_wait<'b: 'a>(
        &mut self,
        dma_start: ir::InstId,
        dst_pattern: ir::AccessPattern<'a>,
    ) -> InstId {
        let sync_flag = dma_start.to_operand(self);
        let has_side_effects = {
            let start_op = self.function.inst(dma_start).operator();
            if let ir::op::DmaStart { has_side_effects, .. } = start_op {
                *has_side_effects
            } else {
                panic!("expected a DmaStart operator")
            }
        };
        self.inst(op::DmaWait {
            sync_flag,
            dst_pattern,
            has_side_effects,
            dma_start,
        })
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

    /// Returns the variable holding the result of an instruction. Creates it if
    /// necessary.
    pub fn get_inst_variable(&mut self, inst_id: InstId) -> ir::VarId {
        self.function
            .inst(inst_id)
            .result_variable()
            .unwrap_or_else(|| {
                unwrap!(self.function.add_variable(ir::VarDef::Inst(inst_id)))
            })
    }

    /// Creates a new variable that takes the last value of another variable produced in
    /// a loop nest.
    pub fn create_last_variable(
        &mut self,
        var: ir::VarId,
        logical_dims: &[&LogicalDim],
    ) -> ir::VarId {
        let dims = flatten(logical_dims.iter().cloned()).collect();
        self.function
            .add_variable(ir::VarDef::Last(var, dims))
            .unwrap()
    }

    /// Creates a new variable that takes point-to-point the value of another variable, in
    /// another loop nest.
    pub fn create_dim_map_variable(
        &mut self,
        var: ir::VarId,
        logical_mapping: &[(&LogicalDim, &LogicalDim)],
    ) -> ir::VarId {
        let mapping = logical_mapping
            .iter()
            .flat_map(|&(lhs, rhs)| lhs.iter().zip_eq(rhs))
            .map(|(lhs, rhs)| self.function.map_dimensions([lhs, rhs]))
            .collect();
        self.function
            .add_variable(ir::VarDef::DimMap(var, mapping))
            .unwrap()
    }

    /// Creates point-to-point communication to import the variable in the current loop
    /// nest following mapped dimensions. Returns the original variable if no dimension
    /// needs to be mapped.
    pub fn map_variable(&mut self, var: ir::VarId) -> ir::VarId {
        let mapped_dims = {
            let original_var = self.function.variable(var);
            self.open_dims
                .iter()
                .map(|(&lhs, &rhs)| (lhs, rhs))
                .filter(|(new_dim, old_dim)| new_dim != old_dim)
                .filter(|(_, old_dim)| original_var.dimensions().contains(&old_dim))
                .collect_vec()
        };
        if mapped_dims.is_empty() {
            return var;
        }
        let mapping = mapped_dims
            .into_iter()
            .map(|(new_dim, old_dim)| self.function.map_dimensions([old_dim, new_dim]))
            .collect();
        self.function
            .add_variable(ir::VarDef::DimMap(var, mapping))
            .unwrap()
    }

    /// Creates a variable initialized with the value of `init` and then takes a value
    /// produced at the last iteration of `dims`. The loop-carried variable must be with
    /// `set_loop_carried_variable`.
    pub fn create_fby_variable<V: ToVariable>(
        &mut self,
        init: V,
        dims: &[&LogicalDim],
    ) -> ir::VarId {
        let init = init.to_variable(self);
        let dims = dims.iter().flat_map(|dim| dim.iter()).collect();
        unwrap!(self.function.add_variable(ir::VarDef::Fby {
            init,
            prev: None,
            dims
        }))
    }

    /// Set the loop-carried dependency `loop_carried` of a variable `fby` created with
    /// `create_fby_variable`.
    pub fn set_loop_carried_variable<V: ToVariable>(
        &mut self,
        fby: ir::VarId,
        loop_carried: V,
    ) {
        let loop_carried = loop_carried.to_variable(self);
        unwrap!(self.function.set_loop_carried_variable(fby, loop_carried))
    }

    /// Applies an action on the function.
    pub fn action(&mut self, action: Action) {
        self.actions.push(action)
    }

    /// Opens a new dimension.
    pub fn open_dim(&mut self, size: ir::Size<'a>) -> LogicalDim {
        self.open_tiled_dim(size, TilingPattern::default())
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
        tiling_pattern: TilingPattern,
    ) -> LogicalDim {
        let (logical_id, real_ids) = unwrap!(self.function.add_logical_dim(
            size,
            tiling_pattern.tiling_factors.clone(),
            tiling_pattern.tile_sizes.clone(),
        ));
        self.open_dims.extend(real_ids.iter().map(|&id| (id, id)));
        LogicalDim {
            logical_id,
            real_ids,
            tiling_pattern,
        }
    }

    /// Opens a new dimension mapped to an existing one.
    ///
    /// The size of the new dim is inherited from the mapped dim.
    /// The dimension mapped to is closed if needed.
    pub fn open_mapped_dim(&mut self, old_dim: &LogicalDim) -> LogicalDim {
        let size = self.function.logical_dim(old_dim.id()).total_size().clone();
        let (new_id, new_dims) = unwrap!(self.function.add_logical_dim(
            size.clone(),
            old_dim.tiling_pattern.tiling_factors.clone(),
            old_dim.tiling_pattern.tile_sizes.clone(),
        ));
        for (old, &new) in old_dim.iter().zip_eq(&new_dims) {
            self.open_dims.remove(&old);
            self.open_dims.insert(new, old);
        }
        LogicalDim {
            logical_id: new_id,
            real_ids: new_dims,
            tiling_pattern: old_dim.tiling_pattern.clone(),
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
    pub fn allocate_shared(&mut self, t: ir::Type, len: u32) -> ir::MemId {
        let id = self.allocate(t, len, true);
        self.actions
            .push(Action::MemSpace(id.into(), MemSpace::SHARED));
        id
    }

    /// Allocates a memory block.
    pub fn allocate(&mut self, t: ir::Type, len: u32, private: bool) -> ir::MemId {
        assert!(
            private,
            "allocating non-private memory is not yet supported"
        );
        self.function.add_mem_block(t, len)
    }

    /// Builds both an induction variable for a tensor memory access and the corresponding
    /// access pattern.
    pub fn tensor_access(
        &mut self,
        addr: &AutoOperand<'a>,
        mem_id: Option<ir::MemId>,
        t: ir::Type,
        dims: &[&LogicalDim],
    ) -> (ir::IndVarId, ir::AccessPattern<'a>) {
        let base = self.get_op(addr);
        let logical_increments = self.tensor_increments(t, dims);
        let increments = self.logical_to_real_increments(logical_increments);
        let dims = increments.iter().cloned().collect();
        let ind_var = unwrap!(ir::InductionVar::new(increments, base));
        let ind_var_id = self.function.add_ind_var(ind_var);
        (ind_var_id, AccessPattern::Tensor { t, mem_id, dims })
    }

    /// Generates the access pattern corresponding to accessing a tensor of the given
    /// type. Increments must be given by increasing order.
    pub fn tensor_access_pattern(
        &self,
        mem: Option<ir::MemId>,
        t: ir::Type,
        increments: Vec<(&LogicalDim, ir::Size<'a>)>,
    ) -> AccessPattern<'a> {
        let dims = self.logical_to_real_increments(increments);
        AccessPattern::Tensor {
            t,
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
    pub fn tensor_increments<'b>(
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
