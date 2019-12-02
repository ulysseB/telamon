//! Helper struct to build a `Function`.
use crate::device::Device;
use crate::helper::{AutoOperand, LogicalDim, MetaStatement, TilingPattern};
use crate::ir::{self, op, Parameter, Type};
use crate::ir::{AccessPattern, Function, InstId, Operand, Operator, Signature};
use crate::search_space::{Action, DimKind, InstFlag, MemSpace, Order, SearchSpace};
use fxhash::FxHashMap;
use itertools::Itertools;
use log::debug;
use std::borrow::Borrow;
use std::sync::Arc;
use utils::*;

/// Helper to build a `Function`.
pub struct Builder {
    function: Function<()>,
    open_dims: FxHashMap<ir::DimId, ir::DimId>,
    actions: Vec<Action>,
}

impl Builder {
    /// Creates a new `Builder` for a `Function` with the given signature.
    pub fn new(signature: Arc<Signature>, device: Arc<dyn Device>) -> Builder {
        Builder {
            function: Function::new(signature, device),
            open_dims: FxHashMap::default(),
            actions: Vec::new(),
        }
    }

    /// Returns the function created by the builder
    pub fn get(self) -> SearchSpace {
        debug!("{:?}", self.actions);
        SearchSpace::new(self.function, self.actions).expect("invalid IR instance")
    }

    /// Returns the function created by the builder
    pub fn get_clone(&self) -> SearchSpace {
        let function = self.function.clone();
        SearchSpace::new(function, self.actions.clone()).expect("invalid IR instance")
    }

    /// Returns an operand from an `AutoOperand`.
    fn get_op(&mut self, op: &dyn AutoOperand) -> Operand<()> {
        op.get(self)
    }

    /// Creates a binary operator.
    pub fn binop(
        &mut self,
        op: ir::BinOp,
        lhs: &dyn AutoOperand,
        rhs: &dyn AutoOperand,
    ) -> InstId {
        let lhs_op = self.get_op(lhs);
        let rhs_op = self.get_op(rhs);
        let rounding = default_rounding(lhs_op.t());
        self.inst(op::BinOp(op, lhs_op, rhs_op, rounding))
    }

    /// Adds an `Add` instruction to the fuction.
    pub fn add(&mut self, lhs: &dyn AutoOperand, rhs: &dyn AutoOperand) -> InstId {
        self.binop(ir::BinOp::Add, lhs, rhs)
    }

    /// Adds a `Sub` instruction to the function.
    pub fn sub(&mut self, lhs: &dyn AutoOperand, rhs: &dyn AutoOperand) -> InstId {
        self.binop(ir::BinOp::Sub, lhs, rhs)
    }

    /// Adds a `Mul` instruction to the function. Defaults to low mode.
    pub fn mul(&mut self, lhs: &dyn AutoOperand, rhs: &dyn AutoOperand) -> InstId {
        let lhs_op = self.get_op(lhs);
        let rhs_op = self.get_op(rhs);
        let t = lhs_op.t();
        let rounding = default_rounding(t);
        self.inst(op::Mul(lhs_op, rhs_op, rounding, t))
    }

    /// Adds a 'Mul` instruction with a wide mode to the function.
    pub fn mul_ex(
        &mut self,
        lhs: &dyn AutoOperand,
        rhs: &dyn AutoOperand,
        t: Type,
    ) -> InstId {
        let lhs_op = self.get_op(lhs);
        let rhs_op = self.get_op(rhs);
        let rounding = default_rounding(t);
        let op = op::Mul(lhs_op, rhs_op, rounding, t);
        self.inst(op)
    }

    /// Adds a `Mad` or `Fma` instruction to the function. Defaults to low or wide mode
    /// depending on the operand types.
    pub fn mad(
        &mut self,
        mul_lhs: &dyn AutoOperand,
        mul_rhs: &dyn AutoOperand,
        add_rhs: &dyn AutoOperand,
    ) -> InstId {
        let mul_lhs_op = self.get_op(mul_lhs);
        let mul_rhs_op = self.get_op(mul_rhs);
        let add_rhs_op = self.get_op(add_rhs);
        let rounding = default_rounding(mul_lhs_op.t());
        let op = op::Mad(mul_lhs_op, mul_rhs_op, add_rhs_op, rounding);
        self.inst(op)
    }

    /// Adds a `Max` instruction to the fuction.
    pub fn max(&mut self, lhs: &dyn AutoOperand, rhs: &dyn AutoOperand) -> InstId {
        let lhs_op = self.get_op(lhs);
        let rhs_op = self.get_op(rhs);
        let rounding = op::Rounding::Exact;
        let op = ir::BinOp::Max;
        self.inst(op::BinOp(op, lhs_op, rhs_op, rounding))
    }

    /// Adds a `Div` instruction to the fuction.
    pub fn div(&mut self, lhs: &dyn AutoOperand, rhs: &dyn AutoOperand) -> InstId {
        self.binop(ir::BinOp::Div, lhs, rhs)
    }

    /// Adds a `Mov` instruction to the function.
    pub fn mov(&mut self, arg: &dyn AutoOperand) -> InstId {
        let arg_op = self.get_op(arg);
        self.inst(op::UnaryOp(ir::UnaryOp::Mov, arg_op))
    }

    /// Adds an `Exp` instruction to the function.
    pub fn exp(&mut self, arg: &dyn AutoOperand) -> InstId {
        let arg_op = self.get_op(arg);
        let t = arg_op.t();
        self.inst(op::UnaryOp(ir::UnaryOp::Exp(t), arg_op))
    }

    /// Adds a coherent load from global memory instruction to the function.
    pub fn ld(
        &mut self,
        ret_type: Type,
        addr: &dyn AutoOperand,
        pattern: AccessPattern,
    ) -> InstId {
        self.ld_ex(ret_type, addr, pattern, InstFlag::COHERENT)
    }

    /// Adds a non-coherent load from global memory instruction to the function.
    pub fn ld_nc(
        &mut self,
        ret_type: Type,
        addr: &dyn AutoOperand,
        pattern: AccessPattern,
    ) -> InstId {
        self.ld_ex(ret_type, addr, pattern, InstFlag::ALL)
    }

    /// Adds a load instruction with the given flags and memory block.
    pub fn ld_ex(
        &mut self,
        ret_type: Type,
        addr: &dyn AutoOperand,
        pattern: AccessPattern,
        flags: InstFlag,
    ) -> InstId {
        let addr_op = self.get_op(addr);
        let inst_id = self.inst(op::Ld(ret_type, addr_op, pattern));
        self.actions.push(Action::InstFlag(inst_id, flags));
        inst_id
    }

    /// Adds a store instruction.
    pub fn st(
        &mut self,
        addr: &dyn AutoOperand,
        val: &dyn AutoOperand,
        pattern: AccessPattern,
    ) -> InstId {
        self.st_ex(addr, val, true, pattern, InstFlag::ALL)
    }

    /// Adds a store instruction with the given flags and memory block.
    pub fn st_ex(
        &mut self,
        addr: &dyn AutoOperand,
        val: &dyn AutoOperand,
        side_effect: bool,
        pattern: AccessPattern,
        flags: InstFlag,
    ) -> InstId {
        let addr_op = self.get_op(addr);
        let val_op = self.get_op(val);
        let inst_id = self.inst(op::St(addr_op, val_op, side_effect, pattern));
        self.actions.push(Action::InstFlag(inst_id, flags));
        inst_id
    }

    /// Adds a cast instruction to the given type.
    pub fn cast(&mut self, val: &dyn AutoOperand, t: Type) -> InstId {
        let val_op = self.get_op(val);
        self.inst(op::UnaryOp(ir::UnaryOp::Cast(t), val_op))
    }

    /// Restricts the order between two basic blocks. Does not restricts LINK and NPACK
    /// flags.
    pub fn order(
        &mut self,
        lhs: &dyn MetaStatement,
        rhs: &dyn MetaStatement,
        order: Order,
    ) {
        for lhs in lhs.borrow().ids() {
            for rhs in rhs.borrow().ids() {
                self.action(Action::Order(lhs, rhs, order));
            }
        }
    }

    /// Inserts an instruction in the function.
    fn inst(&mut self, op: Operator<()>) -> InstId {
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
        let dims = logical_dims.iter().cloned().flatten().collect();
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

    /// Applies an action on the function.
    pub fn action(&mut self, action: Action) {
        self.actions.push(action)
    }

    /// Opens a new dimension.
    pub fn open_dim(&mut self, size: ir::Size) -> LogicalDim {
        self.open_tiled_dim(size, TilingPattern::default())
    }

    /// Opens a nest of new dimension with the given kinds and sizes.
    pub fn open_dim_ex(&mut self, size: ir::Size, kind: DimKind) -> LogicalDim {
        let dim = self.open_dim(size);
        self.actions.push(Action::DimKind(dim[0], kind));
        dim
    }

    /// Open multiple dimensions to represent a tiled dimension.
    pub fn open_tiled_dim(
        &mut self,
        size: ir::Size,
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

    pub fn with_tiled_dims<I, F, T>(&mut self, tilings: I, body_fn: F) -> T
    where
        I: IntoIterator<Item = (ir::Size, TilingPattern)>,
        F: FnOnce(&[LogicalDim], &mut Self) -> T,
    {
        let dims: Vec<_> = tilings
            .into_iter()
            .map(|(size, tiling)| self.open_tiled_dim(size, tiling))
            .collect();

        let result = body_fn(&dims, self);

        for dim in &dims {
            self.close_dim(dim);
        }

        result
    }

    /// Returns a constant size.
    pub fn cst_size(&self, size: u32) -> ir::Size {
        ir::Size::new_const(size)
    }

    /// Returns a parameter size.
    pub fn param_size(&self, param: &str, max_size: u32) -> ir::Size {
        ir::Size::new_param(Arc::clone(self.find_param(param)), max_size)
    }

    /// Allocates a memory block in shared memory.
    pub fn allocate_shared(&mut self, size: u32) -> ir::MemId {
        let id = self.allocate(size, true);
        self.actions.push(Action::MemSpace(id, MemSpace::SHARED));
        id
    }

    /// Allocates a memory block.
    pub fn allocate(&mut self, size: u32, private: bool) -> ir::MemId {
        assert!(
            private,
            "allocating non-private memory is not yet supported"
        );
        self.function.add_mem_block(size)
    }

    /// Builds both an induction variable for a tensor memory access and the corresponding
    /// access pattern.
    pub fn tensor_access(
        &mut self,
        addr: &dyn AutoOperand,
        mem_id: Option<ir::MemId>,
        t: ir::Type,
        dims: &[&LogicalDim],
    ) -> (ir::IndVarId, ir::AccessPattern) {
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
        mem: Option<ir::MemId>,
        increments: Vec<(&LogicalDim, ir::Size)>,
    ) -> AccessPattern {
        let dims = self.logical_to_real_increments(increments);
        AccessPattern::Tensor {
            mem_id: mem,
            dims: dims.into_iter().collect(),
        }
    }

    /// Builds an induction variable.
    pub fn induction_var(
        &mut self,
        base: &dyn AutoOperand,
        dims: Vec<(&LogicalDim, ir::Size)>,
    ) -> ir::IndVarId {
        let base = self.get_op(base);
        let dims = self.logical_to_real_increments(dims);
        self.function
            .add_ind_var(unwrap!(ir::InductionVar::new(dims, base)))
    }

    /// Converts increments on logical dimensions to increment on real dimensions.
    fn logical_to_real_increments(
        &self,
        increments: Vec<(&LogicalDim, ir::Size)>,
    ) -> Vec<(ir::DimId, ir::PartialSize)> {
        increments
            .into_iter()
            .flat_map(|(dim, size)| {
                let mut size: ir::PartialSize = size.into();
                // `dimensions()` returns the dimensions from inner-most to outer-most, but we want
                // them from outer-most to inner-most for display.  Note that we don't otherwise
                // depend on the order of the increments; they are only used as a DimId ->
                // PartialSize mapping.
                self.function
                    .logical_dim(dim.id())
                    .dimensions()
                    .map(move |dim| {
                        let increment = size.clone();
                        size *= self.function.dim(dim).size();
                        (dim, increment)
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
            })
            .collect()
    }

    /// Returns the list of increment to access an n-dimensional tensor.
    fn tensor_increments<'b>(
        &self,
        t: ir::Type,
        dims: &[&'b LogicalDim],
    ) -> Vec<(&'b LogicalDim, ir::Size)> {
        let data_size = ir::Size::new_const(unwrap!(t.len_byte()));
        dims.iter()
            .rev()
            .scan(data_size, |size, &dim| {
                let increment = size.clone();
                *size *= self.function.logical_dim(dim.id()).total_size();
                Some((dim, increment))
            })
            .collect()
    }

    /// Creates a dim-map operand.
    pub fn dim_map(
        &self,
        base: ir::InstId,
        dim_map: &[(&LogicalDim, &LogicalDim)],
        scope: ir::DimMapScope<()>,
    ) -> ir::Operand<()> {
        let dim_map = dim_map
            .iter()
            .flat_map(|&(lhs, rhs)| lhs.iter().zip_eq(rhs.iter()));
        let inst = self.function.inst(base);
        ir::Operand::new_inst(inst, ir::DimMap::new(dim_map), scope)
    }

    /// Finds a paramter given its name.
    pub fn find_param(&self, param: &str) -> &Arc<Parameter> {
        unwrap!(self
            .function
            .signature()
            .params
            .iter()
            .find(|p| p.name == param))
    }

    pub fn new_access(
        &mut self,
        name: &str,
        strides: Vec<(ir::IndexExpr, ir::Size)>,
    ) -> ir::AccessId {
        let param = self.find_param(name).clone();
        self.function.accesses_mut().add(param, strides)
    }

    /// Returns a reference to the function being built.
    pub fn function(&self) -> &ir::Function<()> {
        &self.function
    }

    /// Returns the list of open dimensions with the dimensions they are mapped to.
    pub(super) fn open_dims(&self) -> impl Iterator<Item = (ir::DimId, ir::DimId)> + '_ {
        self.open_dims.iter().map(|(&new, &old)| (new, old))
    }
}

/// Returns the default rounding for a given operand type.
fn default_rounding(t: Type) -> op::Rounding {
    if t.is_integer() {
        op::Rounding::Exact
    } else {
        op::Rounding::Nearest
    }
}
