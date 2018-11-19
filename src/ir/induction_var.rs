use ir;
use utils::*;

/// Unique identifier for `InductionVar`
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct IndVarId(pub u32);

/// A multidimentional induction variable. No dimension should appear twice in dims.
#[derive(Clone, Debug)]
pub struct InductionVar<'a> {
    dims: Vec<(ir::DimId, ir::MemAccessStride<'a>)>,
    base: ir::Operand<'a>,
}

impl<'a> InductionVar<'a> {
    /// Creates a new induction var. Size represents the increment over each diemnsion
    /// taken independenly.
    pub fn new<L>(
        dims: Vec<(ir::DimId, ir::MemAccessStride<'a>)>,
        base: ir::Operand<'a>,
        fun: &ir::Function<L>,
    ) -> Result<Self, ir::Error> {
        ir::TypeError::check_integer(base.t())?;
        // Assert dimensions are unique.
        let mut dim_ids = HashSet::default();
        for &(id, _) in &dims {
            if !dim_ids.insert(id) {
                return Err(ir::Error::DuplicateIncrement { dim: id });
            }
        }
        // TODO(cleanup): return errors instead of panicing
        if let ir::Operand::Variable(var, ..) = base {
            match fun.variable(var).use_mode() {
                ir::VarUseMode::FromRegisters => (),
                // TODO(search_space): allow in-memory induction variables
                _ => unimplemented!("in-memory operands for induction variables"),
            }
        }
        Ok(InductionVar { dims, base })
    }

    /// Returns the base operand of the induction variable.
    pub fn base(&self) -> &ir::Operand<'a> {
        &self.base
    }

    /// Returns the list of induction dimensions along with the corresponding increments.
    pub fn dims(&self) -> &[(ir::DimId, ir::MemAccessStride<'a>)] {
        &self.dims
    }
}
