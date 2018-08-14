use ir;
use utils::*;

/// Unique identifier for `InductionVar`
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct IndVarId(pub u32);

/// A multidimentional induction variable. No dimension should appear twice in dims.
#[derive(Clone, Debug)]
pub struct InductionVar<'a, L = ir::LoweringMap> {
    dims: Vec<(ir::DimId, ir::Size<'a>)>,
    base: ir::Operand<'a, L>,
}

impl<'a, L> InductionVar<'a, L> {
    /// Creates a new induction var. Size represents the increment over each diemnsion
    /// taken independenly.
    pub fn new(dims: Vec<(ir::DimId, ir::Size<'a>)>, base: ir::Operand<'a, L>)
        -> Result<Self, ir::Error>
    {
        ir::TypeError::check_integer(base.t())?;
        // Assert dimensions are unique.
        let mut dim_ids = HashSet::default();
        for &(id, _) in &dims {
            if !dim_ids.insert(id) {
                return Err(ir::Error::DuplicateIncrement { dim: id });
            }
        }
        // TODO(cleanup): return errors instead of panicing
        match base {
            ir::Operand::Reduce(..) =>
                panic!("induction variables cannot perform reductions"),
            ir::Operand::Inst(.., ir::DimMapScope::Global(..)) =>
                // TODO(search_space): allow dim map lowering for induction variables
                unimplemented!("dim map lowering for induction vars is not implemented yet"),
            _ => (),
        }
        Ok(InductionVar { dims, base })
    }

    /// Renames a dimension.
    pub fn merge_dims(&mut self, lhs: ir::DimId, rhs: ir::DimId) {
        self.base.merge_dims(lhs, rhs);
    }

    /// Returns the base operand of the induction variable.
    pub fn base(&self) -> &ir::Operand<'a, L> { &self.base }

    /// Returns the list of induction dimensions along with the corresponding increments.
    pub fn dims(&self) -> &[(ir::DimId, ir::Size<'a>)] { &self.dims }
}

impl<'a> InductionVar<'a, ()> {
    pub fn freeze(self, cnt: &mut ir::Counter) -> InductionVar<'a> {
        InductionVar { dims: self.dims, base: self.base.freeze(cnt) }
    }
}
