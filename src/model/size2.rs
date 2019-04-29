struct DimSizes {
    sizes: FnvHashMap<ir::DimId, Result<u64, DimSize>>,
}

impl DimSizes {
    fn new(
        space: &SearchSpace,
        context: &dyn Context,
    ) -> FnvHashMap<ir::DimId, Sym<DimSize, u64>> {
    }
}
