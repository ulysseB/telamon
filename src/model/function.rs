pub struct Function<'a> {
    space: &'a SearchSpace,
    context: &'a dyn Context,

    dim_sizes: DimSize, // TODO: RefCell?
}

impl<'a> Function<'a> {
    fn new(space: &'a SearchSpace, context: &'a dyn Context) -> Self {
        Function { context, space }
    }

    fn context(&self) -> &dyn Context {
        self.context
    }

    fn ir_instance(&self) -> &ir::Function {
        self.space.ir_instance()
    }

    fn domain(&self) -> &DomainStore {
        self.space.domain()
    }
}
