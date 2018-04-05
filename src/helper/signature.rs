//! Helper functions to create a function signature and bind parameters.
use device::{self, ScalarArgument};
use ir::{self, Signature, Parameter, mem};
use helper::tensor::{DimSize, Tensor};
use std::sync::Arc;

/// Helper struct to build a `Signature`.
pub struct Builder<'a, AM> where AM: device::ArgMap + device::Context + 'a {
    context: &'a mut AM,
    signature: Signature,
}

impl<'a, AM> Builder<'a, AM> where AM: device::ArgMap + device::Context + 'a,
{
    /// Creates a new builder for a function with the given name.
    pub fn new(name: &str, context: &'a mut AM) -> Self {
        let signature = Signature {
            name: name.to_string(),
            params: vec![],
            mem_blocks: 0
        };
        Builder { context, signature }
    }

    /// Creates a new parameter and binds it to the given value.
    pub fn scalar<T: ScalarArgument>(&mut self, name: &str, arg: T) {
        let param = Parameter { name: name.to_string(), t: T::t(), };
        self.context.bind_scalar(&param, arg);
        self.signature.params.push(param);
    }

    /// Creates a new parameter and binds it to a freshly allocated an array.
    pub fn array<S: ScalarArgument>(&mut self, name: &str, size: usize)
        -> (ir::mem::Id, Arc<AM::Array>)
    {
        let id = self.alloc_array_id();
        let param = Parameter { name: name.to_string(), t: ir::Type::PtrTo(id), };
        let array = self.context.bind_array::<S>(&param, size);
        self.signature.params.push(param);
        (id, array)
    }

    /// Allocates an n-dimensional array.
    pub fn tensor<'b, S: ScalarArgument>(&mut self, name: &'b str,
                                         dim_sizes: Vec<DimSize<'b>>,
                                         read_only: bool) -> Tensor<'b> {
        let len = dim_sizes.iter().map(|&s| self.eval_size(s) as usize)
            .product::<usize>();
        let mem_id = self.array::<S>(name, len).0;
        Tensor::new(name, dim_sizes, S::t(), read_only, mem_id)
    }

    /// Evaluates a size in the context.
    pub fn eval_size(&self, size: DimSize) -> u32 {
        match size {
            DimSize::Const(s) => s,
            DimSize::Param(p) => unwrap!(self.context.param_as_size(p)),
        }
    }

    /// Returns the `Signature` created by the builder.
    pub fn get(self) -> Signature { self.signature }

    /// Allocates an array ID.
    fn alloc_array_id(&mut self) -> mem::Id {
        let id = mem::Id::External(self.signature.mem_blocks);
        self.signature.mem_blocks += 1;
        id
    }
}
