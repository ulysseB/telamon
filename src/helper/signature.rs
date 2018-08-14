//! Helper functions to create a function signature and bind parameters.
use device::{self, ScalarArgument, write_array};
use ir::{self, Signature, Parameter};
use itertools::Itertools;
use helper::tensor::{DimSize, Tensor};
use rand;
use std::sync::Arc;

/// Helper struct to build a `Signature`.
pub struct Builder<'a, AM> where AM: device::ArgMap + device::Context + 'a {
    rng: rand::XorShiftRng,
    context: &'a mut AM,
    signature: Signature,
}

impl<'a, AM> Builder<'a, AM> where AM: device::ArgMap + device::Context + 'a {
    /// Creates a new builder for a function with the given name.
    pub fn new(name: &str, context: &'a mut AM) -> Self {
        let signature = Signature {
            name: name.to_string(),
            params: vec![],
            mem_blocks: 0
        };
        let rng = rand::XorShiftRng::new_unseeded();
        Builder { context, signature, rng }
    }

    /// Creates a new parameter and binds it to the given value.
    pub fn scalar<T: ScalarArgument>(&mut self, name: &str, arg: T) {
        let param = Parameter { name: name.to_string(), t: T::t(), };
        self.context.bind_scalar(&param, arg);
        self.signature.params.push(param);
    }

    /// Creates a new parameter and binds it to a freshly allocated an array.
    pub fn array<S: ScalarArgument>(&mut self, name: &str, size: usize)
        -> (ir::MemId, Arc<AM::Array>)
    {
        let id = self.alloc_array_id();
        let param = Parameter { name: name.to_string(), t: ir::Type::PtrTo(id), };
        let array = self.context.bind_array::<S>(&param, size);
        let random = (0..size).map(|_| S::gen_random(&mut self.rng)).collect_vec();
        write_array(array.as_ref(), &random);
        self.signature.params.push(param);
        (id, array)
    }

    /// Allocates an n-dimensional array.
    pub fn tensor<'b, S: ScalarArgument>(&mut self, name: &'b str,
                                         dim_sizes: Vec<DimSize<'b>>,
                                         read_only: bool) -> Tensor<'b, S>
        where <AM as device::ArgMap>::Array: 'b
    {
        let len = dim_sizes.iter().map(|s| s.eval(self.context) as usize)
            .product::<usize>();
        let (mem_id, array) = self.array::<S>(name, len);
        Tensor::new(name, dim_sizes, read_only, mem_id, array)
    }

    /// Returns the `Signature` created by the builder.
    pub fn get(self) -> Signature { self.signature }

    /// Returns the underlying context.
    pub fn context(&self) -> &AM { self.context }

    /// Allocates an array ID.
    fn alloc_array_id(&mut self) -> ir::MemId {
        let id = ir::MemId::External(self.signature.mem_blocks);
        self.signature.mem_blocks += 1;
        id
    }
}
