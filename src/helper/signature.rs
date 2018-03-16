//! Helper functions to create a function signature and bind parameters.
use device;
use ir::{Signature, Parameter, mem};

/// Helper struct to build a `Signature`.
pub struct Builder<'a, 'b> where 'b: 'a {
    context: &'a mut device::Context<'b>,
    signature: Signature,
}

impl<'a, 'b> Builder<'a, 'b> {
    /// Creates a new builder for a function with the given name.
    pub fn new(name: &str, context: &'a mut device::Context<'b>) -> Self {
        let signature = Signature {
            name: name.to_string(),
            params: vec![],
            mem_blocks: 0
        };
        Builder { context, signature }
    }

    /// Creates a new parameter and binds it to the given value.
    pub fn param<T: device::Argument + 'b>(&mut self, name: &str, arg: T) {
        let param = Parameter { name: name.to_string(), t: arg.t(), };
        self.context.bind_param(&param, Box::new(arg));
        self.signature.params.push(param);
    }

    /// Allocates an array ID.
    pub fn alloc_array_id(&mut self) -> mem::Id {
        let id = mem::Id::External(self.signature.mem_blocks);
        self.signature.mem_blocks += 1;
        id
    }

    /// Creates a new parameter and binds it to a freshly allocated an array.
    pub fn array(&mut self, name: &str, size: usize) -> mem::Id {
        let id = self.alloc_array_id();
        let array = self.context.allocate_array(id, size);
        let param = Parameter { name: name.to_string(), t: array.t(), };
        self.context.bind_param(&param, array);
        self.signature.params.push(param);
        id
    }

    /// Returns the `Signature` created by the builder.
    pub fn get(self) -> Signature { self.signature }
}
