//! Helper functions to create a function signature and bind parameters.
use crate::device::{self, ArgMapExt, ArrayArgumentExt, ScalarArgument};
use crate::helper::tensor::{DimSize, Tensor};
use crate::ir::Signature;
use itertools::Itertools;
use rand::prelude::*;
use std::sync::Arc;
use utils::unwrap;

/// Memory initialization strategies.
#[derive(Debug, Copy, Clone)]
pub enum MemInit {
    /// Memory is randomly filled.  This is the default behavior.
    RandomFill,
    /// Memory is left uninitialized
    Uninit,
}

impl Default for MemInit {
    fn default() -> Self {
        MemInit::RandomFill
    }
}

/// Helper struct to build a `Signature`.
pub struct Builder<'a, AM>
where
    AM: device::Context + 'a,
{
    mem_init: MemInit,
    rng: rand::XorShiftRng,
    context: &'a mut AM,
    signature: Signature,
}

impl<'a, AM> Builder<'a, AM>
where
    AM: device::Context + 'a,
{
    /// Creates a new builder for a function with the given name.
    pub fn new(name: &str, context: &'a mut AM) -> Self {
        let signature = Signature {
            name: name.to_string(),
            params: vec![],
        };
        //let rng = rand::XorShiftRng::from_seed(Default::default());
        let rng = rand::XorShiftRng::from_rng(rand::thread_rng()).unwrap();
        Builder {
            mem_init: MemInit::Uninit,
            context,
            signature,
            rng,
        }
    }

    /// Arrays are filled with random data if set to true.
    pub fn set_mem_init(&mut self, mem_init: MemInit) {
        self.mem_init = mem_init;
    }

    /// Creates a new parameter and binds it to the given value.
    pub fn scalar<'b, T: ScalarArgument>(&mut self, name: &str, arg: T)
    where
        AM: device::ArgMap<'b>,
    {
        self.signature.add_scalar(name.to_string(), T::t());
        let param = unwrap!(self.signature.params.last());
        self.context.bind_scalar(param, arg);
    }

    /// Creates a new `i32` paramter and returns a size equals to this parameter. Sets the
    /// maximal size to the current size.
    pub fn max_size<'b>(&mut self, name: &'b str, size: u32) -> DimSize<'b>
    where
        AM: device::ArgMap<'b>,
    {
        self.scalar(name, size as i32);
        DimSize {
            factor: 1,
            params: vec![name],
            max_size: size,
        }
    }

    /// Creates a new parameter and binds it to a freshly allocated an array.
    pub fn array<'b, S: ScalarArgument>(
        &mut self,
        name: &str,
        size: usize,
    ) -> Arc<dyn device::ArrayArgument + 'b>
    where
        AM: device::ArgMap<'b>,
    {
        self.signature
            .add_array(&*self.context.device(), name.to_string(), S::t());
        let param = unwrap!(self.signature.params.last());
        let array = self.context.bind_array::<S>(param, size);
        let rng = &mut self.rng;
        match self.mem_init {
            MemInit::RandomFill => {
                let random = (0..size).map(|_| S::gen_random(rng)).collect_vec();
                array.as_ref().write(&random);
            }
            MemInit::Uninit => (),
        }
        array
    }

    /// Allocates an n-dimensional array.
    pub fn tensor<'b, S: ScalarArgument>(
        &mut self,
        name: &'b str,
        dim_sizes: Vec<DimSize<'b>>,
        read_only: bool,
    ) -> Tensor<'b, S>
    where
        AM: device::ArgMap<'b>,
    {
        let len = dim_sizes
            .iter()
            .map(|s| s.eval(self.context) as usize)
            .product::<usize>();
        let array = self.array::<S>(name, len);
        Tensor::new(name, dim_sizes, read_only, array)
    }

    /// Returns the `Signature` created by the builder.
    pub fn get(self) -> Signature {
        self.signature
    }

    /// Returns the underlying context.
    pub fn context(&self) -> &AM {
        self.context
    }
}
