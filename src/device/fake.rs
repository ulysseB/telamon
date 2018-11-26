use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use codegen;
use explorer::Candidate;
use ir;

use super::{
    ArgMap, ArrayArgument, AsyncCallback, AsyncEvaluator, Context, Device, EvalMode,
    ScalarArgument,
};

/// A fake context to use when we don't actually care about the
/// evaluation results.  This wraps any device for the performance
/// model, but always return 1 for all evaluation results.
#[derive(Debug, Default)]
pub struct FakeContext<D> {
    device: D,
    parameters: HashMap<String, Option<u32>>,
}

impl<D: Device> FakeContext<D> {
    pub fn new(device: D) -> Self {
        FakeContext {
            device,
            parameters: HashMap::new(),
        }
    }
}

impl<D: Device> Context for FakeContext<D> {
    fn device(&self) -> &Device {
        &self.device
    }

    fn evaluate(&self, _: &codegen::Function, _: EvalMode) -> Result<f64, ()> {
        Ok(1.0)
    }

    fn benchmark(&self, _: &codegen::Function, num_samples: usize) -> Vec<f64> {
        vec![1.0; num_samples]
    }

    fn param_as_size(&self, name: &str) -> Option<u32> {
        self.parameters[name]
    }

    fn async_eval<'b, 'c>(
        &self,
        _: usize,
        _: EvalMode,
        inner: &(Fn(&mut AsyncEvaluator<'b, 'c>) + Sync),
    ) {
        struct FakeEvaluator<'a, 'b> {
            phantom: PhantomData<(&'a (), &'b ())>,
        }

        impl<'a, 'b, 'c> AsyncEvaluator<'a, 'c> for FakeEvaluator<'a, 'b>
        where
            'a: 'b,
            'c: 'b,
        {
            fn add_kernel(
                &mut self,
                candidate: Candidate<'a>,
                callback: AsyncCallback<'a, 'c>,
            ) {
                codegen::Function::build(&candidate.space);
                callback.call(candidate, 1.0);
            }
        }

        inner(&mut FakeEvaluator {
            phantom: PhantomData,
        });
    }
}

impl<D: Device> ArgMap for FakeContext<D> {
    type Array = FakeArray;

    fn bind_scalar<S: ScalarArgument>(&mut self, param: &ir::Parameter, value: S) {
        assert_eq!(param.t, S::t());

        self.parameters.insert(param.name.clone(), value.as_size());
    }

    fn bind_array<S: ScalarArgument>(
        &mut self,
        _: &ir::Parameter,
        _: usize,
    ) -> Arc<Self::Array> {
        Arc::new(FakeArray)
    }
}

/// A fake array implementation which doesn't read or write anything.
pub struct FakeArray;

impl ArrayArgument for FakeArray {
    fn read_i8(&self) -> Vec<i8> {
        Vec::new()
    }

    fn write_i8(&self, _: &[i8]) {}
}
