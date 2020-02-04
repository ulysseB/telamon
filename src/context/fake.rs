use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use crate::codegen::{self, DevicePrinter};
use crate::device::{fake::Device as FakeDevice, Device, ParamsHolder};
use crate::explorer::Candidate;
use crate::ir;

use super::{
    ArgMap, ArrayArgument, AsyncCallback, AsyncEvaluator, EvalMode, KernelEvaluator,
    ScalarArgument,
};

/// A fake context to use when we don't actually care about the
/// evaluation results.  This wraps any device for the performance
/// model, but always return 1 for all evaluation results.
#[derive(Debug, Default)]
pub struct Context<D = FakeDevice> {
    device: Arc<D>,
    parameters: HashMap<String, Option<u32>>,
}

impl<D: Device> Context<D> {
    pub fn new(device: D) -> Self {
        Context {
            device: Arc::new(device),
            parameters: HashMap::new(),
        }
    }
}

impl<D: Device> ParamsHolder for Context<D> {
    fn param_as_size(&self, name: &str) -> Option<u32> {
        self.parameters[name]
    }
}

impl<D: Device> DevicePrinter for Context<D> {
    fn print(&self, fun: &codegen::Function, out: &mut dyn Write) {}
}

impl<D: Device> super::Context for Context<D> {
    fn device(&self) -> Arc<dyn Device> {
        Arc::<D>::clone(&self.device)
    }

    fn params(&self) -> &dyn ParamsHolder {
        self
    }

    fn printer(&self) -> &dyn DevicePrinter {
        self
    }

    fn evaluate(&self, _: &codegen::Function, _: EvalMode) -> Result<f64, ()> {
        Ok(1.0)
    }

    fn benchmark(&self, _: &codegen::Function, num_samples: usize) -> Vec<f64> {
        vec![1.0; num_samples]
    }

    fn async_eval<'c>(
        &self,
        _: usize,
        _: EvalMode,
        inner: &(dyn Fn(&mut dyn AsyncEvaluator<'c>) + Sync),
    ) {
        struct FakeEvaluator;

        impl<'b> AsyncEvaluator<'b> for FakeEvaluator {
            fn add_dyn_kernel(
                &mut self,
                candidate: Candidate,
                callback: AsyncCallback<'b>,
            ) {
                use std::fmt;

                struct FakeCode;

                impl fmt::Display for FakeCode {
                    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                        write!(fmt, "<...>")
                    }
                }

                impl KernelEvaluator for FakeCode {
                    fn evaluate(&mut self) -> Option<f64> {
                        Some(1.)
                    }
                }

                codegen::Function::build(&candidate.space);
                callback.call(candidate, &mut FakeCode);
            }
        }

        inner(&mut FakeEvaluator);
    }
}

impl<D: Device> ArgMap for Context<D> {
    fn bind_erased_scalar(
        &mut self,
        param: &ir::Parameter,
        value: Box<dyn ScalarArgument>,
    ) {
        assert_eq!(param.t, value.get_type());

        self.parameters.insert(param.name.clone(), value.as_size());
    }

    fn bind_erased_array(
        &mut self,
        _: &ir::Parameter,
        _: ir::Type,
        _: usize,
    ) -> Arc<dyn ArrayArgument> {
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
