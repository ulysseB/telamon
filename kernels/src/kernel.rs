//! Abstracts kernels so we can build generic methods to test them.
use itertools::Itertools;
use std;
use telamon::{codegen, device, explorer, ir, model};
use telamon::explorer::montecarlo;
use telamon::helper::SignatureBuilder;
use telamon::search_space::SearchSpace;

/// Indicates what is being tested.
pub enum TestKind { Functional, Bound }

/// A kernel that can be compiled, benchmarked and used for correctness tests.
pub trait Kernel: Sized {
    /// The input parameters of the kernel.
    type Parameters;

    /// The name of the function computed by the kernel.
    fn name() -> &'static str;

    /// Builds the signature of the kernel in the builder and returns an object that
    /// stores enough information to later build the kernel body and check its result.
    /// The `is_generic` flag indicates if th sizes should be instantiated.
    fn build_signature<AM>(parameters: Self::Parameters,
                           is_generic: bool,
                           builder: &mut SignatureBuilder<AM>) -> Self
        where AM: device::ArgMap + device::Context;

    /// Builder the kernel body in the given builder. This builder should be based on the
    /// signature created by `build_signature`.
    fn build_body<'a>(&self, signature: &'a ir::Signature, device: &'a device::Device)
        -> Vec<SearchSpace<'a>>;

    /// Ensures the generated code performs the correct operation.
    fn check_result(&self, context: &device::Context) -> Result<(), String>;

    /// Generates, executes and tests the output of candidates for the kernel.
    fn test<AM>(test_kind: TestKind,
                params: Self::Parameters,
                num_descend: usize,
                context: &mut AM)
        where AM: device::ArgMap + device::Context
    {
        let kernel;
        let signature = {
            let mut builder = SignatureBuilder::new(Self::name(), context);
            kernel = Self::build_signature(params, false, &mut builder);
            builder.get()
        };
        let candidates = kernel.build_body(&signature, context.device()).into_iter()
            .map(|space| {
                let bound = model::bound(&space, context);
                explorer::Candidate::new(space, bound)
            }).collect_vec();
        let mut num_deadends = 0;
        let mut num_runs = 0;
        while num_runs < num_descend {
            let order = explorer::config::NewNodeOrder::WeightedRandom;
            let cut = std::f64::INFINITY;
            let candidate_idx = montecarlo::next_cand_index(order, &candidates, cut);
            let candidate = candidates[unwrap!(candidate_idx)].clone();
            let leaf = montecarlo::descend(order, context, candidate, cut);
            if let Some(leaf) = leaf {
                let device_fn = codegen::Function::build(&leaf.space);
                let runtime = unwrap!(context.evaluate(&device_fn),
                    "evaluation failed for kernel {}, with actions {:?}",
                    Self::name(), leaf.actions);
                match test_kind {
                    TestKind::Functional => {
                        if let Err(err) = kernel.check_result(context) {
                            panic!("incorrect output for kernel {}, with actions {:?}: {}",
                                   Self::name(), leaf.actions, err)
                        }
                    },
                    TestKind::Bound => {
                        if leaf.bound.value() > runtime {
                            panic!("bound {} > {}ns for kernel {} with actions {:?}",
                                   leaf.bound, runtime, Self::name(), leaf.actions)
                        }
                    },
                }
                num_runs += 1;
            } else {
                num_deadends += 1;
                if num_deadends >= 10 * (1+num_runs) {
                    panic!("too many dead-ends for kernel {}", Self::name())
                }
            }
        }
    }

    // FIXME: benchmark method, that compares against reference implementations, dependending on
    // the features enabled
}

/// A kernel that can be compiled on CUDA GPUs.
#[cfg(feature="cuda")]
pub trait CudaKernel: Kernel {
    /// Returns the execution time (in nanoseconds) of the kernel using the reference
    /// implementation on CUDA.
    fn benchmark_fast(&self) -> f64;
}
