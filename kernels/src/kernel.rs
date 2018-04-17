//! Abstracts kernels so we can build generic methods to test them.
use itertools::Itertools;
use telamon::{codegen, device, explorer, ir, model};
use telamon::explorer::montecarlo;
use telamon::helper::SignatureBuilder;
use telamon::search_space::SearchSpace;

/// A kernel that can be compiled, benchmarked and used for correctness tests.
pub trait Kernel<'a>: Sized {
    /// The input parameters of the kernel.
    type Parameters;
    /// The values to expect as output.
    type ExpectedOutput;

    /// The name of the function computed by the kernel.
    fn name() -> &'static str;

    /// Builds the signature of the kernel in the builder and returns an object that
    /// stores enough information to later build the kernel body and check its result.
    /// The `is_generic` flag indicates if th sizes should be instantiated.
    fn build_signature<AM>(parameters: Self::Parameters,
                           is_generic: bool,
                           builder: &mut SignatureBuilder<AM>) -> Self
        where AM: device::ArgMap + device::Context + 'a;

    /// Builder the kernel body in the given builder. This builder should be based on the
    /// signature created by `build_signature`.
    fn build_body<'b>(&self, signature: &'b ir::Signature, device: &'b device::Device)
        -> Vec<SearchSpace<'b>>;

    /// Computes the expected output.
    fn get_expected_output(&self, &device::Context) -> Self::ExpectedOutput;

    /// Ensures the generated code performs the correct operation.
    fn check_result(&self, expected: &Self::ExpectedOutput, context: &device::Context)
        -> Result<(), String>;

    /// Generates, executes and tests the output of candidates for the kernel.
    fn test_correctness<AM>(params: Self::Parameters,
                            num_descend: usize,
                            context: &mut AM)
        where AM: device::ArgMap + device::Context + 'a
    {
        let kernel;
        let signature = {
            let mut builder = SignatureBuilder::new(Self::name(), context);
            kernel = Self::build_signature(params, false, &mut builder);
            builder.get()
        };
        let expected_output = kernel.get_expected_output(context);
        let candidates = kernel.build_body(&signature, context.device()).into_iter()
            .map(|space| {
                let bound = model::bound(&space, context);
                explorer::Candidate::new(space, bound)
            }).collect_vec();
        let mut num_deadends = 0;
        let mut num_runs = 0;
        while num_runs < num_descend {
            let order = explorer::config::NewNodeOrder::WeightedRandom;
            let cut = 10e8f64;
            let candidate_idx = montecarlo::next_cand_index(order, &candidates, cut);
            let candidate = candidates[unwrap!(candidate_idx)].clone();
            let leaf = montecarlo::descend(order, context, candidate, cut);
            if let Some(leaf) = leaf {
                let device_fn = codegen::Function::build(&leaf.space);
                unwrap!(context.evaluate(&device_fn),
                    "evaluation failed for kernel {}, with actions {:?}",
                    Self::name(), leaf.actions);
                if let Err(err) = kernel.check_result(&expected_output, context) {
                    panic!("incorrect output for kernel {}, with actions {:?}: {}",
                           Self::name(), leaf.actions, err)
                }
                num_runs += 1;
            } else {
                num_deadends += 1;
                if num_deadends >= 20 * (1+num_runs) {
                    panic!("too many dead-ends for kernel {}", Self::name())
                }
            }
        }
        // FIXME: use an epsilon ?
        println!("num_deadends: {}", num_deadends);
    }

    // TODO(test): check bound method
    // TODO(test): benchmark method, that compares against reference implementations,
    // dependending on the features enabled.
    // * For this we need a benchmark method in the context
    // TODO(test): benchmark the number of deadends
}
