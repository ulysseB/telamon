//! Abstracts kernels so we can build generic methods to test them.
use itertools::Itertools;
use telamon::{codegen, device, explorer, ir, model};
use telamon::explorer::montecarlo;
use telamon::helper::SignatureBuilder;
use telamon::model::Bound;
use telamon::search_space::SearchSpace;
use std;

/// Ignore candidates with a too big bound in tests.
const CUT: f64 = 10e8f64;

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
    fn test_correctness<AM>(params: Self::Parameters, num_tests: usize, context: &mut AM)
        where AM: device::ArgMap + device::Context + 'a
    {
        let kernel;
        let signature = {
            let mut builder = SignatureBuilder::new(Self::name(), context);
            kernel = Self::build_signature(params, true, &mut builder);
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
        while num_runs < num_tests {
            let order = explorer::config::NewNodeOrder::WeightedRandom;
            let candidate_idx = montecarlo::next_cand_index(order, &candidates, CUT);
            let candidate = candidates[unwrap!(candidate_idx)].clone();
            let leaf = montecarlo::descend(order, context, candidate, CUT);
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
        println!("num_deadends: {}", num_deadends);
    }

    /// Tests the correctness of the bound of kernels and returns the list of tested leafs
    /// along with the actual evaluation time.
    fn test_bound<AM>(params: Self::Parameters, num_tests: usize, context: &mut AM)
        -> Vec<(Vec<explorer::choice::ActionEx>, Bound, f64)>
            where AM: device::ArgMap + device::Context + 'a
    {
        let kernel;
        let signature = {
            let mut builder = SignatureBuilder::new(Self::name(), context);
            kernel = Self::build_signature(params, true, &mut builder);
            builder.get()
        };
        let candidates = kernel.build_body(&signature, context.device()).into_iter()
            .map(|space| {
                let bound = model::bound(&space, context);
                explorer::Candidate::new(space, bound)
            }).collect_vec();
        let mut leaves = Vec::new();
        while leaves.len() < num_tests {
            let order = explorer::config::NewNodeOrder::WeightedRandom;
            let mut candidates = std::borrow::Cow::Borrowed(&candidates);
            let mut bounds = Vec::new();
            loop {
                let idx = montecarlo::next_cand_index(order, candidates.iter(), CUT);
                let idx = if let Some(idx) = idx { idx } else { break; };
                bounds.push(candidates[idx].bound.clone());
                let choice_opt = explorer::choice::list(&candidates[idx].space).next();
                if let Some(choice) = choice_opt {
                    let new_nodes = candidates[idx].apply_choice(context, choice).into_iter()
                        .filter(|x| x.bound.value() < CUT).collect_vec();
                    candidates = std::borrow::Cow::Owned(new_nodes);
                } else {
                    let leaf = &candidates[idx];
                    let device_fn = codegen::Function::build(&leaf.space);
                    let runtime = unwrap!(context.evaluate(&device_fn),
                        "evaluation failed for kernel {}, with actions {:?}",
                        Self::name(), leaf.actions);
                    for bound in &bounds {
                        assert!(bound.value() < runtime);
                    }
                    let actions = leaf.actions.iter().cloned().collect();
                    leaves.push((actions, leaf.bound.clone(), runtime));
                    break;
                }
            }
        }
        leaves
    }

    // TODO(test): benchmark method, that compares against reference implementations,
    // dependending on the features enabled.
    // * For this we need a benchmark method in the context
}

// TODO(test): exploit bounds benchmarks
// TODO(test): benchmark the number of deadends
// TODO(test): exploit runtime benchmarks
