//! Abstracts kernels so we can build generic methods to test them.
use itertools::Itertools;
use num_cpus;
use rayon::prelude::*;
use std::sync::{atomic, Mutex};
use telamon::{codegen, device, explorer, ir, model};
use telamon::explorer::{Candidate, local_selection};
use telamon::helper::SignatureBuilder;
use telamon::model::Bound;
use telamon::search_space::SearchSpace;
use std;

/// Ignore candidates with a too big bound in tests.
const CUT: f64 = 10e8f64;
/// Maximal number of deadends to accept before failing.
// TODO(cleanup): tune MAX_DEADEND_RATIO
const MAX_DEADEND_RATIO: usize = 20;

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
                Candidate::new(space, bound)
            }).collect_vec();
        let mut num_deadends = 0;
        let mut num_runs = 0;
        while num_runs < num_tests {
            let order = explorer::config::NewNodeOrder::WeightedRandom;
            let bounds = candidates.iter().map(|c| c.bound.value()).enumerate();
            let candidate_idx = local_selection::pick_index(order, bounds, CUT);
            let candidate = candidates[unwrap!(candidate_idx)].clone();
            let leaf = local_selection::descend(order, context, candidate, CUT);
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
                if num_deadends >= MAX_DEADEND_RATIO * (1+num_runs) {
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
                Candidate::new(space, bound)
            }).collect_vec();
        let leaves = Mutex::new(Vec::new());
        let num_tested = atomic::AtomicUsize::new(0);
        context.async_eval(num_cpus::get(), &|evaluator| {
            while num_tested.fetch_add(1, atomic::Ordering::SeqCst) < num_tests {
                if let Some((leaf, bounds)) = descend_check_bounds(&candidates, context) {
                    let leaves = &leaves;
                    evaluator.add_kernel(leaf, (move |leaf: Candidate, runtime: f64, _| {
                        for bound in &bounds { assert!(bound.value() < runtime); }
                        let actions = leaf.actions.iter().cloned().collect();
                        let mut leaves = unwrap!(leaves.lock());
                        leaves.push((actions, leaf.bound.clone(), runtime));
                    }).into());
                } else {
                    num_tested.fetch_sub(1, atomic::Ordering::SeqCst);
                }
            }
        });
        unwrap!(leaves.into_inner())
    }

    /// Runs the search and benchmarks the resulting candidate.
    fn benchmark<AM>(config: &explorer::Config,
                     params: Self::Parameters,
                     num_samples: usize,
                     context: &mut AM) -> Vec<f64>
        where AM: device::ArgMap + device::Context + 'a
    {
        let kernel;
        let signature = {
            let mut builder = SignatureBuilder::new(Self::name(), context);
            kernel = Self::build_signature(params, true, &mut builder);
            builder.get()
        };
        let search_space = kernel.build_body(&signature, context.device());
        let best = unwrap!(explorer::find_best(config, context, search_space),
                           "no candidates found for kernel {}", Self::name());
        let best_fn = codegen::Function::build(&best);
        context.benchmark(&best_fn, num_samples)
    }

    /// Computes the probability of encountering a dead-end when descending in the search
    /// tree.
    fn deadend_ratio<AM>(params: Self::Parameters,
                         num_samples: usize,
                         context: &mut AM) -> f64
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
                Candidate::new(space, bound)
            }).collect_vec();
        let num_deadends = (0..num_samples).into_par_iter().filter(|_| {
            let order = explorer::config::NewNodeOrder::WeightedRandom;
            let inf = std::f64::INFINITY;
            let bounds = candidates.iter().map(|c| c.bound.value()).enumerate();
            let candidate_idx = local_selection::pick_index(order, bounds, inf);
            let candidate = candidates[unwrap!(candidate_idx)].clone();
            local_selection::descend(order, context, candidate, inf).is_none()
        }).count();
        num_deadends as f64 / num_samples as f64
    }
}

/// Descend along a path in the search tree and stores the bounds encountered on the way.
fn descend_check_bounds<'a>(candidates: &[Candidate<'a>], context: &device::Context)
    -> Option<(Candidate<'a>, Vec<Bound>)>
{
    let order = explorer::config::NewNodeOrder::WeightedRandom;
    let mut candidates = std::borrow::Cow::Borrowed(candidates);
    let mut bounds = Vec::new();
    loop {
        let idx = if let Some(idx) = {
            let idx_bounds = candidates.iter().map(|c| c.bound.value()).enumerate();
            local_selection::pick_index(order, idx_bounds, CUT)
        } { idx } else { return None };
        bounds.push(candidates[idx].bound.clone());
        let choice_opt = explorer::choice::list(&candidates[idx].space).next();
        if let Some(choice) = choice_opt {
            let new_nodes = candidates[idx].apply_choice(context, choice).into_iter()
                .filter(|x| x.bound.value() < CUT).collect_vec();
            candidates = std::borrow::Cow::Owned(new_nodes);
        } else {
            return Some((candidates[idx].clone(), bounds));
        }
    }
}

// TODO(test): exploit bounds benchmarks
// TODO(test): benchmark the number of deadends
