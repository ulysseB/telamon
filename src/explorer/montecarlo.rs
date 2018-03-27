//! File containing all functions relative to expanding a candidate by a simple montecarlo search

use device::Context;
use itertools::Itertools;
use explorer::candidate::Candidate;
use explorer::choice;
use explorer::config::{BanditConfig, NewNodeOrder};
use rand::{Rng, thread_rng};
use rand::distributions::{ Weighted, WeightedChoice, IndependentSample};
use std;
use utils::*;

/// A recursive function that takes a candidate and expands it until we have a completely specified
/// candidate that we can pass to the evaluator, or we find a dead-end
pub fn montecarlo_descend<'a>(config: &BanditConfig, 
                          context: &Context, 
                          candidate: Candidate<'a>, 
                          cut: f64) 
    -> Option<Candidate<'a>>
{
    let choice_opt = choice::list(&candidate.space).next();
    if let Some(choice) = choice_opt {
        let new_nodes = candidate.apply_choice(context, choice).into_iter()
            .filter(|x| x.bound.value() < cut)
            .collect_vec();
        if new_nodes.is_empty() {
            None
        } else { 
            let chosen_candidate = choose_next_cand(config, new_nodes, cut);
            montecarlo_descend(config, context, chosen_candidate, cut)
        } 
    }
    else { Some(candidate) }
}

/// Called in montecarlo_descend, dispatch the choice of the next candidate according to our
/// configuration
pub fn choose_next_cand<'a>(config: &BanditConfig, 
                        mut new_nodes: Vec<Candidate<'a>>, 
                        cut: f64) -> Candidate<'a> {
    let ind = match config.new_nodes_order {
        NewNodeOrder::Api => 0,
        NewNodeOrder::WeightedRandom => choose_cand_weighted(&new_nodes.iter().collect_vec(), cut),
          NewNodeOrder::Bound => choose_cand_best(&new_nodes.iter().collect_vec()),
          NewNodeOrder::Random => choose_cand_rand(&new_nodes.iter().collect_vec()),
    };
    new_nodes.remove(ind)
}


/// Given a vector of candidate reference, returns the index of the candidate with the minimum
/// bound
pub fn choose_cand_best<'a>(new_nodes: &Vec<&Candidate<'a>>) -> usize {
    new_nodes.iter().enumerate().min_by(|x1, x2| cmp_f64(x1.1.bound.value(), x2.1.bound.value()))
        .map(|x| x.0)
    // We checked in montecarlo_descend that new_nodes is not empty
        .unwrap()
}

/// Given a vector of candidate reference, just choose randomly the next candidate
pub fn choose_cand_rand<'a>(new_nodes: &Vec<&Candidate<'a>>) -> usize {
    let mut rng = thread_rng();
    rng.gen_range(0, new_nodes.len())
}

/// Given a vector of candidate references, returns the index of a weighted sort on the candidate
/// bounds
pub fn choose_cand_weighted<'a>(new_nodes: &Vec<&'a Candidate<'a>>, cut: f64) -> usize {
    let mut weighted_items = vec![];
    let mut rng = thread_rng();
    let max_bound = new_nodes.iter().max_by(|x1, x2|
                                            cmp_f64(x1.bound.value(), x2.bound.value()))
        .map(|x| x.bound.value()).expect("New_nodes should not be empty !!!!");
    for (ind, x) in new_nodes.iter().enumerate() {
        if cut.is_infinite() {
            let x_weight = (10f64 * max_bound / x.bound.value()).floor() as u32 ;
            weighted_items.push(Weighted{weight: x_weight, item: ind});
        } else {
            assert!(x.bound.value() <= cut, "Compare bound fail, cut {:.3e}, cand: {:.3e}", cut, x.bound.value());
            let weight = (1000f64 * (1f64 - x.bound.value()/cut)).floor() as u32;
            let weight = std::cmp::max(1, weight);
            weighted_items.push(Weighted { weight, item: ind});
        }
    }
    WeightedChoice::new(&mut weighted_items).ind_sample(&mut rng)
}
