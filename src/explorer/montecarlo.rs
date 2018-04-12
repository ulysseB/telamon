//! File containing all functions relative to expanding a candidate by a simple montecarlo search

use device::Context;
use itertools::Itertools;
use explorer::candidate::Candidate;
use explorer::choice;
use explorer::config::NewNodeOrder;
use rand::{Rng, thread_rng};
use rand::distributions::{ Weighted, WeightedChoice, IndependentSample};
use std;
use utils::*;

/// A recursive function that takes a candidate and expands it until we have a completely specified
/// candidate that we can pass to the evaluator, or we find a dead-end
pub fn descend<'a>(order: NewNodeOrder,
                   context: &Context,
                   candidate: Candidate<'a>,
                   cut: f64) -> Option<Candidate<'a>> {
    let choice_opt = choice::list(&candidate.space).next();
    if let Some(choice) = choice_opt {
        let new_nodes = candidate.apply_choice(context, choice).into_iter()
            .filter(|x| x.bound.value() < cut)
            .collect_vec();
        choose_next_cand(order, new_nodes, cut)
            .and_then(|node| descend(order, context, node, cut))
    } else { Some(candidate) }
}

/// Called in montecarlo_descend, dispatch the choice of the next candidate according to our
/// configuration
pub fn choose_next_cand<'a>(order: NewNodeOrder,
                            mut new_nodes: Vec<Candidate<'a>>,
                            cut: f64) -> Option<Candidate<'a>> {
    let idx = next_cand_index(order, &new_nodes, cut);
    idx.map(|idx| new_nodes.remove(idx))
}

/// Returns the index of the next candidate to consider.
pub fn next_cand_index<'a, IT>(order: NewNodeOrder, nodes: IT, cut: f64) -> Option<usize>
    where IT: IntoIterator<Item=&'a Candidate<'a>>, IT::IntoIter: Clone
{
    match order {
        NewNodeOrder::Api => {
            if nodes.into_iter().next().is_some() { Some(0) } else { None }
        },
        NewNodeOrder::WeightedRandom => choose_cand_weighted(nodes, cut),
        NewNodeOrder::Bound => choose_cand_best(nodes),
        NewNodeOrder::Random => choose_cand_rand(nodes),
    }
}


/// Given a vector of candidate reference, returns the index of the candidate with the minimum
/// bound.
fn choose_cand_best<'a, IT>(nodes: IT) -> Option<usize>
    where IT: IntoIterator<Item=&'a Candidate<'a>>
{
    nodes.into_iter().enumerate()
        .min_by(|x1, x2| cmp_f64(x1.1.bound.value(), x2.1.bound.value()))
        .map(|x| x.0)
}

/// Given a vector of candidate reference, just choose randomly the next candidate
fn choose_cand_rand<'a, IT>(nodes: IT) -> Option<usize>
    where IT: IntoIterator<Item=&'a Candidate<'a>>
{
    let len = nodes.into_iter().count();
    if len == 0 { None } else { Some(thread_rng().gen_range(0, len)) }
}

/// Given a vector of candidate references, returns the index of a weighted sort on the candidate
/// bounds
fn choose_cand_weighted<'a, IT>(nodes: IT, cut: f64) -> Option<usize>
    where IT: IntoIterator<Item=&'a Candidate<'a>>, IT::IntoIter: Clone
{
    let mut weighted_items = vec![];
    let mut rng = thread_rng();
    let nodes = nodes.into_iter();
    let max_bound = nodes.clone()
        .max_by(|x1, x2| cmp_f64(x1.bound.value(), x2.bound.value()))
        .map(|x| x.bound.value())?;
    for (ind, x) in nodes.enumerate() {
        if cut.is_infinite() {
            let x_weight = (10f64 * max_bound / x.bound.value()).floor() as u32 ;
            weighted_items.push(Weighted{weight: x_weight, item: ind});
        } else {
            assert!(x.bound.value() <= cut,
                "Compare bound fail, cut {:.3e}, cand: {:.3e}", cut, x.bound.value());
            let weight = (1000f64 * (1f64 - x.bound.value()/cut)).floor() as u32;
            let weight = std::cmp::max(1, weight);
            weighted_items.push(Weighted { weight, item: ind});
        }
    }
    Some(WeightedChoice::new(&mut weighted_items).ind_sample(&mut rng))
}
