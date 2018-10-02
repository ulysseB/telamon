//! Provides different methods to select a candidate in a list.
use device::Context;
use explorer::candidate::Candidate;
use explorer::choice;
use explorer::config::NewNodeOrder;
use rand::distributions::{IndependentSample, Weighted, WeightedChoice};
use rand::{thread_rng, Rng};
use std;
use utils::*;

/// A recursive function that takes a candidate and expands it until we have a
/// completely specified candidate that we can pass to the evaluator, or we
/// find a dead-end
pub fn descend<'a>(
    order: NewNodeOrder,
    context: &Context,
    candidate: Candidate<'a>,
    cut: f64,
) -> Option<Candidate<'a>>
{
    let choice_opt = choice::list(&candidate.space).next();
    if let Some(choice) = choice_opt {
        let new_nodes = candidate.apply_choice(context, choice);
        pick_candidate(order, new_nodes, cut)
            .and_then(|node| descend(order, context, node, cut))
    } else {
        Some(candidate)
    }
}

/// Called in montecarlo_descend, dispatch the choice of the next candidate
/// according to our configuration
pub fn pick_candidate<'a>(
    order: NewNodeOrder,
    mut new_nodes: Vec<Candidate<'a>>,
    cut: f64,
) -> Option<Candidate<'a>>
{
    let idx = {
        let items = new_nodes.iter().map(|c| c.bound.value()).enumerate();
        pick_index(order, items, cut)
    };
    idx.map(|idx| new_nodes.remove(idx))
}

/// Returns the index of the next candidate to consider.
pub fn pick_index<IT>(order: NewNodeOrder, nodes: IT, cut: f64) -> Option<usize>
where
    IT: IntoIterator<Item = (usize, f64)>,
    IT::IntoIter: Clone,
{
    let nodes = nodes.into_iter().filter(|&(_, b)| b < cut);
    match order {
        NewNodeOrder::Api => {
            if nodes.into_iter().next().is_some() {
                Some(0)
            } else {
                None
            }
        }
        NewNodeOrder::WeightedRandom => choose_cand_weighted(nodes, cut),
        NewNodeOrder::Bound => choose_cand_best(nodes),
        NewNodeOrder::Random => choose_cand_rand(nodes),
    }
}

/// Given a vector of candidate reference, returns the index of the candidate
/// with the minimum bound.
fn choose_cand_best<IT>(nodes: IT) -> Option<usize>
where IT: Iterator<Item = (usize, f64)> {
    nodes.min_by(|x1, x2| cmp_f64(x1.1, x2.1)).map(|x| x.0)
}

/// Given a vector of candidate reference, just choose randomly the next
/// candidate
fn choose_cand_rand<IT>(mut nodes: IT) -> Option<usize>
where IT: Iterator<Item = (usize, f64)> + Clone {
    let len = nodes.clone().count();
    if len == 0 {
        None
    } else {
        nodes.nth(thread_rng().gen_range(0, len)).map(|x| x.0)
    }
}

/// Given a vector of candidate references, returns the index of a weighted
/// sort on the candidate bounds
fn choose_cand_weighted<IT>(nodes: IT, cut: f64) -> Option<usize>
where IT: Iterator<Item = (usize, f64)> + Clone {
    let mut weighted_items = vec![];
    let mut rng = thread_rng();
    let max_bound = nodes
        .clone()
        .max_by(|&x1, &x2| cmp_f64(x1.1, x2.1))
        .map(|x| x.1)?;
    for (ind, x) in nodes {
        if cut.is_infinite() {
            let x_weight = 1 + (10f64 * max_bound / x).floor() as u32;
            weighted_items.push(Weighted {
                weight: x_weight,
                item: ind,
            });
        } else {
            assert!(
                x <= cut,
                "Compare bound fail, cut {:.3e}, cand: {:.3e}",
                cut,
                x
            );
            let weight = (1000f64 * (1f64 - x / cut)).floor() as u32;
            let weight = std::cmp::max(1, weight);
            weighted_items.push(Weighted { weight, item: ind });
        }
    }
    Some(WeightedChoice::new(&mut weighted_items).ind_sample(&mut rng))
}
