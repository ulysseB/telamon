//! Provides different methods to select a candidate in a list.
use device::Context;
use explorer::candidate::Candidate;
use explorer::choice;
use explorer::config::{ChoiceOrdering, NewNodeOrder};
use rand::distributions::{Weighted, WeightedChoice, WeightedIndex};
use rand::prelude::*;
use std;
use utils::*;

/// A recursive function that takes a candidate and expands it until we have a completely specified
/// candidate that we can pass to the evaluator, or we find a dead-end
pub fn descend<'a>(
    choice_order: &ChoiceOrdering,
    node_order: NewNodeOrder,
    context: &Context,
    candidate: Candidate<'a>,
    cut: f64,
) -> Option<Candidate<'a>> {
    let choice_opt = choice::list(choice_order, &candidate.space).next();
    if let Some(choice) = choice_opt {
        let mut new_nodes = candidate.apply_choice(context, choice);
        let idx = node_order.pick_candidate(&new_nodes, cut)?;
        let node = new_nodes.swap_remove(idx);
        descend(choice_order, node_order, context, node, cut)
    } else {
        Some(candidate)
    }
}

fn softmax(values: &[f64]) -> Vec<f64> {
    // normalize before computing softmax
    let fmin = values.iter().min_by(|x, y| cmp_f64(**x, **y)).unwrap();
    let div = values.iter().map(|x| (x - fmin).exp()).sum::<f64>();
    values.iter().map(|x| (x - fmin).exp() / div).collect()
}

pub fn rave<'a>(
    choice_order: &ChoiceOrdering,
    node_order: NewNodeOrder,
    context: &Context,
    candidate: Candidate<'a>,
    cut: f64,
    amaf: &Fn(&choice::ActionEx) -> Option<f64>,
) -> Option<Candidate<'a>> {
    let choice = choice::list(choice_order, &candidate.space).next();
    if let Some(choice) = choice {
        // Compute the AMAFs once at the beginning to avoid races.
        let amafs = choice.iter().map(amaf).collect::<Vec<_>>();

        // If all the decisions have a prior value, use it
        if amafs.iter().all(|amaf| amaf.is_some()) {
            let mut choice = choice;
            let mut rng = thread_rng();
            let mut probs = softmax(
                &amafs
                    .into_iter()
                    .map(|amaf| amaf.unwrap())
                    .collect::<Vec<_>>()[..],
            );

            // Pick decisions according to the probabilities.  If this results in an invalid
            // candidate, try again until all choices have been exhausted.
            while probs.iter().sum::<f64>() > 0. {
                let dist = WeightedIndex::new(&probs).unwrap();
                let index = dist.sample(&mut rng);
                if let Ok(child) =
                    candidate.apply_decision(context, choice[index].clone())
                {
                    if child.bound.value() >= cut {
                        probs[index] = 0.0;
                    } else {
                        return rave(choice_order, node_order, context, child, cut, amaf);
                    }
                } else {
                    probs[index] = 0.0;
                }
            }

            None
        } else {
            // Otherwise, pick one at random which doesn't have a prior value yet.  Actually, don't
            // do this, because it can lock the search in an invalid portion of the space.
            let actions = choice;
            /*
            .into_iter()
            .zip(amafs.into_iter())
            .filter_map(
                |(action, amaf)| {
                    if amaf.is_none() {
                        Some(action)
                    } else {
                        None
                    }
                },
            )
            .collect::<Vec<_>>();*/

            let node = {
                let mut new_nodes = candidate.apply_choice(context, actions);
                let idx = node_order.pick_candidate(&new_nodes, cut)?;
                new_nodes.swap_remove(idx)
            };
            rave(choice_order, node_order, context, node, cut, amaf)
        }
    } else {
        Some(candidate)
    }
}

impl NewNodeOrder {
    /// Called in montecarlo_descend, dispatch the choice of the next candidate according to our
    /// configuration
    pub fn pick_candidate<'a>(
        &self,
        new_nodes: &[Candidate<'a>],
        cut: f64,
    ) -> Option<usize> {
        let items = new_nodes.iter().map(|c| c.bound.value()).enumerate();
        self.pick_index(items, cut)
    }

    /// Returns the index of the next candidate to consider.
    pub fn pick_index<IT>(&self, nodes: IT, cut: f64) -> Option<usize>
    where
        IT: Iterator<Item = (usize, f64)> + Clone,
    {
        let nodes = nodes.filter(|&(_, b)| b < cut);
        match self {
            NewNodeOrder::Api => nodes.into_iter().next().map(|(idx, _)| idx),
            NewNodeOrder::WeightedRandom => choose_cand_weighted(nodes, cut),
            NewNodeOrder::Bound => choose_cand_best(nodes),
            NewNodeOrder::Random => choose_cand_rand(nodes),
        }
    }
}

/// Given a vector of candidate reference, returns the index of the candidate with the minimum
/// bound.
fn choose_cand_best<IT>(nodes: IT) -> Option<usize>
where
    IT: Iterator<Item = (usize, f64)>,
{
    nodes.min_by(|x1, x2| cmp_f64(x1.1, x2.1)).map(|x| x.0)
}

/// Given a vector of candidate reference, just choose randomly the next candidate
fn choose_cand_rand<IT>(mut nodes: IT) -> Option<usize>
where
    IT: Iterator<Item = (usize, f64)> + Clone,
{
    let len = nodes.clone().count();
    if len == 0 {
        None
    } else {
        nodes.nth(thread_rng().gen_range(0, len)).map(|x| x.0)
    }
}

/// Given a vector of candidate references, returns the index of a weighted sort on the candidate
/// bounds
fn choose_cand_weighted<IT>(nodes: IT, cut: f64) -> Option<usize>
where
    IT: Iterator<Item = (usize, f64)> + Clone,
{
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
                x < cut,
                "Compare bound fail, cut {:.3e}, cand: {:.3e}",
                cut,
                x
            );
            let weight = (1000f64 * (1f64 - x / cut)).floor() as u32;
            let weight = std::cmp::max(1, weight);
            weighted_items.push(Weighted { weight, item: ind });
        }
    }
    Some(WeightedChoice::new(&mut weighted_items).sample(&mut rng))
}
