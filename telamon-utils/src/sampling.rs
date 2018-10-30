use rand::{
    distributions::{Weighted, WeightedChoice},
    prelude::*,
};
use std::{self, ops::Index};

/// Newtype wrapper around a probability value.  This represents a floating point number in [0, 1].
/// A probability can safely be converted to a floating point number, but there is no `From<f64>`
/// implementation for `Probability` since that conversion can fail if the floating point is not in
/// the [0, 1] range.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Probability(f64);

impl Probability {
    /// Create a new probability from a number in [0, 1].
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in the [0, 1] range.
    pub fn new(p: f64) -> Self {
        assert!(
            0f64 <= p && p <= 1f64,
            "probability must be in [0, 1] range"
        );

        Probability(p)
    }

    /// Converts back a probability to a f64 number.
    pub fn into_f64(self) -> f64 {
        self.into()
    }
}

impl Into<f64> for Probability {
    fn into(self) -> f64 {
        self.0
    }
}

/// Represents a value which was sampled with a given probability.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sampled<T> {
    /// The sampled value.
    pub value: T,

    /// The prior probability to select this value when sampling.
    pub probability: Probability,
}

impl<T> Sampled<T> {
    pub fn as_ref(&self) -> Sampled<&T> {
        Sampled {
            value: &self.value,
            probability: self.probability,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WeightedSampler<T> {
    sampled: Vec<Sampled<T>>,
}

impl<T> WeightedSampler<T> {
    /// Create a new weighted sampler from an iterator of weighted values.
    pub fn new<I>(mut weight_fn: impl FnMut(&T) -> f64, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let (weights, items): (Vec<_>, Vec<_>) = iter
            .into_iter()
            .map(|item| (weight_fn(&item), item))
            .unzip();

        // Use log sum exp trick for better accuracy when computing the total weight.  This is needed
        // when the weights are very large and adding them loses precision.
        let log_weights = weights.into_iter().map(f64::ln).collect::<Vec<_>>();
        let max_log_weight = log_weights
            .iter()
            .cloned()
            .fold(std::f64::NEG_INFINITY, f64::max);
        let log_total_weight = max_log_weight + log_weights
            .iter()
            .cloned()
            .map(|log_weight| (log_weight - max_log_weight).exp())
            .sum::<f64>()
            .ln();

        WeightedSampler {
            sampled: items
                .into_iter()
                .zip(
                    log_weights.into_iter().map(|log_weight| {
                        Probability((log_weight - log_total_weight).exp())
                    }),
                ).map(|(value, probability)| Sampled { value, probability })
                .collect(),
        }
    }

    /// Returns an iterator over the sampled values.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> {
        self.sampled.iter().map(|sampled| &sampled.value)
    }

    pub fn probabilities(&self) -> impl ExactSizeIterator<Item = Probability> + '_ {
        self.sampled.iter().map(|sampled| sampled.probability)
    }

    pub fn len(&self) -> usize {
        self.sampled.len()
    }

    /// Sample one value from the vector
    pub fn sample(&self, epsilon: f64, rng: &mut impl Rng) -> Option<Sampled<&T>> {
        self.filter_sample(|_| true, epsilon, rng)
    }

    /// # Arguments
    ///
    /// * predicate - The
    /// * epsilon - The
    /// * rng - The
    pub fn filter_sample(
        &self,
        mut predicate: impl FnMut(&T) -> bool,
        epsilon: f64,
        rng: &mut impl Rng,
    ) -> Option<Sampled<&T>> {
        assert!(
            0.0 <= epsilon && epsilon <= 1.0,
            "epsilon must be between 0 and 1; got {}",
            epsilon
        );

        let (items, probabilities): (Vec<_>, Vec<_>) = self
            .sampled
            .iter()
            .filter_map(move |sampled| {
                if sampled.probability.0 > 0.0 && predicate(&sampled.value) {
                    Some((&sampled.value, sampled.probability.0))
                } else {
                    None
                }
            }).unzip();

        // We need to renormalize the probabilities for the filtered subset.  If the filtered
        // subset has probability zero, we have nothing to sample from.
        let total_probability = probabilities.iter().cloned().sum::<f64>();

        if total_probability == 0.0 {
            return None;
        }

        let len = probabilities.len();
        assert!(
            len <= u32::max_value() as usize,
            "number of non-zero weights cannot exceed {}",
            u32::max_value()
        );

        let sample_epsilon = epsilon / len as f64;
        let resolution = (u32::max_value() / len as u32) as f64;

        let Sampled { value, probability } = WeightedChoice::new(
            &mut probabilities
                .into_iter()
                .map(move |probability| {
                    (probability / total_probability) * (1f64 - epsilon) + sample_epsilon
                }).enumerate()
                .map(|(index, probability)| Weighted {
                    item: Sampled {
                        value: index,
                        probability: Probability(probability),
                    },
                    weight: (probability * resolution) as u32,
                }).collect::<Vec<_>>(),
        ).sample(rng);

        Some(Sampled {
            value: &items[value],
            probability,
        })
    }
}

impl<T> Index<usize> for WeightedSampler<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.sampled[index].value
    }
}

#[cfg(test)]
mod tests {
    use rand::{self, prelude::*};

    use super::{Probability, WeightedSampler};

    fn is_close(x: f64, p: f64, tol: f64) -> bool {
        (x - p).abs() < tol
    }

    fn approx_eq(lhs: f64, rhs: f64) -> bool {
        (lhs - rhs).abs() < ::std::f64::EPSILON
    }

    #[test]
    fn uniform() {
        let mut rng = rand::prng::XorShiftRng::from_seed(Default::default());
        let sampler = WeightedSampler::new(|_| 1., [0, 1, 2, 3, 4].into_iter());

        let expected = Probability(1. / 5.);

        assert_eq!(sampler.sample(0.0, &mut rng).unwrap().probability, expected);
        assert_eq!(sampler.sample(0.5, &mut rng).unwrap().probability, expected);
        assert_eq!(sampler.sample(1.0, &mut rng).unwrap().probability, expected);
    }

    fn bernoulli(p: f64) {
        let mut rng = rand::prng::XorShiftRng::from_seed(Default::default());
        let sampler = WeightedSampler::new(
            |b| if *b { p } else { 1. - p },
            vec![true, false].into_iter(),
        );

        assert!(is_close(
            sampler.probabilities().map(|p| p.0).sum::<f64>(),
            1.,
            1e-15
        ));
        assert!(
            sampler
                .probabilities()
                .zip(vec![p, 1. - p].into_iter())
                .all(|(lhs, rhs)| approx_eq(lhs.0, rhs))
        );

        let total = 1_000;
        let (mut num_true, mut num_false) = (0, 0);
        for _ in 0..total {
            let sampled = sampler.sample(0.0, &mut rng).unwrap();
            if *sampled.value {
                assert!(approx_eq(sampled.probability.0, p));
                num_true += 1
            } else {
                assert!(approx_eq(sampled.probability.0, 1. - p));;;
                num_false += 1
            }
        }

        let tol = f64::max(p, 1. - p) / (total as f64).sqrt();

        assert!(is_close(num_true as f64 / total as f64, p, tol));
        assert!(is_close(num_false as f64 / total as f64, 1. - p, tol));
    }

    fn bernoulli_filter(p: f64) {
        let mut rng = rand::prng::XorShiftRng::from_seed(Default::default());
        let sampler = WeightedSampler::new(
            |b| {
                b.as_ref()
                    .map(|b| if *b { p } else { 1. - p })
                    .unwrap_or(7.)
            },
            vec![Some(true), None, Some(false), None].into_iter(),
        );

        assert!(is_close(
            sampler.probabilities().map(|p| p.0).sum::<f64>(),
            1.,
            1e-15
        ));

        let probabilities = sampler.probabilities().collect::<Vec<_>>();
        assert!(approx_eq(probabilities[1].0, 7. / 15.));
        assert!(approx_eq(probabilities[3].0, 7. / 15.));

        let total = 1_000;
        let (mut num_true, mut num_false) = (0, 0);
        for _ in 0..total {
            let sampled = sampler
                .filter_sample(Option::is_some, 0.0, &mut rng)
                .unwrap();

            if sampled.value.unwrap() {
                assert!(approx_eq(sampled.probability.0, p));
                num_true += 1
            } else {
                assert!(approx_eq(sampled.probability.0, 1. - p));
                num_false += 1
            }
        }

        let tol = f64::max(p, 1. - p) / (total as f64).sqrt();

        assert!(is_close(num_true as f64 / total as f64, p, tol));
        assert!(is_close(num_false as f64 / total as f64, 1. - p, tol));
    }

    macro_rules! bernoulli_tests {
        ($($name:ident: $bernoulli:ident($value:expr),)*) => {
            $(
                #[test]
                fn $name() {
                    $bernoulli($value)
                }
            )*
        };
    }

    bernoulli_tests! {
        bernoulli_0_10: bernoulli(0.10),
        bernoulli_0_25: bernoulli(0.25),
        bernoulli_0_50: bernoulli(0.50),
        bernoulli_0_75: bernoulli(0.75),
        bernoulli_0_90: bernoulli(0.90),

        bernoulli_filter_0_10: bernoulli_filter(0.10),
        bernoulli_filter_0_25: bernoulli_filter(0.25),
        bernoulli_filter_0_50: bernoulli_filter(0.50),
        bernoulli_filter_0_75: bernoulli_filter(0.75),
        bernoulli_filter_0_90: bernoulli_filter(0.90),
    }
}
