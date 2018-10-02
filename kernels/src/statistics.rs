//! Statistical analysis for benchmarking.
use std;
use utils::*;

/// Estimates a mean within a confidence interval.
pub struct Estimate {
    pub unit: &'static str,
    pub value: f64,
    pub interval: f64,
    pub confidence: f64,
}

impl std::fmt::Display for Estimate {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{value:.2e}{unit} (+/-{interval:.2e}{unit} @ {confidence:.0}%)",
            value = self.value,
            unit = self.unit,
            interval = self.interval,
            confidence = self.confidence * 100.
        )
    }
}

/// Computes the mean of a data set.
pub fn mean(data: &[f64]) -> f64 { data.iter().cloned().sum::<f64>() / data.len() as f64 }

/// Computes the mean and the confidence interval of the data set. The
/// requested degree of confidence must be between 0 and 1.
pub fn estimate_mean(
    mut data: Vec<f64>,
    confidence: f64,
    unit: &'static str,
) -> Estimate
{
    assert!(0. <= confidence && confidence <= 1.);
    let mean = mean(&data);
    for item in &mut data {
        *item = (*item - mean).abs();
    }
    data.sort_by(|&x, &y| cmp_f64(x, y));
    let idx = std::cmp::min((data.len() as f64 * confidence).ceil() as usize, data.len());
    Estimate {
        value: mean,
        unit,
        interval: data[idx],
        confidence,
    }
}

/// Computes the error margin of a ratio between answer of a binary choice
/// given the number of samples with a 95% confidence interval.
pub fn estimate_ratio(ratio: f64, num_samples: usize) -> Estimate {
    let z = 1.96; // From a table, with a confidence interval of 95%.
    let interval = z * (ratio * (1. - ratio) / num_samples as f64).sqrt();
    Estimate {
        value: ratio,
        unit: "",
        interval,
        confidence: 0.95,
    }
}
