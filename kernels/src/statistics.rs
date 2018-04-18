//! Statistical analysis for benchmarking.
use std;
use utils::*;

/// Computes the mean of a data set.
pub fn mean(data: &[f64]) -> f64 {
    data.iter().cloned().sum::<f64>()/data.len() as f64
}

/// Computes the mean and the confidence interval of the data set. The requested degree
/// of confidence must be between 0 and 1.
pub fn estimate_mean(mut data: Vec<f64>, confidence: f64) -> MeanEstimate {
    assert!(0. <= confidence && confidence <= 1.);
    let mean = mean(&data);
    for item in &mut data {
        *item = (*item-mean).abs();
    }
    data.sort_by(|&x, &y| cmp_f64(x, y));
    let idx = std::cmp::min((data.len() as f64 * confidence).ceil() as usize, data.len());
    MeanEstimate { mean, interval: data[idx], confidence }
}

/// Estimates a mean within a confidence interval.
pub struct MeanEstimate {
    pub mean: f64,
    pub interval: f64,
    pub confidence: f64,
}

impl std::fmt::Display for MeanEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:.2e} (+/-{:.2e} @ {:.0}%)",
               self.mean, self.interval, self.confidence*100.0)
    }
}
