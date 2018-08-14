//! 1D linear regression.
use std::fmt;

/// A linear regression predictor.
#[derive(Clone, Debug)]
pub struct LinearRegression {
    pub slope: f64,
    pub offset: f64,
    pub error_r2: f64,
}

impl LinearRegression {
    /// Train a linear regression using the least square error.
    pub fn train(x: &[f64], y: &[f64]) -> LinearRegression {
        assert_eq!(x.len(), y.len());
        let x_mean = mean(x);
        let y_mean = mean(y);
        let mut a = 0f64;
        let mut b = 0f64;
        for (x, y) in x.iter().zip(y) {
            a += (x - x_mean) * (y - y_mean);
            b += (x - x_mean).powi(2);
        }
        let slope = a/b;
        let offset = y_mean - slope * x_mean;
        let mut ss_reg = 0f64;
        let mut ss_tot = 0f64;
        for (x, y) in x.iter().zip(y) {
            ss_reg += (x * slope + offset - y).powi(2);
            ss_tot += (y_mean - y).powi(2);
        };
        LinearRegression { slope, offset, error_r2: 1.0 - ss_reg/ss_tot }
    }
}

impl fmt::Display for LinearRegression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "y = {:.4e} * x + {:.2}, r2 = {:.8}",
               self.slope, self.offset, self.error_r2)
    }
}

/// Computes the mean value of a slice.
pub fn mean(x: &[f64]) -> f64 { x.iter().sum::<f64>()/(x.len() as f64) }
