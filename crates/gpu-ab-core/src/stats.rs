//! Statistics over per-pair ratios

use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Summary of per-pair ratios with a two-sided 95% t-interval for their mean
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PairStats {
    pub n: usize,
    pub mean: f64,
    /// Sample standard deviation (n - 1); from an A/A run this is the paired noise
    pub sd: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    /// The t quantile used, for df = n - 1
    pub t: f64,
    pub min: f64,
    pub max: f64,
}

/// Mean, spread and 95% t-interval of the ratios; `None` for fewer than two
pub fn pair_stats(ratios: &[f64]) -> Option<PairStats> {
    let n = ratios.len();
    if n < 2 {
        return None;
    }

    let nf = n as f64;
    let mean = ratios.iter().sum::<f64>() / nf;
    let var = ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    let sd = var.sqrt();
    let t = t_975(n - 1);
    let half = t * sd / nf.sqrt();

    Some(PairStats {
        n,
        mean,
        sd,
        ci_low: mean - half,
        ci_high: mean + half,
        t,
        min: ratios.iter().copied().fold(f64::INFINITY, f64::min),
        max: ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

/// 97.5th percentile of Student's t with `df` degrees of freedom
pub fn t_975(df: usize) -> f64 {
    StudentsT::new(0.0, 1.0, df as f64)
        .expect("df >= 1 gives a valid distribution")
        .inverse_cdf(0.975)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_quantiles_match_tables() {
        for (df, want) in [(1, 12.706), (5, 2.571), (7, 2.365), (11, 2.201), (30, 2.042)] {
            assert!((t_975(df) - want).abs() < 1e-3, "df {df}: {} vs {want}", t_975(df));
        }
    }

    #[test]
    fn interval_matches_hand_computation() {
        // mean 1.012, sd 0.004 over 6 pairs: half width 2.5706 * 0.004 / sqrt(6) = 0.004198
        let r = [1.008, 1.016, 1.012, 1.008, 1.016, 1.012];
        let s = pair_stats(&r).unwrap();
        assert_eq!(s.n, 6);
        assert!((s.mean - 1.012).abs() < 1e-12);
        assert!((s.sd - 0.0035777).abs() < 1e-6);
        let half = 2.570_582 * s.sd / 6f64.sqrt();
        assert!((s.ci_low - (1.012 - half)).abs() < 1e-6);
        assert!((s.ci_high - (1.012 + half)).abs() < 1e-6);
        assert_eq!((s.min, s.max), (1.008, 1.016));
    }

    #[test]
    fn needs_two_pairs() {
        assert!(pair_stats(&[1.0]).is_none());
        let s = pair_stats(&[1.0, 1.0]).unwrap();
        assert_eq!((s.sd, s.ci_low, s.ci_high), (0.0, 1.0, 1.0));
    }
}
