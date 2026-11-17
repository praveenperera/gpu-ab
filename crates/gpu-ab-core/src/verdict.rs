//! The acceptance rule, as the README's "The verdict" documents it
//!
//! Ratios are speedups, above 1 when the experiment wins. With `bar = 1 + 2 * noise`:
//!
//! - ACCEPT: the 95% interval's lower end is above 1 and the mean is at least `bar`
//! - REJECT: the interval's upper end is below `bar`, so the data rule out a gain that
//!   clears it (a significant slowdown is one case)
//! - INCONCLUSIVE: anything else, including too few valid pairs or no A/A noise figure

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::stats::PairStats;

/// Fewest valid pairs a decision may rest on
pub const MIN_PAIRS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Decision {
    Accept,
    Reject,
    Inconclusive,
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Decision::Accept => "ACCEPT",
            Decision::Reject => "REJECT",
            Decision::Inconclusive => "INCONCLUSIVE",
        })
    }
}

/// A decision and the reason for it
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    pub decision: Decision,
    pub reason: String,
    /// `1 + 2 * noise`, when a noise figure was available
    pub bar: Option<f64>,
}

/// Applies the rule to the valid pairs' statistics and an A/A noise figure (the standard
/// deviation of A/A per-pair ratios, as a fraction)
pub fn decide(stats: Option<&PairStats>, noise: Option<f64>) -> Verdict {
    let bar = noise.map(|n| 1.0 + 2.0 * n);
    let verdict = |decision, reason: String| Verdict { decision, reason, bar };

    let Some(s) = stats else {
        return verdict(Decision::Inconclusive, "fewer than 2 valid pairs".to_owned());
    };

    if s.n < MIN_PAIRS {
        return verdict(
            Decision::Inconclusive,
            format!("only {} valid pairs, need {MIN_PAIRS}", s.n),
        );
    }

    let ci = format!("95% CI x{:.4}-{:.4}", s.ci_low, s.ci_high);
    let Some(bar) = bar else {
        if s.ci_high < 1.0 {
            return verdict(Decision::Reject, format!("{ci} is entirely below x1"));
        }

        return verdict(
            Decision::Inconclusive,
            format!("{ci}, but no A/A noise figure to set the bar"),
        );
    };

    if s.ci_low > 1.0 && s.mean >= bar {
        verdict(
            Decision::Accept,
            format!("{ci} excludes x1 and mean x{:.4} clears the bar x{bar:.4}", s.mean),
        )
    } else if s.ci_high < bar {
        verdict(
            Decision::Reject,
            format!("{ci} lies below the bar x{bar:.4} (2x A/A noise)"),
        )
    } else if s.ci_low <= 1.0 {
        verdict(
            Decision::Inconclusive,
            format!("{ci} includes x1 and reaches the bar x{bar:.4}"),
        )
    } else {
        verdict(
            Decision::Inconclusive,
            format!("{ci} excludes x1, but mean x{:.4} is under the bar x{bar:.4}", s.mean),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::pair_stats;

    fn run(ratios: &[f64], noise: Option<f64>) -> Decision {
        decide(pair_stats(ratios).as_ref(), noise).decision
    }

    #[test]
    fn clear_win_is_accepted() {
        let r = [1.020, 1.024, 1.018, 1.022, 1.021, 1.019];
        assert_eq!(run(&r, Some(0.002)), Decision::Accept);
    }

    #[test]
    fn significant_but_under_twice_noise_is_not_accepted() {
        // tight interval around x1.006, above 1 but wholly under the bar x1.010
        let tight = [1.006, 1.0065, 1.0055, 1.006, 1.0062, 1.0058];
        assert_eq!(run(&tight, Some(0.005)), Decision::Reject);

        // interval x1.0043-1.0118 excludes 1 and reaches the bar, but the mean doesn't
        let r = [1.004, 1.012, 1.008, 1.004, 1.012, 1.008];
        let v = decide(pair_stats(&r).as_ref(), Some(0.005));
        assert_eq!(v.decision, Decision::Inconclusive);
        assert!(v.reason.contains("under the bar"), "{}", v.reason);
    }

    #[test]
    fn noise_level_wobble_is_rejected_and_wide_interval_is_inconclusive() {
        let aa = [1.001, 0.999, 1.002, 0.998, 1.000, 1.001];
        assert_eq!(run(&aa, Some(0.002)), Decision::Reject);
        let wide = [1.08, 0.95, 1.10, 0.97, 1.06, 0.99];
        assert_eq!(run(&wide, Some(0.002)), Decision::Inconclusive);
    }

    #[test]
    fn slowdown_is_rejected_without_noise() {
        let r = [0.97, 0.975, 0.968, 0.972, 0.971, 0.969];
        assert_eq!(run(&r, None), Decision::Reject);
        let win = [1.020, 1.024, 1.018, 1.022, 1.021, 1.019];
        assert_eq!(run(&win, None), Decision::Inconclusive);
    }

    #[test]
    fn too_few_pairs_is_inconclusive() {
        let r = [1.05, 1.05, 1.05, 1.05, 1.05];
        assert_eq!(run(&r, Some(0.001)), Decision::Inconclusive);
        assert_eq!(run(&[1.05], Some(0.001)), Decision::Inconclusive);
    }
}
