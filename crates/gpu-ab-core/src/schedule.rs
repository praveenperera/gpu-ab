//! Arm order for a paired run: warmup arms, then pairs in ABBA order

use serde::{Deserialize, Serialize};

/// The two arms of a comparison. In an A/A run the experiment is a second copy of the
/// control binary
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Arm {
    Control,
    Experiment,
}

impl Arm {
    pub fn short(self) -> &'static str {
        match self {
            Arm::Control => "ctl",
            Arm::Experiment => "exp",
        }
    }
}

/// One scheduled bench run
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slot {
    /// Position in the run, from 0
    pub index: usize,
    pub arm: Arm,
    /// Pair this slot belongs to; `None` for warmup arms, which aren't counted
    pub pair: Option<usize>,
}

impl Slot {
    /// Directory-friendly label such as `00-w-ctl` or `05-p1-exp`
    pub fn label(&self) -> String {
        match self.pair {
            Some(p) => format!("{:02}-p{p}-{}", self.index, self.arm.short()),
            None => format!("{:02}-w-{}", self.index, self.arm.short()),
        }
    }
}

/// Builds the run order: `warmup` uncounted arms alternating from the control, then
/// `pairs` pairs alternating AB and BA, so drift within a pair cancels across pairs
///
/// The first slot is always a control arm, so the reference fingerprint comes from the
/// control binary
pub fn abba(warmup: usize, pairs: usize) -> Vec<Slot> {
    let warm = (0..warmup).map(|i| (if i % 2 == 0 { Arm::Control } else { Arm::Experiment }, None));
    let paired = (0..pairs).flat_map(|p| {
        let order = if p % 2 == 0 {
            [Arm::Control, Arm::Experiment]
        } else {
            [Arm::Experiment, Arm::Control]
        };
        order.map(|arm| (arm, Some(p)))
    });

    warm.chain(paired)
        .enumerate()
        .map(|(index, (arm, pair))| Slot { index, arm, pair })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(slots: &[Slot]) -> String {
        slots
            .iter()
            .map(|s| match (s.arm, s.pair) {
                (Arm::Control, None) => 'a',
                (Arm::Experiment, None) => 'b',
                (Arm::Control, Some(_)) => 'A',
                (Arm::Experiment, Some(_)) => 'B',
            })
            .collect()
    }

    #[test]
    fn abba_order_with_warmup() {
        let slots = abba(2, 6);
        assert_eq!(order(&slots), "abABBAABBAABBA");
        assert_eq!(slots[0].arm, Arm::Control);
        assert!(slots.iter().enumerate().all(|(i, s)| s.index == i));

        // each pair has one arm of each kind, back to back
        for p in 0..6 {
            let members: Vec<&Slot> = slots.iter().filter(|s| s.pair == Some(p)).collect();
            assert_eq!(members.len(), 2);
            assert_eq!(members[1].index, members[0].index + 1);
            assert_ne!(members[0].arm, members[1].arm);
        }
    }

    #[test]
    fn odd_warmup_and_no_warmup_start_with_control() {
        assert_eq!(order(&abba(3, 2)), "abaABBA");
        assert_eq!(order(&abba(0, 3)), "ABBAAB");
        assert_eq!(abba(0, 1)[0].label(), "00-p0-ctl");
        assert_eq!(abba(1, 1)[2].label(), "02-p0-exp");
    }
}
