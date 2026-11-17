//! Decides whether a GPU performance change is a real win
//!
//! gpu-ab times two frozen bench binaries (control and experiment) in ABBA-ordered pairs
//! in one sitting on one GPU, checks every arm's output against the control's, records
//! GPU state around each arm, and applies a fixed acceptance rule ([`verdict`]) to the
//! per-pair ratios. A/A runs (the control against a copy of itself) measure the
//! paired noise that rule needs
//!
//! The flow is [`plan::prepare`] (freeze binaries, write `plan.json` and the queue job
//! spec), then [`run::execute`] on the GPU host, which writes `result.json` and a
//! `LOG.md` entry

pub mod adapter;
pub mod freeze;
pub mod gpu;
pub mod job;
pub mod plan;
pub mod record;
pub mod run;
pub mod schedule;
pub mod stats;
pub mod time;
pub mod verdict;
