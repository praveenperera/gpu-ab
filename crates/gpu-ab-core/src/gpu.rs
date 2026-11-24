//! GPU state and host environment around each arm
//!
//! On Linux they come from `nvidia-smi` and `/proc` ([`nvidia`]); on macOS from `ioreg`,
//! IOReport and the thermal and power settings ([`apple`]). The probe is chosen at build
//! time. `nvidia-smi` and `ioreg` are looked up on `PATH`, which is how tests substitute
//! fake ones

use serde::{Deserialize, Serialize};

pub mod apple;
#[cfg(target_os = "macos")]
mod ioreport;
#[cfg(not(target_os = "macos"))]
mod nvidia;

pub use apple::{AppleArm, AppleState, BenchSampler};

/// Throttle-reason bits that mean the GPU slowed itself down: hardware slowdown, software
/// and hardware thermal slowdown, and the hardware power brake
pub const SLOWDOWN_BITS: u64 = 0x8 | 0x20 | 0x40 | 0x80;

/// One reading of every GPU and of the compute processes on them
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuSnapshot {
    pub at: String,
    /// NVIDIA GPUs; empty on Apple silicon, whose state is in `apple`
    pub gpus: Vec<GpuState>,
    /// CUDA processes on any GPU; any entry makes the arm invalid
    pub compute_apps: Vec<ComputeApp>,
    /// 1, 5 and 15 minute host load averages
    pub loadavg: Option<[f64; 3]>,
    /// Apple silicon GPU state; absent on NVIDIA hosts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apple: Option<AppleState>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuState {
    pub index: u32,
    pub clocks_sm_mhz: Option<f64>,
    pub clocks_mem_mhz: Option<f64>,
    pub temperature_c: Option<f64>,
    pub power_w: Option<f64>,
    pub utilization_pct: Option<f64>,
    /// `clocks_throttle_reasons.active` bitmask as printed, such as `0x0000000000000004`
    pub throttle_reasons: String,
}

impl GpuState {
    /// Whether a thermal or hardware slowdown bit was set
    pub fn slowed_down(&self) -> bool {
        let hex = self.throttle_reasons.trim_start_matches("0x");
        u64::from_str_radix(hex, 16).is_ok_and(|bits| bits & SLOWDOWN_BITS != 0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeApp {
    pub pid: u32,
    pub name: String,
    pub used_memory_mib: Option<f64>,
}

/// Host, driver and GPU facts recorded once per run
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Environment {
    pub host: String,
    pub kernel: String,
    pub cpu: Option<String>,
    /// NVIDIA driver version, or the AGX GPU driver's version on Apple silicon
    pub driver: String,
    /// Highest CUDA version the driver supports, from the `nvidia-smi` banner
    pub cuda: Option<String>,
    /// macOS version and build, such as `macOS 27.0.1 (26A434)`; absent on Linux
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
    pub gpus: Vec<GpuInfo>,
    pub loadavg: Option<[f64; 3]>,
    /// Queue job and resource from the homebased job environment, when run in one
    pub queue_job: Option<String>,
    pub queue_resource: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    /// Unified memory on Apple silicon
    pub memory_total_mib: Option<f64>,
    pub clocks_max_sm_mhz: Option<f64>,
    pub power_limit_w: Option<f64>,
    /// GPU core count on Apple silicon
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cores: Option<u32>,
    /// Frequency of each GPU performance state on Apple silicon, the off state first, in
    /// the order IOReport reports their residency
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pstates_mhz: Option<Vec<f64>>,
}

/// Reads the state recorded before and after each arm: on Linux clocks, temperature,
/// power, throttle reasons and compute processes; on macOS every Metal client's GPU time,
/// GPU performance-state residency and energy, thermal pressure and power mode
pub fn snapshot() -> eyre::Result<GpuSnapshot> {
    #[cfg(target_os = "macos")]
    return apple::snapshot();
    #[cfg(not(target_os = "macos"))]
    return nvidia::snapshot();
}

/// Reads host, driver and GPU facts
pub fn environment() -> eyre::Result<Environment> {
    #[cfg(target_os = "macos")]
    return apple::environment();
    #[cfg(not(target_os = "macos"))]
    return nvidia::environment();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slowdown_bits() {
        let state = |t: &str| GpuState {
            index: 0,
            clocks_sm_mhz: None,
            clocks_mem_mhz: None,
            temperature_c: None,
            power_w: None,
            utilization_pct: None,
            throttle_reasons: t.to_owned(),
        };
        assert!(
            !state("0x0000000000000004").slowed_down(),
            "power cap is normal under load"
        );
        assert!(!state("0x0000000000000001").slowed_down());
        assert!(state("0x0000000000000040").slowed_down());
        assert!(!state("[N/A]").slowed_down());
    }
}
