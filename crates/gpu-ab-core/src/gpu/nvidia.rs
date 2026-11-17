//! NVIDIA GPU state through `nvidia-smi`, and host facts from `/proc`

use std::{fs, process::Command};

use eyre::{Result, WrapErr, bail, eyre};

use super::{ComputeApp, Environment, GpuInfo, GpuSnapshot, GpuState};
use crate::time::now_ct;

/// Reads clocks, temperature, power, throttle reasons and compute processes
pub(super) fn snapshot() -> Result<GpuSnapshot> {
    let rows = query(&[
        "--query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,clocks_throttle_reasons.active",
        "--format=csv,noheader,nounits",
    ])?;
    let gpus = rows
        .iter()
        .map(|r| {
            let [index, sm, mem, temp, power, util, throttle] = fields(r)?;
            Ok(GpuState {
                index: index.parse().wrap_err_with(|| format!("GPU index in {r:?}"))?,
                clocks_sm_mhz: num(sm),
                clocks_mem_mhz: num(mem),
                temperature_c: num(temp),
                power_w: num(power),
                utilization_pct: num(util),
                throttle_reasons: throttle.to_owned(),
            })
        })
        .collect::<Result<_>>()?;

    let apps = query(&[
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])?;
    let compute_apps = apps
        .iter()
        .map(|r| {
            // process names are paths and may hold commas, so split around them
            let (pid, rest) = r.split_once(',').ok_or_else(|| eyre!("compute app row {r:?}"))?;
            let (name, mem) = rest.rsplit_once(',').ok_or_else(|| eyre!("compute app row {r:?}"))?;
            Ok(ComputeApp {
                pid: pid.trim().parse().wrap_err_with(|| format!("pid in {r:?}"))?,
                name: name.trim().to_owned(),
                used_memory_mib: num(mem.trim()),
            })
        })
        .collect::<Result<_>>()?;

    Ok(GpuSnapshot {
        at: now_ct(),
        gpus,
        compute_apps,
        loadavg: loadavg(),
        apple: None,
    })
}

/// Reads host, driver, CUDA and GPU facts
pub(super) fn environment() -> Result<Environment> {
    let rows = query(&[
        "--query-gpu=index,name,driver_version,memory.total,clocks.max.sm,power.limit",
        "--format=csv,noheader,nounits",
    ])?;
    let mut driver = String::new();
    let gpus = rows
        .iter()
        .map(|r| {
            let [index, name, drv, mem, clk, power] = fields(r)?;
            driver = drv.to_owned();
            Ok(GpuInfo {
                index: index.parse().wrap_err_with(|| format!("GPU index in {r:?}"))?,
                name: name.to_owned(),
                memory_total_mib: num(mem),
                clocks_max_sm_mhz: num(clk),
                power_limit_w: num(power),
                cores: None,
                pstates_mhz: None,
            })
        })
        .collect::<Result<_>>()?;

    let banner = run_smi(&[])?;
    let cuda = banner
        .split("CUDA Version:")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .map(str::to_owned);

    Ok(Environment {
        host: read_trimmed("/proc/sys/kernel/hostname").unwrap_or_default(),
        kernel: read_trimmed("/proc/sys/kernel/osrelease").unwrap_or_default(),
        cpu: fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_owned())
        }),
        driver,
        cuda,
        os: None,
        gpus,
        loadavg: loadavg(),
        queue_job: std::env::var("HOMEBASED_JOB_ID").ok(),
        queue_resource: std::env::var("HOMEBASED_RESOURCE").ok(),
    })
}

fn run_smi(args: &[&str]) -> Result<String> {
    let out = Command::new("nvidia-smi")
        .args(args)
        .output()
        .wrap_err("running nvidia-smi")?;
    if !out.status.success() {
        bail!(
            "nvidia-smi {} failed ({}): {}",
            args.join(" "),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }

    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn query(args: &[&str]) -> Result<Vec<String>> {
    Ok(run_smi(args)?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect())
}

fn fields<const N: usize>(row: &str) -> Result<[&str; N]> {
    let parts: Vec<&str> = row.split(',').map(str::trim).collect();
    match parts.try_into() {
        Ok(arr) => Ok(arr),
        Err(parts) => bail!("expected {N} fields from nvidia-smi, got {}: {row:?}", parts.len()),
    }
}

/// Parses a number, treating `[N/A]` and friends as missing
fn num(s: &str) -> Option<f64> {
    s.parse().ok()
}

fn loadavg() -> Option<[f64; 3]> {
    let text = fs::read_to_string("/proc/loadavg").ok()?;
    let mut it = text.split_whitespace().map(|v| v.parse::<f64>().ok());
    Some([it.next()??, it.next()??, it.next()??])
}

fn read_trimmed(path: &str) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_owned())
}
