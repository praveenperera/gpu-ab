//! Apple silicon GPU state, read without root
//!
//! - `ioreg`: every Metal client's cumulative GPU time (`AppUsage` `accumulatedGPUTime`,
//!   in ns, summed over a process's command queues), the instantaneous
//!   `Device Utilization %`, and the GPU model, core count, driver and P-state table
//! - IOReport: cumulative GPU performance-state residency and GPU and Neural Engine energy
//! - `notifyutil`: the system thermal pressure level; `pmset -g`: the power mode
//!
//! Other projects' Core ML jobs, iOS simulators and browsers use the GPU outside the
//! queue, and CPU load doesn't show them. An arm is invalid when processes other than
//! the bench used the GPU for at least [`OTHER_GPU_MAX_SHARE`] of the time between the
//! `ioreg` readings around it, from the difference in their GPU time. WindowServer is
//! recorded but exempt: it composites the display all the time, and the apps that make
//! it work hard have GPU clients of their own
//!
//! The check guards GPU arms only. A [`BenchSampler`] reads the bench's own clients while
//! the arm runs, since they close when the bench exits, before the second reading; an arm
//! whose bench used under [`OWN_GPU_IDLE_SHARE`] of the GPU can't be slowed by other GPU
//! users, so their use is recorded but doesn't invalidate it

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::mpsc,
    thread::JoinHandle,
};

use serde::{Deserialize, Serialize};

/// Largest share of an arm that other processes together may spend on the GPU before the
/// arm is invalid
pub const OTHER_GPU_MAX_SHARE: f64 = 0.01;

/// Largest share of an arm the bench's own processes may spend on the GPU while the arm
/// still counts as using none, which exempts it from [`OTHER_GPU_MAX_SHARE`]
pub const OWN_GPU_IDLE_SHARE: f64 = 0.001;

/// The display compositor, recorded separately and exempt from [`OTHER_GPU_MAX_SHARE`]
pub const WINDOW_SERVER: &str = "WindowServer";

/// The bench's own Metal clients, read while its arm ran
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BenchClients {
    /// The bench and each of its descendants seen with a Metal client
    pub pids: BTreeSet<u32>,
    /// Latest GPU time read for each of their clients, by registry id
    pub gpu_time_ns: BTreeMap<u64, u64>,
    /// Whether the clients were read at least once during the arm; without a reading the
    /// bench's own use is unknown, and the arm is checked as a GPU arm
    pub sampled: bool,
}

/// Reads the bench's Metal clients from `ioreg` while its arm runs: after 100 ms, then at
/// doubling intervals up to once a second. Each reading takes about 20 ms of one core, the
/// same in both arms
///
/// A client closes when its process exits, which is before the reading after the arm, so
/// only readings taken during the arm see the bench's own GPU time. A reading sees only
/// the clients open at that moment, so GPU work done entirely by child processes that start
/// and exit between two readings is missed, and the arm counts as using no GPU; a bench
/// whose GPU process lives through most of the arm is always seen
pub struct BenchSampler {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<BenchClients>>,
}

impl BenchSampler {
    /// Starts reading the clients of process `bench_pid` and its descendants; on NVIDIA
    /// hosts it reads nothing
    pub fn start(bench_pid: u32) -> Self {
        let (stop, stopped) = mpsc::channel();
        #[cfg(target_os = "macos")]
        let thread = Some(std::thread::spawn(move || probe::watch_bench(bench_pid, &stopped)));
        #[cfg(not(target_os = "macos"))]
        let thread = {
            let _ = (bench_pid, stopped);
            None
        };

        Self {
            stop: Some(stop),
            thread,
        }
    }

    /// Stops reading and returns what was read
    pub fn finish(mut self) -> BenchClients {
        // closing the channel wakes the thread, which returns after any reading in flight
        self.stop.take();
        self.thread.take().and_then(|t| t.join().ok()).unwrap_or_default()
    }
}

/// Apple GPU readings at one instant
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppleState {
    /// `Device Utilization %` from the accelerator's `PerformanceStatistics`
    pub device_utilization_pct: Option<f64>,
    /// Every Metal client with GPU time so far
    pub clients: Vec<GpuClient>,
    /// When `clients` was read, in ns of `CLOCK_UPTIME_RAW`
    #[serde(default)]
    pub clients_read_ns: Option<u64>,
    /// `nominal`, `moderate`, `heavy`, `trapping` or `sleeping`
    pub thermal_pressure: Option<String>,
    /// `automatic`, `low power` or `high power`
    pub power_mode: Option<String>,
    /// Cumulative residency of each GPU performance state, the off state first, in
    /// IOReport ticks
    pub gpu_residency: Option<Vec<i64>>,
    /// Cumulative GPU energy, in joules
    pub gpu_energy_j: Option<f64>,
    /// Cumulative Neural Engine energy, in joules; it updates only every few seconds
    pub ane_energy_j: Option<f64>,
    /// When the IOReport counters were read, in ns of `CLOCK_UPTIME_RAW`
    #[serde(default)]
    pub ioreport_read_ns: Option<u64>,
}

/// One Metal client's cumulative GPU time, summed over its command queues; a process can
/// have several clients
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuClient {
    /// IORegistry entry id, which a new client never reuses, unlike a pid
    pub id: u64,
    pub pid: u32,
    /// Process name as ioreg prints it, cut to 16 characters
    pub name: String,
    pub gpu_time_ns: u64,
    /// Latest `lastSubmittedTime` over its command queues, in ns of `CLOCK_UPTIME_RAW`
    #[serde(default)]
    pub last_submitted_ns: u64,
}

/// What happened on an Apple GPU during one arm, from the snapshots around it
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppleArm {
    /// GPU time of every process other than the bench and WindowServer, as a share of the
    /// time between the `ioreg` readings around the arm
    pub other_share: f64,
    /// Those processes, busiest first
    pub others: Vec<ClientUse>,
    pub window_server_share: f64,
    /// The bench's own GPU time, read during the arm, as a share of the same time; absent
    /// when it wasn't read
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub own_share: Option<f64>,
    /// Share of the arm the GPU spent out of its off state
    pub gpu_active_share: Option<f64>,
    /// Mean GPU frequency while active, weighted by residency
    pub gpu_mean_mhz: Option<f64>,
    pub gpu_power_w: Option<f64>,
    pub ane_power_w: Option<f64>,
    /// Processes that submitted GPU work during the arm while their GPU time went down,
    /// because a command queue closed, so how much they used can't be measured
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unmeasured: Vec<GpuProcess>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuProcess {
    pub pid: u32,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClientUse {
    pub pid: u32,
    pub name: String,
    pub gpu_ms: f64,
    /// GPU time as a share of the time between the readings
    pub share: f64,
}

impl AppleArm {
    /// Compares the snapshots around an arm that ran for `wall_s` seconds, whose own clients
    /// were read as `bench`; `pstates_mhz` is the environment's P-state table
    ///
    /// Shares and power are over the time between the readings, which is a little longer
    /// than the arm; `wall_s` stands in when a reading has no timestamp
    pub fn between(
        before: &AppleState,
        after: &AppleState,
        wall_s: f64,
        bench: &BenchClients,
        pstates_mhz: Option<&[f64]>,
    ) -> Self {
        // in ns, so a share is one division of whole nanoseconds: summing rounded
        // per-process shares let a total of exactly the limit come out just under it
        let window_ns = |b: Option<u64>, a: Option<u64>| match (b, a) {
            (Some(b), Some(a)) if a > b => (a - b) as f64,
            _ => wall_s * 1e9,
        };
        let clients_ns = window_ns(before.clients_read_ns, after.clients_read_ns);
        let ioreport_s = window_ns(before.ioreport_read_ns, after.ioreport_read_ns) / 1e9;

        let start: BTreeMap<u64, u64> = before.clients.iter().map(|c| (c.id, c.gpu_time_ns)).collect();
        // the bench's clients that outlived it are in the second reading too
        let mut own = bench.gpu_time_ns.clone();
        for c in after.clients.iter().filter(|c| bench.pids.contains(&c.pid)) {
            let latest = own.entry(c.id).or_default();
            *latest = (*latest).max(c.gpu_time_ns);
        }
        let own_ns: u64 = own
            .iter()
            .map(|(id, ns)| ns.saturating_sub(start.get(id).copied().unwrap_or(0)))
            .sum();

        let mut by_pid: BTreeMap<u32, (String, u64)> = BTreeMap::new();
        let mut unmeasured: Vec<GpuProcess> = Vec::new();
        let mut window_server_ns = 0;
        for c in &after.clients {
            if bench.pids.contains(&c.pid) {
                continue;
            }

            // a client made during the arm counts from zero. Freeing a command queue drops
            // its time from the sum, which then can go down: harmless if the client sent
            // nothing during the arm, unmeasurable if it did
            let was = start.get(&c.id).copied().unwrap_or(0);
            if c.gpu_time_ns < was {
                let submitted = before.clients_read_ns.is_none_or(|t| c.last_submitted_ns > t);
                if submitted && c.name != WINDOW_SERVER && !unmeasured.iter().any(|p| p.pid == c.pid) {
                    unmeasured.push(GpuProcess {
                        pid: c.pid,
                        name: c.name.clone(),
                    });
                }

                continue;
            }

            let used = c.gpu_time_ns - was;
            if used == 0 {
                continue;
            }

            if c.name == WINDOW_SERVER {
                window_server_ns += used;
                continue;
            }

            by_pid.entry(c.pid).or_insert_with(|| (c.name.clone(), 0)).1 += used;
        }

        let other_ns: u64 = by_pid.values().map(|(_, ns)| ns).sum();
        let window_server_share = window_server_ns as f64 / clients_ns;
        let mut others: Vec<ClientUse> = by_pid
            .into_iter()
            .map(|(pid, (name, ns))| ClientUse {
                pid,
                name,
                gpu_ms: ns as f64 / 1e6,
                share: ns as f64 / clients_ns,
            })
            .collect();
        others.sort_by(|a, b| b.share.total_cmp(&a.share));

        let residency: Option<Vec<f64>> = match (&before.gpu_residency, &after.gpu_residency) {
            (Some(b), Some(a)) if a.len() == b.len() && !a.is_empty() => {
                Some(a.iter().zip(b).map(|(a, b)| (a - b).max(0) as f64).collect())
            }
            _ => None,
        };
        let (gpu_active_share, gpu_mean_mhz) = match &residency {
            Some(r) => {
                let total: f64 = r.iter().sum();
                let active: f64 = r[1..].iter().sum();
                let mean = pstates_mhz
                    .filter(|mhz| mhz.len() == r.len() && active > 0.0)
                    .map(|mhz| r.iter().zip(mhz).skip(1).map(|(t, f)| t * f).sum::<f64>() / active);
                ((total > 0.0).then(|| active / total), mean)
            }
            None => (None, None),
        };

        let power = |b: Option<f64>, a: Option<f64>| Some((a? - b?).max(0.0) / ioreport_s);
        Self {
            other_share: other_ns as f64 / clients_ns,
            others,
            window_server_share,
            own_share: bench.sampled.then(|| own_ns as f64 / clients_ns),
            gpu_active_share,
            gpu_mean_mhz,
            gpu_power_w: power(before.gpu_energy_j, after.gpu_energy_j),
            ane_power_w: power(before.ane_energy_j, after.ane_energy_j),
            unmeasured,
        }
    }

    /// Whether the bench used the GPU itself: its own share is at least
    /// [`OWN_GPU_IDLE_SHARE`], or wasn't read
    pub fn used_gpu(&self) -> bool {
        self.own_share.is_none_or(|s| s >= OWN_GPU_IDLE_SHARE)
    }

    /// Why the arm doesn't count, if other processes used the GPU too much during it
    ///
    /// Only an arm that used the GPU itself can be invalid: other processes' GPU use can't
    /// slow an arm that used none, so for one it is only recorded
    pub fn invalid_reason(&self) -> Option<String> {
        if !self.used_gpu() {
            return None;
        }

        if !self.unmeasured.is_empty() {
            let procs: Vec<String> = self
                .unmeasured
                .iter()
                .map(|p| format!("{} {}", p.pid, p.name))
                .collect();
            return Some(format!(
                "GPU use during the arm can't be measured: {} submitted work while a command queue closed",
                procs.join(", ")
            ));
        }

        if self.other_share < OTHER_GPU_MAX_SHARE {
            return None;
        }

        let procs: Vec<String> = self
            .others
            .iter()
            .take(3)
            .map(|o| format!("{} {} {:.1} ms", o.pid, o.name, o.gpu_ms))
            .collect();
        let more = self.others.len().saturating_sub(3);
        Some(format!(
            "other processes used the GPU for {:.1}% of the arm: {}{}",
            self.other_share * 100.0,
            procs.join(", "),
            if more > 0 {
                format!(" and {more} more")
            } else {
                String::new()
            }
        ))
    }
}

#[cfg(target_os = "macos")]
pub(super) use probe::{environment, snapshot};

#[cfg(test)]
mod tests {
    use super::{AppleArm, AppleState, BenchClients, GpuClient};

    /// A bench read during its arm that spent 500 ms of it on the GPU as pid 99
    fn gpu_arm() -> BenchClients {
        BenchClients {
            pids: [99].into(),
            gpu_time_ns: [(99, 500_000_000)].into(),
            sampled: true,
        }
    }

    fn client(id: u64, pid: u32, gpu_time_ns: u64, last_submitted_ns: u64) -> GpuClient {
        GpuClient {
            id,
            pid,
            name: format!("proc{pid}"),
            gpu_time_ns,
            last_submitted_ns,
        }
    }

    /// Readings of `clients` taken at `read_ns`
    fn state(read_ns: u64, clients: Vec<GpuClient>) -> AppleState {
        AppleState {
            device_utilization_pct: None,
            clients,
            clients_read_ns: Some(read_ns),
            thermal_pressure: None,
            power_mode: None,
            gpu_residency: None,
            gpu_energy_j: None,
            ane_energy_j: None,
            ioreport_read_ns: None,
        }
    }

    #[test]
    fn exactly_the_limit_is_invalid() {
        let before = state(0, (1..=3).map(|i| client(i, i as u32, 0, 0)).collect());
        // 10 ms of a 1 s window split three ways; summed f64 shares fall just under 1%
        let after = state(
            1_000_000_000,
            vec![
                client(1, 1, 3_333_334, 1),
                client(2, 2, 3_333_333, 1),
                client(3, 3, 3_333_333, 1),
            ],
        );
        let arm = AppleArm::between(&before, &after, 1.0, &gpu_arm(), None);
        assert!(arm.invalid_reason().is_some(), "{arm:?}");

        let after = state(1_000_000_000, vec![client(1, 1, 9_999_999, 1)]);
        let arm = AppleArm::between(&before, &after, 1.0, &gpu_arm(), None);
        assert_eq!(arm.invalid_reason(), None);
    }

    #[test]
    fn other_gpu_users_count_only_against_gpu_arms() {
        let before = state(0, vec![client(1, 7, 0, 0)]);
        // another process used 5% of a 1 s arm, and the bench's own client has closed
        let after = state(1_000_000_000, vec![client(1, 7, 50_000_000, 1)]);
        let cpu_arm = BenchClients {
            pids: [99].into(),
            sampled: true,
            ..BenchClients::default()
        };
        let arm = AppleArm::between(&before, &after, 1.0, &cpu_arm, None);
        assert_eq!(arm.own_share, Some(0.0));
        assert!((arm.other_share - 0.05).abs() < 1e-9, "{arm:?}");
        assert_eq!(arm.invalid_reason(), None);

        let arm = AppleArm::between(&before, &after, 1.0, &gpu_arm(), None);
        assert_eq!(arm.own_share, Some(0.5));
        assert!(arm.invalid_reason().is_some_and(|r| r.contains("5.0% of the arm")));

        // unread, the bench's own use is unknown and the arm is checked
        let arm = AppleArm::between(&before, &after, 1.0, &BenchClients::default(), None);
        assert_eq!(arm.own_share, None);
        assert!(arm.invalid_reason().is_some());
    }

    #[test]
    fn replaced_command_queue_is_unmeasured() {
        // the client's total falls from 1 s to 40 ms because a queue closed, but it
        // submitted during the arm, so its use is unknown rather than zero
        let before = state(100, vec![client(1, 7, 1_000_000_000, 50)]);
        let after = state(1_000_000_100, vec![client(1, 7, 40_000_000, 500)]);
        let arm = AppleArm::between(&before, &after, 1.0, &gpu_arm(), None);
        assert!(arm.invalid_reason().is_some_and(|r| r.contains("can't be measured")));

        // the same drop with nothing submitted since the first reading is harmless
        let after = state(1_000_000_100, vec![client(1, 7, 40_000_000, 50)]);
        let arm = AppleArm::between(&before, &after, 1.0, &gpu_arm(), None);
        assert_eq!(arm.invalid_reason(), None);
    }
}

#[cfg(target_os = "macos")]
mod probe {
    use std::{
        collections::BTreeMap,
        process::Command,
        sync::mpsc::{Receiver, RecvTimeoutError},
        time::Duration,
    };

    use eyre::{Result, WrapErr, bail};
    use regex::Regex;

    use super::{AppleState, BenchClients, GpuClient};
    use crate::{
        gpu::{Environment, GpuInfo, GpuSnapshot, ioreport},
        time::now_ct,
    };

    /// The accelerator's properties and its Metal clients
    struct Accelerator {
        model: Option<String>,
        cores: Option<u32>,
        driver: Option<String>,
        utilization: Option<f64>,
        clients: Vec<GpuClient>,
    }

    impl Accelerator {
        fn read() -> Result<Self> {
            let text = ioreg(&["-r", "-l", "-w0", "-c", "IOAccelerator"])?;
            Self::parse(&text)
        }

        /// Parses `ioreg -r -l -w0 -c IOAccelerator`: the accelerator's entry, then one
        /// entry for each user client below it
        fn parse(text: &str) -> Result<Self> {
            let mut entries = text.split("+-o ").filter(|e| !e.trim().is_empty());
            let Some(acc) = entries.next() else {
                bail!("ioreg lists no IOAccelerator; gpu-ab's macOS probe needs an Apple silicon GPU");
            };

            let string = |key: &str| {
                Regex::new(&format!(r#""{}" = "([^"]*)""#, regex::escape(key)))
                    .ok()?
                    .captures(acc)
                    .map(|c| c[1].to_owned())
            };
            let number = |re: &str| Regex::new(re).ok()?.captures(acc)?[1].parse::<f64>().ok();

            let id = Regex::new(r"^\S+\s+<class [^,]*, id 0x([0-9a-f]+)")?;
            let creator = Regex::new(r#""IOUserClientCreator" = "pid (\d+), ([^"]*)""#)?;
            let gpu_time = Regex::new(r#""accumulatedGPUTime"=(\d+)"#)?;
            let submitted = Regex::new(r#""lastSubmittedTime"=(\d+)"#)?;
            let mut clients = Vec::new();
            for entry in entries {
                let (Some(i), Some(c)) = (id.captures(entry), creator.captures(entry)) else {
                    continue;
                };

                let gpu_time_ns: u64 = gpu_time
                    .captures_iter(entry)
                    .filter_map(|t| t[1].parse::<u64>().ok())
                    .sum();
                if gpu_time_ns > 0 {
                    clients.push(GpuClient {
                        id: u64::from_str_radix(&i[1], 16)?,
                        pid: c[1].parse()?,
                        name: c[2].to_owned(),
                        gpu_time_ns,
                        last_submitted_ns: submitted
                            .captures_iter(entry)
                            .filter_map(|t| t[1].parse::<u64>().ok())
                            .max()
                            .unwrap_or(0),
                    });
                }
            }

            Ok(Self {
                model: string("model"),
                cores: number(r#""gpu-core-count" = (\d+)"#).map(|n| n as u32),
                driver: string("IOSourceVersion"),
                utilization: number(r#""Device Utilization %"=(\d+)"#),
                clients,
            })
        }
    }

    /// Reads the clients of `bench_pid` and its descendants until `stop` closes, for
    /// [`super::BenchSampler`]
    pub(super) fn watch_bench(bench_pid: u32, stop: &Receiver<()>) -> BenchClients {
        let mut bench = BenchClients {
            pids: [bench_pid].into(),
            ..BenchClients::default()
        };
        // whether each pid seen descends from the bench; a process can't gain an ancestor
        let mut descends: BTreeMap<u32, bool> = BTreeMap::new();
        let mut wait = Duration::from_millis(100);
        while let Err(RecvTimeoutError::Timeout) = stop.recv_timeout(wait) {
            wait = (wait * 2).min(Duration::from_secs(1));
            let Ok(acc) = Accelerator::read() else {
                continue;
            };

            bench.sampled = true;
            for c in acc.clients {
                if !*descends.entry(c.pid).or_insert_with(|| descends_from(c.pid, bench_pid)) {
                    continue;
                }

                bench.pids.insert(c.pid);
                let latest = bench.gpu_time_ns.entry(c.id).or_default();
                *latest = (*latest).max(c.gpu_time_ns);
            }
        }

        bench
    }

    /// Whether `pid` is `ancestor` or one of its descendants
    fn descends_from(mut pid: u32, ancestor: u32) -> bool {
        // a bound on the walk in case a parent can't be read consistently
        for _ in 0..64 {
            if pid == ancestor {
                return true;
            }

            match parent_pid(pid) {
                Some(parent) if parent > 1 && parent != pid => pid = parent,
                _ => return false,
            }
        }

        false
    }

    fn parent_pid(pid: u32) -> Option<u32> {
        let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
        // SAFETY: proc_pidinfo writes at most `size` bytes into the struct it is given
        let n = unsafe {
            libc::proc_pidinfo(
                pid as libc::c_int,
                libc::PROC_PIDTBSDINFO,
                0,
                (&raw mut info).cast(),
                size,
            )
        };
        (n == size).then_some(info.pbi_ppid)
    }

    pub(in crate::gpu) fn snapshot() -> Result<GpuSnapshot> {
        let acc = Accelerator::read()?;
        let clients_read_ns = uptime_ns();
        let report = ioreport::sample();
        let ioreport_read_ns = report.is_some().then(uptime_ns).flatten();
        let report = report.unwrap_or_default();
        Ok(GpuSnapshot {
            at: now_ct(),
            gpus: Vec::new(),
            compute_apps: Vec::new(),
            loadavg: loadavg(),
            apple: Some(AppleState {
                device_utilization_pct: acc.utilization,
                clients: acc.clients,
                clients_read_ns,
                thermal_pressure: thermal_pressure(),
                power_mode: power_mode(),
                gpu_residency: report.residency,
                gpu_energy_j: report.gpu_energy_j,
                ane_energy_j: report.ane_energy_j,
                ioreport_read_ns,
            }),
        })
    }

    fn uptime_ns() -> Option<u64> {
        let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: clock_gettime writes only the timespec it is given
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_UPTIME_RAW, &mut t) };
        (rc == 0).then(|| t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64)
    }

    pub(in crate::gpu) fn environment() -> Result<Environment> {
        let acc = Accelerator::read()?;
        let states = ioreport::sample().and_then(|s| s.residency).map(|r| r.len());
        // keep the table only when it lines up with the residency IOReport reports, since
        // which voltage-states entry is the GPU's isn't documented
        let pstates_mhz = gpu_pstates_mhz().filter(|t| Some(t.len()) == states);
        let os = sysctl("kern.osproductversion").map(|v| match sysctl("kern.osversion") {
            Some(build) => format!("macOS {v} ({build})"),
            None => format!("macOS {v}"),
        });

        Ok(Environment {
            host: sysctl("kern.hostname").unwrap_or_default(),
            kernel: sysctl("kern.osrelease").unwrap_or_default(),
            cpu: sysctl("machdep.cpu.brand_string"),
            driver: acc.driver.unwrap_or_default(),
            cuda: None,
            os,
            gpus: vec![GpuInfo {
                index: 0,
                name: acc.model.unwrap_or_else(|| "Apple GPU".to_owned()),
                memory_total_mib: sysctl("hw.memsize")
                    .and_then(|b| b.parse::<f64>().ok())
                    .map(|b| b / 1048576.0),
                clocks_max_sm_mhz: pstates_mhz.as_ref().map(|t| t.iter().copied().fold(0.0, f64::max)),
                power_limit_w: None,
                cores: acc.cores,
                pstates_mhz,
            }],
            loadavg: loadavg(),
            queue_job: std::env::var("HOMEBASED_JOB_ID").ok(),
            queue_resource: std::env::var("HOMEBASED_RESOURCE").ok(),
        })
    }

    /// The GPU's P-state frequencies from the power manager's `voltage-states9` table of
    /// (Hz, mV) pairs, which is the GPU's on M1 to M4
    fn gpu_pstates_mhz() -> Option<Vec<f64>> {
        let text = ioreg(&["-r", "-l", "-w0", "-d1", "-n", "pmgr"]).ok()?;
        let hex = Regex::new(r#""voltage-states9" = <([0-9a-f]+)>"#)
            .ok()?
            .captures(&text)?[1]
            .to_owned();
        let bytes: Vec<u8> = (0..hex.len() / 2)
            .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16))
            .collect::<Result<_, _>>()
            .ok()?;
        let mhz: Vec<f64> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|pair| u32::from_le_bytes([pair[0], pair[1], pair[2], pair[3]]) as f64 / 1e6)
            .collect();
        (!mhz.is_empty()).then_some(mhz)
    }

    fn thermal_pressure() -> Option<String> {
        let out = command("notifyutil", &["-g", "com.apple.system.thermalpressurelevel"])?;
        let level = match out.split_whitespace().last()? {
            "0" => "nominal",
            "1" => "moderate",
            "2" => "heavy",
            "3" => "trapping",
            "4" => "sleeping",
            other => return Some(format!("level {other}")),
        };
        Some(level.to_owned())
    }

    /// `powermode` on desktops (0 automatic, 1 low power, 2 high power), `lowpowermode`
    /// on laptops
    fn power_mode() -> Option<String> {
        let out = command("pmset", &["-g"])?;
        let setting = |key: &str| {
            out.lines().find_map(|l| {
                let mut it = l.split_whitespace();
                (it.next() == Some(key)).then(|| it.next().map(str::to_owned)).flatten()
            })
        };
        let mode = match (setting("powermode").as_deref(), setting("lowpowermode").as_deref()) {
            (Some("1"), _) | (None, Some("1")) => "low power",
            (Some("2"), _) => "high power",
            (Some("0"), _) | (None, Some("0")) => "automatic",
            _ => return None,
        };
        Some(mode.to_owned())
    }

    fn ioreg(args: &[&str]) -> Result<String> {
        let out = Command::new("ioreg").args(args).output().wrap_err("running ioreg")?;
        if !out.status.success() {
            bail!(
                "ioreg {} failed ({}): {}",
                args.join(" "),
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }

        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    fn command(program: &str, args: &[&str]) -> Option<String> {
        let out = Command::new(program).args(args).output().ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    }

    fn sysctl(key: &str) -> Option<String> {
        let v = command("sysctl", &["-n", key])?.trim().to_owned();
        (!v.is_empty()).then_some(v)
    }

    fn loadavg() -> Option<[f64; 3]> {
        let mut avg = [0.0; 3];
        // SAFETY: getloadavg writes at most the 3 elements it is given
        let n = unsafe { libc::getloadavg(avg.as_mut_ptr(), 3) };
        (n == 3).then_some(avg)
    }
}
