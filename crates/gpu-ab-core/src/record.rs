//! The run's result file (`result.json`) and its ready-to-paste `LOG.md` entry

use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use eyre::{Result, WrapErr, bail};
use serde::{Deserialize, Serialize};

use crate::{
    gpu::{AppleArm, Environment, GpuSnapshot},
    plan::{Mode, Noise, Plan},
    schedule::{Arm, Slot},
    stats::{PairStats, pair_stats},
    time::{heading_ct, heading_from, now_ct},
    verdict::{MIN_PAIRS, Verdict, decide},
};

pub const RESULT_FILE: &str = "result.json";
pub const LOG_FILE: &str = "LOG-entry.md";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Running,
    Complete,
    /// Stopped by a failed command or an output mismatch; no verdict
    Failed,
}

/// Everything a run measured and decided
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub tool: String,
    pub status: Status,
    pub error: Option<String>,
    pub started: String,
    pub finished: Option<String>,
    pub run_dir: PathBuf,
    pub plan: Plan,
    pub environment: Option<Environment>,
    /// Fingerprint every arm had to match, taken from the first control arm
    pub reference_fingerprint: Option<String>,
    pub arms: Vec<ArmRecord>,
    pub pairs: Vec<PairRecord>,
    pub stats: Option<PairStats>,
    /// Noise figure the verdict used
    pub noise: Option<Noise>,
    pub verdict: Option<Verdict>,
}

/// One completed bench arm whose output passed its check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmRecord {
    pub slot: Slot,
    pub label: String,
    pub dir: PathBuf,
    pub metric: f64,
    /// Wall time of the whole arm, including model load
    pub wall_s: f64,
    pub fingerprint: Option<String>,
    /// Whether the gate passed; `None` when the bench has no gate
    pub gate_passed: Option<bool>,
    /// False when another process was on the GPU before or after the arm, or, on Apple
    /// silicon, used it during an arm whose bench used the GPU itself
    pub valid: bool,
    pub invalid_reason: Option<String>,
    pub gpu_before: GpuSnapshot,
    pub gpu_after: GpuSnapshot,
    /// Apple GPU use during the arm, from the two snapshots; absent on NVIDIA hosts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apple: Option<AppleArm>,
}

/// One control/experiment pair
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairRecord {
    pub pair: usize,
    /// `AB` or `BA`
    pub order: String,
    pub control: f64,
    pub experiment: f64,
    /// Speedup, above 1 when the experiment wins
    pub ratio: f64,
    /// False when either arm was invalid; invalid pairs are left out of the statistics
    pub valid: bool,
}

impl RunResult {
    pub fn new(plan: Plan, run_dir: PathBuf) -> Self {
        Self {
            tool: concat!("gpu-ab ", env!("CARGO_PKG_VERSION")).to_owned(),
            status: Status::Running,
            error: None,
            started: now_ct(),
            finished: None,
            run_dir,
            plan,
            environment: None,
            reference_fingerprint: None,
            arms: Vec::new(),
            pairs: Vec::new(),
            stats: None,
            noise: None,
            verdict: None,
        }
    }

    pub fn load(run_dir: &Path) -> Result<Self> {
        let path = run_dir.join(RESULT_FILE);
        let text = fs::read_to_string(&path).wrap_err_with(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).wrap_err_with(|| format!("parsing {}", path.display()))
    }

    /// Writes `result.json`, and `LOG-entry.md` once the run has ended
    pub fn write(&self) -> Result<()> {
        fs::write(
            self.run_dir.join(RESULT_FILE),
            serde_json::to_string_pretty(self)? + "\n",
        )?;
        if self.status != Status::Running {
            fs::write(self.run_dir.join(LOG_FILE), self.log_entry())?;
        }

        Ok(())
    }

    /// Pairs the arms, computes the statistics and applies the verdict rule
    ///
    /// The noise figure is `noise` if given, else the plan's, else, for an A/A run, the
    /// run's own spread
    pub fn summarize(&mut self, noise: Option<Noise>) {
        let better = self.plan.bench.metric.better;
        self.pairs = (0..self.plan.pairs)
            .filter_map(|p| {
                let members: Vec<&ArmRecord> = self.arms.iter().filter(|a| a.slot.pair == Some(p)).collect();
                let ctl = members.iter().find(|a| a.slot.arm == Arm::Control)?;
                let exp = members.iter().find(|a| a.slot.arm == Arm::Experiment)?;
                Some(PairRecord {
                    pair: p,
                    order: if ctl.slot.index < exp.slot.index { "AB" } else { "BA" }.to_owned(),
                    control: ctl.metric,
                    experiment: exp.metric,
                    ratio: better.ratio(ctl.metric, exp.metric),
                    valid: ctl.valid && exp.valid,
                })
            })
            .collect();

        let ratios: Vec<f64> = self.pairs.iter().filter(|p| p.valid).map(|p| p.ratio).collect();
        self.stats = pair_stats(&ratios);
        self.noise = noise
            .or_else(|| self.plan.noise.clone())
            .or_else(|| match (self.plan.mode, &self.stats) {
                (Mode::AA, Some(s)) => Some(Noise {
                    sd: s.sd,
                    source: "this run".to_owned(),
                    bench: None,
                }),
                _ => None,
            });
        self.verdict = Some(decide(self.stats.as_ref(), self.noise.as_ref().map(|n| n.sd)));
    }

    /// Ready-to-paste Markdown entry for an experiment log; lines gpu-ab can't know
    /// are left as `<fill in>`
    pub fn log_entry(&self) -> String {
        let plan = &self.plan;
        let bench = format!("{} {}", plan.adapter_name, plan.bench_name);
        // prefer the job the run actually ran in over the id prepare suggested
        let env = self.environment.as_ref();
        let resource = env
            .and_then(|e| e.queue_resource.clone())
            .unwrap_or_else(|| plan.job.resource.clone());
        let job_id = env
            .and_then(|e| e.queue_job.clone())
            .unwrap_or_else(|| plan.job.id.to_string());
        let job = format!("{resource} job {}", short(&job_id));
        let mut s = String::new();

        let title = match (self.status, plan.mode, &self.stats, &self.verdict) {
            (Status::Failed | Status::Running, _, _, _) => format!("{bench}: refused, {}", self.error_head()),
            (_, Mode::AA, Some(st), Some(v)) => format!("{bench} A/A: noise {}, {}", pct(st.sd), v.decision),
            (_, Mode::AB, Some(st), Some(v)) => {
                format!("{bench}: {}, x{:.3} {}", v.decision, st.mean, plan.bench.metric.name)
            }
            (_, _, _, Some(v)) => format!("{bench}: {}", v.decision),
            _ => bench.clone(),
        };
        // date the entry by when the run ended, so a later `report` doesn't re-date it
        let when = self
            .finished
            .as_deref()
            .and_then(heading_from)
            .unwrap_or_else(heading_ct);
        let _ = writeln!(s, "## {when} {} {title}", plan.label);

        let fill = plan.mode == Mode::AB && self.status == Status::Complete;
        if fill {
            s.push_str("- Row and hypothesis: <fill in>\n- Ceiling: <fill in>\n");
        }

        let _ = writeln!(
            s,
            "- Base {}; ctl bin {}, exp bin {}",
            self.base(),
            short(&plan.control.sha256),
            short(&plan.experiment.sha256)
        );

        if self.status != Status::Complete {
            let _ = writeln!(
                s,
                "- Refused in {job} after {} checked arms: {}",
                self.arms.len(),
                self.error.as_deref().unwrap_or("run did not finish")
            );
            let _ = writeln!(s, "- Record: {}", self.run_dir.join(RESULT_FILE).display());
            return s;
        }

        let invalid = self.arms.iter().filter(|a| !a.valid).count();
        let mode = if plan.mode == Mode::AA { "A/A" } else { "A/B" };
        if let (Some(st), Some(v)) = (&self.stats, &self.verdict) {
            let noise = match &self.noise {
                Some(n) => format!("A/A noise {} ({})", pct(n.sd), n.source),
                None => "no A/A noise figure".to_owned(),
            };
            let _ = writeln!(
                s,
                "- {mode}: {job}, {} valid of {} pairs ABBA (+{} warmup), mean x{:.4}, 95% CI x{:.4}-{:.4}, sd {}; {noise}; {}: {}",
                st.n,
                plan.pairs,
                plan.warmup,
                st.mean,
                st.ci_low,
                st.ci_high,
                pct(st.sd),
                v.decision,
                v.reason
            );
        } else if let Some(v) = &self.verdict {
            let _ = writeln!(s, "- {mode}: {job}, {} pairs; {}: {}", plan.pairs, v.decision, v.reason);
        }

        let unit = plan
            .bench
            .metric
            .unit
            .as_deref()
            .map(|u| format!(" {u}"))
            .unwrap_or_default();
        let med = |arm| {
            median(
                self.arms
                    .iter()
                    .filter(|a| a.slot.pair.is_some() && a.valid && a.slot.arm == arm)
                    .map(|a| a.metric),
            )
        };
        let _ = writeln!(
            s,
            "- Metric: {} ({} is better), median ctl {}{unit}, exp {}{unit}; {invalid} invalid arms",
            plan.bench.metric.name,
            match plan.bench.metric.better {
                crate::adapter::Better::Lower => "lower",
                crate::adapter::Better::Higher => "higher",
            },
            fmt_opt(med(Arm::Control)),
            fmt_opt(med(Arm::Experiment)),
        );

        let checked = self.arms.len();
        let mut acc = Vec::new();
        if let Some(fp) = &self.reference_fingerprint {
            acc.push(format!(
                "output fingerprint {} identical on all {checked} arms",
                short(fp)
            ));
        }

        if plan.bench.fingerprint.gate.is_some() {
            acc.push(format!("gate passed on all {checked} arms"));
        }

        let _ = writeln!(s, "- Accuracy: {}", acc.join("; "));
        let _ = writeln!(s, "- Env: {}", self.env_line());

        if fill {
            s.push_str("- Guards: <fill in>\n- Profile: <fill in>\n- Cause: <fill in>\n- Not tested: <fill in>\n- Decision: <fill in>\n");
        }

        let _ = writeln!(s, "- Record: {}", self.run_dir.join(RESULT_FILE).display());
        s
    }

    fn error_head(&self) -> String {
        let err = self.error.as_deref().unwrap_or("run did not finish");
        let head = err.lines().next().unwrap_or(err);
        if head.chars().count() > 80 {
            format!("{}...", head.chars().take(80).collect::<String>())
        } else {
            head.to_owned()
        }
    }

    fn base(&self) -> String {
        let commit = |f: &crate::freeze::Frozen| {
            f.git
                .as_ref()
                .map(|g| format!("{}{}", short(&g.commit), if g.dirty { "-dirty" } else { "" }))
        };
        match (commit(&self.plan.control), commit(&self.plan.experiment)) {
            (Some(c), Some(e)) if c == e => c,
            (Some(c), Some(e)) => format!("ctl {c}, exp {e}"),
            (Some(c), None) => c,
            (None, Some(e)) => format!("exp {e}"),
            (None, None) => "<fill in>".to_owned(),
        }
    }

    fn env_line(&self) -> String {
        if self.environment.as_ref().is_some_and(|e| e.os.is_some()) {
            return self.apple_env_line();
        }

        let mut parts = Vec::new();
        if let Some(env) = &self.environment {
            let gpu = env.gpus.first().map(|g| g.name.as_str()).unwrap_or("no GPU");
            parts.push(format!(
                "{gpu}, driver {}, CUDA {}",
                env.driver,
                env.cuda.as_deref().unwrap_or("?")
            ));
        }

        let after: Vec<&crate::gpu::GpuState> = self.arms.iter().flat_map(|a| a.gpu_after.gpus.first()).collect();
        let range = |vals: Vec<f64>| -> Option<(f64, f64)> {
            let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            lo.is_finite().then_some((lo, hi))
        };
        if let Some((lo, hi)) = range(after.iter().filter_map(|g| g.clocks_sm_mhz).collect()) {
            parts.push(format!("SM {lo:.0}-{hi:.0} MHz after arms"));
        }

        if let Some((lo, hi)) = range(after.iter().filter_map(|g| g.temperature_c).collect()) {
            parts.push(format!("{lo:.0}-{hi:.0} C"));
        }

        let slowed = self
            .arms
            .iter()
            .filter(|a| {
                a.gpu_before
                    .gpus
                    .iter()
                    .chain(&a.gpu_after.gpus)
                    .any(|g| g.slowed_down())
            })
            .count();
        parts.push(if slowed == 0 {
            "no thermal or hardware slowdown".to_owned()
        } else {
            format!("slowdown flags on {slowed} arms")
        });

        let loads: Vec<f64> = self
            .arms
            .iter()
            .flat_map(|a| a.gpu_before.loadavg.map(|l| l[0]))
            .collect();
        if let Some((_, hi)) = range(loads) {
            parts.push(format!("host load <= {hi:.2}"));
        }

        parts.join("; ")
    }
}

impl RunResult {
    /// The `Env` line for Apple silicon: what the GPU ran at and who else used it
    fn apple_env_line(&self) -> String {
        let mut parts = Vec::new();
        if let Some(env) = &self.environment {
            let gpu = env.gpus.first();
            let name = gpu.map(|g| g.name.as_str()).unwrap_or("Apple GPU");
            let cores = gpu
                .and_then(|g| g.cores)
                .map(|c| format!(" ({c}-core GPU)"))
                .unwrap_or_default();
            parts.push(format!(
                "{name}{cores}, {}, AGX driver {}",
                env.os.as_deref().unwrap_or("macOS ?"),
                env.driver
            ));
        }

        let arms: Vec<&AppleArm> = self.arms.iter().filter_map(|a| a.apple.as_ref()).collect();
        if let Some((lo, hi)) = range(arms.iter().filter_map(|a| a.gpu_mean_mhz)) {
            parts.push(format!("GPU {} MHz mean while active", span(lo, hi, 0)));
        }

        if let Some((lo, hi)) = range(arms.iter().filter_map(|a| a.gpu_power_w)) {
            parts.push(format!("{} W", span(lo, hi, 1)));
        }

        let snaps = || {
            self.arms
                .iter()
                .flat_map(|a| [&a.gpu_before, &a.gpu_after])
                .filter_map(|s| s.apple.as_ref())
        };
        let pressure = |a: &&ArmRecord| -> Vec<String> {
            [&a.gpu_before, &a.gpu_after]
                .iter()
                .filter_map(|s| s.apple.as_ref()?.thermal_pressure.clone())
                .collect()
        };
        let read = self.arms.iter().filter(|a| !pressure(a).is_empty()).count();
        let hot = self
            .arms
            .iter()
            .filter(|a| pressure(a).iter().any(|t| t != "nominal"))
            .count();
        parts.push(match (read, hot) {
            (0, _) => "thermal pressure not read".to_owned(),
            (_, 0) => "thermal pressure nominal".to_owned(),
            _ => format!("thermal pressure above nominal on {hot} arms"),
        });

        let mut modes: Vec<&str> = snaps().filter_map(|s| s.power_mode.as_deref()).collect();
        modes.sort_unstable();
        modes.dedup();
        if !modes.is_empty() {
            parts.push(format!("power mode {}", modes.join(" and ")));
        }

        if let Some((_, hi)) = range(arms.iter().map(|a| a.other_share)) {
            parts.push(format!("other GPU users <= {}", pct(hi)));
        }

        let idle = arms.iter().filter(|a| !a.used_gpu()).count();
        if idle > 0 {
            parts.push(format!(
                "{idle} arms used no GPU, so other GPU users didn't count against them"
            ));
        }

        if let Some((_, hi)) = range(arms.iter().map(|a| a.window_server_share)) {
            parts.push(format!("WindowServer <= {}", pct(hi)));
        }

        if let Some((_, hi)) = range(arms.iter().filter_map(|a| a.ane_power_w)) {
            parts.push(format!("ANE <= {hi:.1} W"));
        }

        if let Some((_, hi)) = range(self.arms.iter().filter_map(|a| a.gpu_before.loadavg.map(|l| l[0]))) {
            parts.push(format!("host load <= {hi:.2}"));
        }

        parts.join("; ")
    }
}

/// `lo-hi`, or one number when they print the same
fn span(lo: f64, hi: f64, decimals: usize) -> String {
    let (lo, hi) = (format!("{lo:.decimals$}"), format!("{hi:.decimals$}"));
    if lo == hi { lo } else { format!("{lo}-{hi}") }
}

fn range(vals: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
    let (lo, hi) = vals.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)));
    lo.is_finite().then_some((lo, hi))
}

/// Reads the noise figure (per-pair ratio sd) from a finished A/A run's result file
pub fn noise_from_result(path: &Path) -> Result<Noise> {
    let path = if path.is_dir() {
        path.join(RESULT_FILE)
    } else {
        path.to_owned()
    };
    let text = fs::read_to_string(&path).wrap_err_with(|| format!("reading {}", path.display()))?;
    let result: RunResult = serde_json::from_str(&text).wrap_err_with(|| format!("parsing {}", path.display()))?;
    if result.plan.mode != Mode::AA {
        bail!("{} is an A/B run; noise comes from an A/A run", path.display());
    }

    if result.status != Status::Complete {
        bail!("{} did not complete", path.display());
    }

    let Some(stats) = result.stats.filter(|s| s.n >= MIN_PAIRS) else {
        bail!(
            "{} has fewer than {MIN_PAIRS} valid pairs, too few for a noise figure",
            path.display()
        );
    };

    Ok(Noise {
        sd: stats.sd,
        source: format!("{} {} {}", result.plan.bench_name, result.plan.label, path.display()),
        bench: Some((result.plan.adapter_name, result.plan.bench_name)),
    })
}

fn median(vals: impl Iterator<Item = f64>) -> Option<f64> {
    let mut v: Vec<f64> = vals.collect();
    if v.is_empty() {
        return None;
    }

    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    Some(if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    })
}

fn pct(frac: f64) -> String {
    format!("{:.3}%", frac * 100.0)
}

fn short(s: &str) -> &str {
    &s[..s.len().min(8)]
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.4}")).unwrap_or_else(|| "-".to_owned())
}
