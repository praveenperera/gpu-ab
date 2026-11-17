//! A prepared run: the resolved bench, the frozen binaries and the schedule settings,
//! saved as `plan.json` in the run directory

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use eyre::{Result, WrapErr, bail};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    adapter::{Adapter, Bench, Vars},
    freeze::{Frozen, freeze, git_source},
    job,
    time::now_ct,
};

pub const PLAN_FILE: &str = "plan.json";
const PLAN_VERSION: u32 = 1;

/// A/A compares the control with a copy of itself to measure paired noise
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    #[serde(rename = "A/A")]
    AA,
    #[serde(rename = "A/B")]
    AB,
}

/// An A/A noise figure: the standard deviation of per-pair ratios, as a fraction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Noise {
    pub sd: f64,
    /// `supplied`, `this run`, or the A/A result file it came from
    pub source: String,
    /// Adapter and bench of the A/A run it came from, so it isn't applied to another bench
    #[serde(default)]
    pub bench: Option<(String, String)>,
}

/// Queue settings for the job spec
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobSettings {
    /// Job id to submit with, chosen up front so a lost reply can be retried
    pub id: Uuid,
    pub resource: String,
    pub priority: String,
    pub timeout: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub version: u32,
    pub created: String,
    pub label: String,
    pub mode: Mode,
    pub adapter_path: PathBuf,
    pub adapter_name: String,
    pub repo: PathBuf,
    pub bench_name: String,
    /// Copy of the bench as it was at prepare time, so later adapter edits can't change
    /// a queued run
    pub bench: Bench,
    pub warmup: usize,
    pub pairs: usize,
    pub control: Frozen,
    pub experiment: Frozen,
    pub control_env: BTreeMap<String, String>,
    pub experiment_env: BTreeMap<String, String>,
    pub noise: Option<Noise>,
    pub job: JobSettings,
    /// Frozen copy of gpu-ab itself, which the job runs
    pub runner: Frozen,
}

/// Inputs to [`prepare`]
#[derive(Debug, Clone)]
pub struct PrepareOptions {
    pub adapter: PathBuf,
    pub bench: Option<String>,
    pub label: String,
    pub control: PathBuf,
    /// `None` for an A/A run against a copy of the control
    pub experiment: Option<PathBuf>,
    pub control_src: Option<PathBuf>,
    pub experiment_src: Option<PathBuf>,
    pub warmup: usize,
    pub pairs: usize,
    pub control_env: BTreeMap<String, String>,
    pub experiment_env: BTreeMap<String, String>,
    pub noise: Option<Noise>,
    pub run_dir: PathBuf,
    pub resource: String,
    pub priority: String,
    pub timeout: String,
    /// The gpu-ab executable to freeze for the job
    pub runner: PathBuf,
}

/// Creates the run directory, freezes the binaries, and writes `plan.json` and the queue
/// job spec `job.json`
pub fn prepare(opts: &PrepareOptions) -> Result<Plan> {
    let adapter = Adapter::load(&opts.adapter)?;
    let (bench_name, bench) = adapter.bench(opts.bench.as_deref())?;
    if opts.pairs < 2 {
        bail!("need at least 2 pairs, got {}", opts.pairs);
    }

    if opts.label.is_empty() || opts.label.contains(char::is_whitespace) {
        bail!("label must be one non-empty word, got {:?}", opts.label);
    }

    if opts.run_dir.exists() {
        bail!(
            "run directory {} already exists; each run gets a fresh one",
            opts.run_dir.display()
        );
    }

    if let Some(Noise {
        bench: Some((a, b)),
        source,
        ..
    }) = &opts.noise
        && (a != &adapter.name || b != bench_name)
    {
        bail!("noise from {source} is for {a} {b}, not {} {bench_name}", adapter.name);
    }

    // per-arm env is expanded at run time, so catch placeholder typos now
    for value in opts.control_env.values().chain(opts.experiment_env.values()) {
        Vars::check(value)?;
    }

    // resolve everything that can fail before creating the directory, so a typo doesn't
    // leave a half-made run that blocks the retry
    for path in [Some(&opts.control), opts.experiment.as_ref()].into_iter().flatten() {
        if !path.is_file() {
            bail!("binary {} is not a file", path.display());
        }
    }

    let control_git = opts.control_src.as_deref().map(git_source).transpose()?;
    let experiment_git = match &opts.experiment {
        None => control_git.clone(),
        Some(_) => opts.experiment_src.as_deref().map(git_source).transpose()?,
    };

    fs::create_dir_all(opts.run_dir.join("bin"))?;
    let run_dir = opts.run_dir.canonicalize()?;
    let bin = run_dir.join("bin");

    let control = freeze(&opts.control, &bin.join("control"), control_git)?;
    let (mode, experiment) = match &opts.experiment {
        None => (
            Mode::AA,
            freeze(&opts.control, &bin.join("experiment"), experiment_git)?,
        ),
        Some(path) => (Mode::AB, freeze(path, &bin.join("experiment"), experiment_git)?),
    };

    let plan = Plan {
        version: PLAN_VERSION,
        created: now_ct(),
        label: opts.label.clone(),
        mode,
        adapter_path: opts.adapter.canonicalize()?,
        adapter_name: adapter.name.clone(),
        repo: adapter.repo.clone(),
        bench_name: bench_name.to_owned(),
        bench: bench.clone(),
        warmup: opts.warmup,
        pairs: opts.pairs,
        control,
        experiment,
        control_env: opts.control_env.clone(),
        experiment_env: opts.experiment_env.clone(),
        noise: opts.noise.clone(),
        job: JobSettings {
            id: Uuid::new_v4(),
            resource: opts.resource.clone(),
            priority: opts.priority.clone(),
            timeout: opts.timeout.clone(),
        },
        runner: freeze(&opts.runner, &bin.join("gpu-ab"), None)?,
    };

    plan.save(&run_dir)?;
    let spec = serde_json::to_string_pretty(&job::spec(&plan, &run_dir))?;
    fs::write(run_dir.join(job::JOB_FILE), spec + "\n")?;
    Ok(plan)
}

impl Plan {
    pub fn load(run_dir: &Path) -> Result<Self> {
        let path = run_dir.join(PLAN_FILE);
        let text = fs::read_to_string(&path).wrap_err_with(|| format!("reading {}", path.display()))?;
        let plan: Plan = serde_json::from_str(&text).wrap_err_with(|| format!("parsing {}", path.display()))?;
        if plan.version != PLAN_VERSION {
            bail!(
                "{} has plan version {}, this gpu-ab reads {PLAN_VERSION}",
                path.display(),
                plan.version
            );
        }

        Ok(plan)
    }

    pub fn save(&self, run_dir: &Path) -> Result<()> {
        fs::write(run_dir.join(PLAN_FILE), serde_json::to_string_pretty(self)? + "\n")?;
        Ok(())
    }

    /// Frozen binary and extra environment for an arm
    pub fn arm(&self, arm: crate::schedule::Arm) -> (&Frozen, &BTreeMap<String, String>) {
        match arm {
            crate::schedule::Arm::Control => (&self.control, &self.control_env),
            crate::schedule::Arm::Experiment => (&self.experiment, &self.experiment_env),
        }
    }
}
