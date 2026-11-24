//! Executes a prepared run: every arm back to back, each one checked before its timing
//! counts

use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    time::Instant,
};

use eyre::{Result, WrapErr, bail, eyre};
use regex::Regex;

use crate::{
    adapter::{MetricSpec, Vars},
    gpu::{self, AppleArm, BenchSampler, GpuSnapshot},
    plan::Plan,
    record::{ArmRecord, RunResult, Status},
    schedule::{Arm, Slot, abba},
    time::now_ct,
};

/// sha256 of empty input: a hash command whose glob matched nothing prints this
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Runs every arm of the plan in `run_dir`, writing `result.json` as it goes and
/// `LOG-entry.md` at the end
///
/// A failed command or an output mismatch stops the run: the result is written with
/// status `failed` and the error is returned. `progress` receives one line per arm
pub fn execute(run_dir: &Path, progress: &mut dyn FnMut(&str)) -> Result<RunResult> {
    let run_dir = run_dir
        .canonicalize()
        .wrap_err_with(|| format!("run directory {}", run_dir.display()))?;
    let plan = Plan::load(&run_dir)?;
    let arms_dir = run_dir.join("arms");
    if arms_dir.exists() {
        bail!(
            "{} already ran; prepare a fresh run directory instead of re-running it",
            run_dir.display()
        );
    }

    fs::create_dir(&arms_dir)?;
    let mut result = RunResult::new(plan, run_dir);
    match run_arms(&mut result, progress) {
        Ok(()) => {
            result.status = Status::Complete;
            result.finished = Some(now_ct());
            result.summarize(None);
            result.write()?;
            Ok(result)
        }
        Err(err) => {
            result.status = Status::Failed;
            result.error = Some(format!("{err:#}"));
            result.finished = Some(now_ct());
            result.write()?;
            Err(err)
        }
    }
}

/// The control arm every other arm's output is compared with
struct Reference {
    label: String,
    out: PathBuf,
    fingerprint: Option<String>,
}

fn run_arms(result: &mut RunResult, progress: &mut dyn FnMut(&str)) -> Result<()> {
    result.environment = Some(gpu::environment().wrap_err("reading the GPU environment")?);
    result.write()?;

    let slots = abba(result.plan.warmup, result.plan.pairs);
    let total = slots.len();
    let mut reference: Option<Reference> = None;
    for slot in slots {
        let arm = run_arm(result, &slot, &mut reference)
            .wrap_err_with(|| format!("arm {} ({})", slot.label(), slot.arm.short()))?;
        progress(&format!(
            "[{:>2}/{total}] {:<10} {} {:.4}{}  wall {:.1}s{}",
            slot.index + 1,
            arm.label,
            result.plan.bench.metric.name,
            arm.metric,
            result
                .plan
                .bench
                .metric
                .unit
                .as_deref()
                .map(|u| format!(" {u}"))
                .unwrap_or_default(),
            arm.wall_s,
            arm.invalid_reason
                .as_deref()
                .map(|r| format!("  INVALID: {r}"))
                .unwrap_or_default(),
        ));
        result.reference_fingerprint = reference.as_ref().and_then(|r| r.fingerprint.clone());
        result.arms.push(arm);
        result.write()?;
    }

    Ok(())
}

fn run_arm(result: &RunResult, slot: &Slot, reference: &mut Option<Reference>) -> Result<ArmRecord> {
    let plan = &result.plan;
    let bench = &plan.bench;
    let label = slot.label();
    let arm_dir = result.run_dir.join("arms").join(&label);
    let out = arm_dir.join("out");
    fs::create_dir_all(&out)?;
    let stdout_path = arm_dir.join("stdout.txt");
    let stderr_path = arm_dir.join("stderr.txt");

    let (frozen, arm_env) = plan.arm(slot.arm);
    let mut vars = Vars::new();
    vars.set("bin", path_str(&frozen.path))
        .set("out", path_str(&out))
        .set("arm_dir", path_str(&arm_dir))
        .set("stdout", path_str(&stdout_path))
        .set("ctl_out", path_str(reference.as_ref().map_or(&out, |r| &r.out)))
        .set("run_dir", path_str(&result.run_dir))
        .set("repo", path_str(&plan.repo));

    let argv = vars.expand_all(&bench.argv)?;
    let cwd = match &bench.cwd {
        Some(c) => PathBuf::from(vars.expand(c)?),
        None => plan.repo.clone(),
    };
    let env = bench
        .env
        .iter()
        .chain(arm_env)
        .map(|(k, v)| Ok((k.clone(), vars.expand(v)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;

    let gpu_before = gpu::snapshot()?;
    let started = Instant::now();
    // every arm is a fresh process, so no state carries over from the arm before it
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .current_dir(&cwd)
        .envs(&env)
        .stdin(Stdio::null())
        .stdout(File::create(&stdout_path)?)
        .stderr(File::create(&stderr_path)?)
        .spawn()
        .wrap_err_with(|| format!("starting {:?} in {}", argv[0], cwd.display()))?;
    let sampler = BenchSampler::start(child.id());
    let status = child.wait().wrap_err_with(|| format!("waiting for {:?}", argv[0]))?;
    let wall_s = started.elapsed().as_secs_f64();
    let bench_clients = sampler.finish();
    let gpu_after = gpu::snapshot()?;
    let apple = match (&gpu_before.apple, &gpu_after.apple) {
        (Some(before), Some(after)) => {
            let pstates = result
                .environment
                .as_ref()
                .and_then(|e| e.gpus.first())
                .and_then(|g| g.pstates_mhz.as_deref());
            Some(AppleArm::between(before, after, wall_s, &bench_clients, pstates))
        }
        _ => None,
    };

    if !status.success() {
        bail!("bench exited with {status}; stderr tail:\n{}", tail(&stderr_path, 15));
    }

    let mut fingerprint = None;
    if let Some(cmd) = &bench.fingerprint.hash {
        let fp = hash_output(&vars.expand_all(cmd)?, &cwd, &env, &arm_dir)?;
        match reference {
            Some(Reference {
                label: ref_label,
                fingerprint: Some(want),
                ..
            }) if *want != fp => bail!(
                "output fingerprint {} differs from the control's {} (arm {ref_label}); \
                 refusing to time a run whose output changed. Compare {} with the control arm's out/",
                short(&fp),
                short(want),
                out.display(),
            ),
            _ => fingerprint = Some(fp),
        }
    }

    let gate_passed = match &bench.fingerprint.gate {
        Some(cmd) => {
            let argv = vars.expand_all(cmd)?;
            let log = arm_dir.join("gate.txt");
            let status = run_logged(&argv, &cwd, &env, &log)?;
            if !status.success() {
                bail!(
                    "gate {:?} failed with {status}; output tail:\n{}",
                    argv.join(" "),
                    tail(&log, 15)
                );
            }

            Some(true)
        }
        None => None,
    };

    let metric = read_metric(&bench.metric, &vars).wrap_err("reading the metric")?;

    if reference.is_none() {
        if slot.arm != Arm::Control {
            bail!("the first arm must be a control arm");
        }

        *reference = Some(Reference {
            label: label.clone(),
            out: out.clone(),
            fingerprint: fingerprint.clone(),
        });
    }

    let invalid_reason = other_processes(&gpu_before, "before")
        .or_else(|| other_processes(&gpu_after, "after"))
        .or_else(|| apple.as_ref().and_then(AppleArm::invalid_reason));
    Ok(ArmRecord {
        slot: slot.clone(),
        label,
        dir: arm_dir,
        metric,
        wall_s,
        fingerprint,
        gate_passed,
        valid: invalid_reason.is_none(),
        invalid_reason,
        gpu_before,
        gpu_after,
        apple,
    })
}

/// Runs the hash command and returns its trimmed stdout, refusing empty output or the
/// hash of nothing
fn hash_output(argv: &[String], cwd: &Path, env: &BTreeMap<String, String>, arm_dir: &Path) -> Result<String> {
    let log = arm_dir.join("fingerprint.txt");
    let output = Command::new(&argv[0])
        .args(&argv[1..])
        .current_dir(cwd)
        .envs(env)
        .stdin(Stdio::null())
        .output()
        .wrap_err_with(|| format!("starting hash command {:?}", argv[0]))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    fs::write(
        &log,
        format!("{stdout}\n--- stderr\n{}", String::from_utf8_lossy(&output.stderr)),
    )?;

    if !output.status.success() {
        bail!(
            "hash command {:?} failed with {}; see {}",
            argv.join(" "),
            output.status,
            log.display()
        );
    }

    if stdout.is_empty() || stdout.starts_with(EMPTY_SHA256) {
        bail!(
            "hash command {:?} hashed no output; did its paths match anything? See {}",
            argv.join(" "),
            log.display()
        );
    }

    Ok(stdout)
}

fn run_logged(argv: &[String], cwd: &Path, env: &BTreeMap<String, String>, log: &Path) -> Result<ExitStatus> {
    let file = File::create(log)?;
    Command::new(&argv[0])
        .args(&argv[1..])
        .current_dir(cwd)
        .envs(env)
        .stdin(Stdio::null())
        .stdout(file.try_clone()?)
        .stderr(file)
        .status()
        .wrap_err_with(|| format!("starting {:?}", argv[0]))
}

/// Reads the metric named by the adapter from the arm's files
pub fn read_metric(spec: &MetricSpec, vars: &Vars) -> Result<f64> {
    let pattern = vars.expand(&spec.file)?;
    let mut matches: Vec<PathBuf> = glob::glob(&pattern)?.collect::<Result<_, _>>()?;
    let path = match matches.len() {
        1 => matches.remove(0),
        0 => bail!("metric file {pattern} matched nothing"),
        n => bail!("metric file {pattern} matched {n} files, need exactly one"),
    };
    let text = fs::read_to_string(&path).wrap_err_with(|| format!("reading {}", path.display()))?;

    let value = if let Some(re) = &spec.regex {
        let re = Regex::new(re)?;
        let caps = re
            .captures(&text)
            .ok_or_else(|| eyre!("metric regex {re} found nothing in {}", path.display()))?;
        let raw = caps
            .get(1)
            .ok_or_else(|| eyre!("metric regex {re} has no first group"))?
            .as_str();
        raw.trim()
            .parse::<f64>()
            .wrap_err_with(|| format!("metric {raw:?} from {} is not a number", path.display()))?
    } else {
        let ptr = spec.json.as_deref().unwrap_or_default();
        let json: serde_json::Value =
            serde_json::from_str(&text).wrap_err_with(|| format!("parsing {}", path.display()))?;
        json.pointer(ptr)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| eyre!("no number at JSON pointer {ptr} in {}", path.display()))?
    };

    if !(value.is_finite() && value > 0.0) {
        bail!(
            "metric {value} from {} must be a positive finite number",
            path.display()
        );
    }

    Ok(value)
}

fn other_processes(snap: &GpuSnapshot, when: &str) -> Option<String> {
    if snap.compute_apps.is_empty() {
        return None;
    }

    let procs: Vec<String> = snap
        .compute_apps
        .iter()
        .map(|a| format!("{} {}", a.pid, a.name))
        .collect();
    Some(format!("other process on the GPU {when} the arm: {}", procs.join(", ")))
}

fn tail(path: &Path, lines: usize) -> String {
    let text = fs::read_to_string(path).unwrap_or_default();
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

fn short(s: &str) -> &str {
    &s[..s.len().min(12)]
}
