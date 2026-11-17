//! gpu-ab: paired A/B timing of two frozen GPU bench binaries, with every arm's output
//! checked against the control's

use std::{
    collections::BTreeMap,
    env,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{Args, Parser, Subcommand};
use eyre::{Result, WrapErr, bail, eyre};
use gpu_ab::{
    adapter::Adapter,
    job,
    plan::{Mode, Noise, PrepareOptions, prepare},
    record::{LOG_FILE, RESULT_FILE, RunResult, Status, noise_from_result},
    run::execute,
    schedule::abba,
    time::stamp_ct,
    verdict::MIN_PAIRS,
};

#[derive(Parser)]
#[command(version, about = "Decide whether a GPU performance change is a real win")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Freeze the binaries into a new run directory and write plan.json and job.json
    Prepare(Box<PrepareArgs>),
    /// Run every arm of a prepared run; this is the queue job's step
    Run { run_dir: PathBuf },
    /// Print a finished run's LOG entry, optionally re-deciding with another noise figure
    Report {
        run_dir: PathBuf,
        #[command(flatten)]
        noise: NoiseArgs,
    },
}

#[derive(Args)]
struct PrepareArgs {
    /// Adapter file, or a name looked up through the $GPU_AB_ADAPTERS path pattern, where
    /// {name} stands for the name (default $XDG_CONFIG_HOME/gpu-ab/adapters/{name}.toml,
    /// with XDG_CONFIG_HOME defaulting to ~/.config)
    #[arg(long)]
    adapter: String,
    /// Bench in the adapter; optional when it has only one
    #[arg(long)]
    bench: Option<String>,
    /// Short experiment label, such as exp07 or aa
    #[arg(long)]
    label: String,
    /// Control binary
    #[arg(long)]
    ctl: PathBuf,
    /// Experiment binary
    #[arg(long, required_unless_present = "aa", conflicts_with = "aa")]
    exp: Option<PathBuf>,
    /// A/A run: the control against a second frozen copy of itself
    #[arg(long)]
    aa: bool,
    /// Git checkout the control was built from, to record its commit
    #[arg(long)]
    ctl_src: Option<PathBuf>,
    /// Git checkout the experiment was built from
    #[arg(long)]
    exp_src: Option<PathBuf>,
    /// Counted pairs, run in ABBA order
    #[arg(long, default_value_t = 6)]
    pairs: usize,
    /// Uncounted warmup arms before the pairs, alternating from the control
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    /// Extra environment for control arms, KEY=VALUE
    #[arg(long = "ctl-env", value_parser = parse_kv)]
    ctl_env: Vec<(String, String)>,
    /// Extra environment for experiment arms, KEY=VALUE
    #[arg(long = "exp-env", value_parser = parse_kv)]
    exp_env: Vec<(String, String)>,
    #[command(flatten)]
    noise: NoiseArgs,
    /// Run directory; default _scratch/gpu-ab/<repo>-<bench>-<label>-<time> under the
    /// current directory
    #[arg(long)]
    run_dir: Option<PathBuf>,
    /// Queue resource for the optional homebased job spec, job.json
    #[arg(long, default_value = "gpu0")]
    resource: String,
    /// Queue priority for job.json: low, medium or high
    #[arg(long, default_value = "medium")]
    priority: String,
    /// Output-inactivity reminder for job.json (never kills the job)
    #[arg(long, default_value = "1h")]
    timeout: String,
}

#[derive(Args)]
struct NoiseArgs {
    /// A/A noise as the sd of per-pair ratios, such as 0.2% or 0.002
    #[arg(long, conflicts_with = "noise_from")]
    noise: Option<String>,
    /// Take the A/A noise from a finished A/A run directory or result.json
    #[arg(long)]
    noise_from: Option<PathBuf>,
}

impl NoiseArgs {
    fn resolve(&self) -> Result<Option<Noise>> {
        match (&self.noise, &self.noise_from) {
            (Some(n), _) => Ok(Some(Noise {
                sd: parse_noise(n)?,
                source: "supplied".to_owned(),
                bench: None,
            })),
            (None, Some(path)) => Ok(Some(noise_from_result(path)?)),
            (None, None) => Ok(None),
        }
    }
}

fn main() -> ExitCode {
    color_eyre::install().ok();
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("gpu-ab: {err:?}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Cmd::Prepare(args) => cmd_prepare(*args).map(|()| ExitCode::SUCCESS),
        Cmd::Run { run_dir } => cmd_run(&run_dir),
        Cmd::Report { run_dir, noise } => cmd_report(&run_dir, &noise).map(|()| ExitCode::SUCCESS),
    }
}

fn cmd_prepare(args: PrepareArgs) -> Result<()> {
    let adapter = resolve_adapter(&args.adapter)?;
    let run_dir = match args.run_dir {
        Some(d) => d,
        None => {
            let repo = Adapter::load(&adapter)?.name;
            let bench = args.bench.as_deref().map(|b| format!("-{b}")).unwrap_or_default();
            PathBuf::from("_scratch/gpu-ab").join(format!("{repo}{bench}-{}-{}", args.label, stamp_ct()))
        }
    };

    let opts = PrepareOptions {
        adapter,
        bench: args.bench,
        label: args.label,
        control: args.ctl,
        experiment: if args.aa { None } else { args.exp },
        control_src: args.ctl_src,
        experiment_src: args.exp_src,
        warmup: args.warmup,
        pairs: args.pairs,
        control_env: args.ctl_env.into_iter().collect::<BTreeMap<_, _>>(),
        experiment_env: args.exp_env.into_iter().collect::<BTreeMap<_, _>>(),
        noise: args.noise.resolve()?,
        run_dir,
        resource: args.resource,
        priority: args.priority,
        timeout: args.timeout,
        runner: env::current_exe().wrap_err("locating the gpu-ab executable")?,
    };
    let plan = prepare(&opts)?;
    let run_dir = opts.run_dir.canonicalize()?;

    let mode = if plan.mode == Mode::AA { "A/A" } else { "A/B" };
    println!("prepared {mode} run {}", run_dir.display());
    println!("  bench    {} {}", plan.adapter_name, plan.bench_name);
    println!(
        "  control  {} (sha256 {})",
        plan.control.source.display(),
        &plan.control.sha256[..16]
    );
    println!(
        "  exp      {} (sha256 {})",
        plan.experiment.source.display(),
        &plan.experiment.sha256[..16]
    );
    println!(
        "  arms     {} warmup + {} pairs ABBA = {} arms",
        plan.warmup,
        plan.pairs,
        abba(plan.warmup, plan.pairs).len()
    );
    if plan.pairs < MIN_PAIRS {
        println!("  warning  fewer than {MIN_PAIRS} pairs: the verdict can only be INCONCLUSIVE");
    }

    match &plan.noise {
        Some(n) => println!("  noise    {:.3}% ({})", n.sd * 100.0, n.source),
        None if plan.mode == Mode::AB => {
            println!("  noise    none: the verdict can't ACCEPT without --noise or --noise-from")
        }
        None => {}
    }

    // the queue is optional: the frozen gpu-ab copy runs the same way under any scheduler
    println!(
        "\nrun on an otherwise idle GPU with:\n  {} run {}",
        job::shell_quote(&plan.runner.path.display().to_string()),
        job::shell_quote(&run_dir.display().to_string())
    );
    println!(
        "or submit to a homebased queue with:\n  {}",
        job::submit_command(&plan, &run_dir)
    );
    Ok(())
}

fn cmd_run(run_dir: &Path) -> Result<ExitCode> {
    let result = match execute(run_dir, &mut |line| println!("{line}")) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("gpu-ab: run refused: {err:#}");
            if let Ok(dir) = run_dir.canonicalize() {
                eprintln!("record: {}", dir.join(RESULT_FILE).display());
            }

            return Ok(ExitCode::FAILURE);
        }
    };

    print_summary(&result);
    print_wrote(&result);
    Ok(ExitCode::SUCCESS)
}

fn cmd_report(run_dir: &Path, noise: &NoiseArgs) -> Result<()> {
    let mut result = RunResult::load(run_dir)?;
    if let Some(noise) = noise.resolve()? {
        if result.status != Status::Complete {
            bail!(
                "{} did not complete, so it has no verdict to re-decide",
                run_dir.display()
            );
        }

        if let Some((a, b)) = &noise.bench
            && (a != &result.plan.adapter_name || b != &result.plan.bench_name)
        {
            bail!(
                "noise from {} is for {a} {b}, not {} {}",
                noise.source,
                result.plan.adapter_name,
                result.plan.bench_name
            );
        }

        result.summarize(Some(noise));
        result.write()?;
        print_summary(&result);
        print_wrote(&result);
        return Ok(());
    }

    print_summary(&result);
    Ok(())
}

fn print_wrote(result: &RunResult) {
    println!(
        "wrote {} and {}",
        result.run_dir.join(RESULT_FILE).display(),
        result.run_dir.join(LOG_FILE).display()
    );
}

fn print_summary(result: &RunResult) {
    if let Some(v) = &result.verdict {
        println!("\n{}: {}", v.decision, v.reason);
    }

    println!("\n{}", result.log_entry());
}

/// A path that exists is used as is; a bare name is looked up through `GPU_AB_ADAPTERS`
fn resolve_adapter(arg: &str) -> Result<PathBuf> {
    // a directory named like a repo (running from a folder of adapters) isn't an adapter file
    let path = PathBuf::from(arg);
    if path.is_file() {
        return Ok(path);
    }

    if arg.contains('/') || arg.ends_with(".toml") {
        bail!("adapter file {arg} not found");
    }

    let pattern = match env::var("GPU_AB_ADAPTERS") {
        Ok(pattern) => pattern,
        Err(_) => default_adapter_pattern()?,
    };
    if !pattern.contains(ADAPTER_NAME) {
        bail!("GPU_AB_ADAPTERS must contain {ADAPTER_NAME}, got {pattern:?}");
    }

    let found = PathBuf::from(pattern.replace(ADAPTER_NAME, arg));
    if !found.is_file() {
        bail!(
            "no adapter for {arg}: {} not found; pass the adapter's path, or set GPU_AB_ADAPTERS \
             to a path pattern containing {ADAPTER_NAME}",
            found.display()
        );
    }

    Ok(found)
}

/// Stands for the adapter name in `GPU_AB_ADAPTERS`
const ADAPTER_NAME: &str = "{name}";

fn default_adapter_pattern() -> Result<String> {
    let config = match env::var_os("XDG_CONFIG_HOME") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(env::var_os("HOME").ok_or_else(|| eyre!("HOME is unset"))?).join(".config"),
    };
    let pattern = config.join("gpu-ab/adapters").join(format!("{ADAPTER_NAME}.toml"));
    pattern
        .into_os_string()
        .into_string()
        .map_err(|dir| eyre!("config directory {} is not UTF-8", dir.display()))
}

fn parse_kv(s: &str) -> Result<(String, String), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("expected KEY=VALUE, got {s:?}"))?;
    if k.is_empty() {
        return Err(format!("empty key in {s:?}"));
    }

    Ok((k.to_owned(), v.to_owned()))
}

fn parse_noise(s: &str) -> Result<f64> {
    let (num, scale) = match s.strip_suffix('%') {
        Some(n) => (n, 0.01),
        None => (s, 1.0),
    };
    let value = num.trim().parse::<f64>().wrap_err_with(|| format!("noise {s:?}"))? * scale;
    if !(value.is_finite() && (0.0..1.0).contains(&value)) {
        bail!("noise {s:?} must be a fraction below 1, such as 0.002 or 0.2%");
    }

    Ok(value)
}
