//! Runs the CLI end to end against a fake bench script and a fake `nvidia-smi` (Linux) or
//! `ioreg` (macOS), no GPU

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::Value;
use tempfile::TempDir;

const FAKE_SMI: &str = r#"#!/bin/sh
case "$1" in
  --query-gpu=index,name*) echo "0, Fake GPU, 595.99.02, 32607, 3090, 575.00" ;;
  --query-gpu=*) echo "0, 2900, 13801, 55, 310.5, 97, 0x0000000000000004" ;;
  --query-compute-apps=*) [ -n "$FAKE_GPU_BUSY" ] && echo "4242, /usr/bin/other job, 1024" ;;
  *) echo "| NVIDIA-SMI 595.99.02    Driver Version: 595.99.02    CUDA Version: 13.2 |" ;;
esac
exit 0
"#;

/// Stands in for `ioreg -r -l -w0 -c IOAccelerator` and the power manager's P-state table.
/// WindowServer gains 10 s of GPU time per call, far more than any arm lasts, and must not
/// invalidate anything; with `FAKE_GPU_BUSY`, another process gains as much and must. With
/// `FAKE_GPU_DROP`, a process keeps submitting while its GPU time goes down, as when it
/// closes a command queue, which must also invalidate the arm. While a GPU bench runs, its
/// pid is in `bench.pid` beside this script, and that process gains 100 ms per call
const FAKE_IOREG: &str = r#"#!/bin/sh
state="$(dirname "$0")/ioreg.calls"
n=$(cat "$state" 2>/dev/null || echo 0)
n=$((n + 1))
echo $n > "$state"
case "$*" in
  *pmgr*) echo '    "voltage-states9" = <000000007d00000000ca9a3be8030000>'; exit 0 ;;
esac
cat <<END
+-o AGXAcceleratorG16X  <class AGXAcceleratorG16X, id 0x10000040a, registered, matched, active, busy 0 (1 ms), retain 168>
    {
      "PerformanceStatistics" = {"Tiler Utilization %"=1,"Renderer Utilization %"=2,"Device Utilization %"=3}
      "model" = "Fake Apple GPU"
      "gpu-core-count" = 20
      "IOSourceVersion" = "360.34.5"
    }
    
  +-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x10000111b, !registered, !matched, active, busy 0, retain 5>
      {
        "AppUsage" = ({"API"="Metal","lastSubmittedTime"=1,"accumulatedGPUTime"=$((n * 10000000000))},{"API"="Metal","lastSubmittedTime"=0,"accumulatedGPUTime"=0})
        "IOUserClientCreator" = "pid 415, WindowServer"
      }
    
  +-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x1000012b0, !registered, !matched, active, busy 0, retain 5>
      {
        "AppUsage" = ()
        "IOUserClientCreator" = "pid 647, MenuBarAgent"
      }
END
if [ -n "$FAKE_GPU_DROP" ]; then
cat <<END
  +-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x1000012b8, !registered, !matched, active, busy 0, retain 5>
      {
        "AppUsage" = ({"API"="Metal","lastSubmittedTime"=18446744073709551615,"accumulatedGPUTime"=$((1000000000000 - n * 1000))})
        "IOUserClientCreator" = "pid 5151, queue closer"
      }
END
fi
bench=$(cat "$(dirname "$0")/bench.pid" 2>/dev/null)
if [ -n "$bench" ]; then
cat <<END
  +-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x1000012bc, !registered, !matched, active, busy 0, retain 5>
      {
        "AppUsage" = ({"API"="Metal","lastSubmittedTime"=1,"accumulatedGPUTime"=$((n * 100000000))})
        "IOUserClientCreator" = "pid $bench, gpu bench"
      }
END
fi
if [ -n "$FAKE_GPU_BUSY" ]; then
cat <<END
  +-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x1000012b4, !registered, !matched, active, busy 0, retain 5>
      {
        "AppUsage" = ({"API"="Metal","lastSubmittedTime"=1,"accumulatedGPUTime"=$((n * 10000000000))})
        "IOUserClientCreator" = "pid 4242, other job"
      }
END
fi
exit 0
"#;

/// What the fake GPU tools put in every log entry's `Env` line
#[cfg(target_os = "macos")]
const FAKE_ENV: &str = "Fake Apple GPU (20-core GPU)";
#[cfg(not(target_os = "macos"))]
const FAKE_ENV: &str = "CUDA 13.2";

/// How the busy fake process shows up in an invalid arm's reason
#[cfg(target_os = "macos")]
const FAKE_BUSY: &str = "of the arm: 4242 other job ";
#[cfg(not(target_os = "macos"))]
const FAKE_BUSY: &str = "4242 /usr/bin/other job";

/// What a fake bench does before it writes its output
#[derive(Clone, Copy)]
enum Work<'a> {
    /// Nothing, so the arm takes a few milliseconds
    Instant,
    /// Half a second without the GPU
    Cpu,
    /// Half a second on the fake GPU in a child process, which writes its pid to this file
    /// and closes its client before the bench exits, as a real GPU bench's worker does
    Gpu(&'a Path),
}

/// A bench whose metric is `base` plus a small wobble from a shared counter, and whose
/// output file holds `text`
fn bench_script(base: &str, text: &str, work: Work) -> String {
    let work = match work {
        Work::Instant => String::new(),
        Work::Cpu => "sleep 0.5\n".to_owned(),
        Work::Gpu(pid_file) => {
            let f = pid_file.display();
            format!("sh -c 'echo $$ > \"{f}\"; sleep 0.5'\nrm -f \"{f}\"\n")
        }
    };
    format!(
        r#"#!/bin/sh
set -e
out=$1
n=$(cat "$COUNTER" 2>/dev/null || echo 0)
echo $((n + 1)) > "$COUNTER"
{work}echo "{text}" > "$out/result.txt"
echo "loading model"
echo "time: {base}$((n % 3)) s"
"#
    )
}

/// Each fake bench's file name, metric base and output
const BENCHES: [(&str, &str, &str); 3] = [
    ("ctl.sh", "10.0", "speaker A 0.00 1.50"),
    ("fast.sh", "9.5", "speaker A 0.00 1.50"),
    ("broken.sh", "9.5", "speaker B 0.00 1.50"),
];

const ADAPTER: &str = r#"
name = "fake"
repo = "REPO"

[bench.quick]
description = "fake bench"
argv = ["{bin}", "{out}"]
env = { COUNTER = "{run_dir}/counter" }
metric = { name = "wall", unit = "s", better = "lower", regex = 'time: ([0-9.]+) s' }
fingerprint = { hash = ["sh", "-c", "sha256sum {out}/result.txt | cut -c1-64"], gate = ["cmp", "{out}/result.txt", "{ctl_out}/result.txt"] }
"#;

struct Fixture {
    /// Held so the directory is deleted when the fixture drops
    _dir: TempDir,
    /// The temp dir's canonical path; on macOS it sits behind the `/var` symlink
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let root = root.as_path();
        fs::create_dir(root.join("fakebin")).unwrap();
        write_exec(&root.join("fakebin/nvidia-smi"), FAKE_SMI);
        write_exec(&root.join("fakebin/ioreg"), FAKE_IOREG);
        for (name, base, text) in BENCHES {
            write_exec(&root.join(name), &bench_script(base, text, Work::Instant));
        }
        fs::write(
            root.join("adapter.toml"),
            ADAPTER.replace("REPO", root.to_str().unwrap()),
        )
        .unwrap();
        Self {
            root: root.to_owned(),
            _dir: dir,
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// Makes the named benches use the fake GPU themselves
    fn use_gpu(&self, names: &[&str]) {
        let pid_file = self.path("fakebin/bench.pid");
        self.set_work(names, Work::Gpu(&pid_file));
    }

    fn set_work(&self, names: &[&str], work: Work) {
        for name in names {
            let (_, base, text) = BENCHES.iter().find(|b| b.0 == *name).unwrap();
            write_exec(&self.path(name), &bench_script(base, text, work));
        }
    }

    /// Runs gpu-ab with `env` on top, such as `FAKE_GPU_BUSY` for the fake GPU tools
    fn gpu_ab(&self, args: &[&str], env: &[(&str, &str)]) -> Output {
        let path = format!("{}:{}", self.path("fakebin").display(), std::env::var("PATH").unwrap());
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_gpu-ab"));
        cmd.args(args).current_dir(&self.root).env("PATH", path);
        // adapter lookup by name must not depend on the developer's own setup
        cmd.env_remove("GPU_AB_ADAPTERS").env_remove("XDG_CONFIG_HOME");
        cmd.envs(env.iter().copied());
        cmd.output().unwrap()
    }

    /// Prepares a run with the given extra args, checks the job spec, runs it, and
    /// returns the run's output and result.json
    fn prepare_and_run(&self, name: &str, extra: &[&str], env: &[(&str, &str)]) -> (Output, Value) {
        let run_dir = self.path(name);
        let mut args = vec![
            "prepare",
            "--adapter",
            "adapter.toml",
            "--label",
            name,
            "--ctl",
            "ctl.sh",
            "--run-dir",
            run_dir.to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        let prep = self.gpu_ab(&args, env);
        assert!(
            prep.status.success(),
            "prepare failed: {}",
            String::from_utf8_lossy(&prep.stderr)
        );
        let stdout = String::from_utf8_lossy(&prep.stdout);
        assert!(
            stdout.contains(&format!("bin/gpu-ab run {}", run_dir.display())),
            "{stdout}"
        );
        assert!(stdout.contains("resource job submit --job-id"), "{stdout}");

        let job: Value = serde_json::from_str(&fs::read_to_string(run_dir.join("job.json")).unwrap()).unwrap();
        let step = &job["steps"][0]["command"];
        assert_eq!(job["preempt"]["mode"], "wait");
        assert_eq!(job["resource"], "gpu0");
        assert_eq!(step[1], "run");
        assert_eq!(step[2], run_dir.to_str().unwrap());
        assert!(
            Path::new(step[0].as_str().unwrap()).starts_with(&run_dir),
            "job must run the frozen gpu-ab copy"
        );

        let out = self.gpu_ab(&["run", run_dir.to_str().unwrap()], env);
        let result: Value = serde_json::from_str(&fs::read_to_string(run_dir.join("result.json")).unwrap()).unwrap();
        (out, result)
    }
}

fn write_exec(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn aa_run_measures_noise_and_does_not_accept() {
    let fx = Fixture::new();
    let (out, result) = fx.prepare_and_run("aa", &["--aa"], &[]);
    assert!(
        out.status.success(),
        "{}{}",
        stdout(&out),
        String::from_utf8_lossy(&out.stderr)
    );

    assert_eq!(result["status"], "complete");
    assert_eq!(result["plan"]["mode"], "A/A");
    assert_eq!(
        result["plan"]["control"]["sha256"],
        result["plan"]["experiment"]["sha256"]
    );
    assert_ne!(result["plan"]["control"]["path"], result["plan"]["experiment"]["path"]);

    // 2 warmup arms + 6 pairs, in ABBA order after the warmup
    let arms = result["arms"].as_array().unwrap();
    let order: String = arms
        .iter()
        .map(
            |a| match (a["slot"]["arm"].as_str().unwrap(), a["slot"]["pair"].is_null()) {
                ("control", true) => 'a',
                ("experiment", true) => 'b',
                ("control", false) => 'A',
                _ => 'B',
            },
        )
        .collect();
    assert_eq!(order, "abABBAABBAABBA");
    assert!(arms.iter().all(|a| a["valid"] == true && a["gate_passed"] == true));
    // CUDA records keep their shape: the Apple fields appear only on macOS
    let text = fs::read_to_string(fx.path("aa/result.json")).unwrap();
    for key in ["\"apple\"", "\"os\"", "\"cores\""] {
        assert_eq!(text.contains(key), cfg!(target_os = "macos"), "{key}");
    }
    // the fake P-state table doesn't line up with the real IOReport states, so it's dropped
    assert!(!text.contains("\"pstates_mhz\""));
    assert!(
        arms.iter()
            .all(|a| Path::new(a["dir"].as_str().unwrap()).join("stdout.txt").exists())
    );

    // the counter wobble gives a nonzero noise figure, taken from this run
    let sd = result["stats"]["sd"].as_f64().unwrap();
    assert!(sd > 0.0 && sd < 0.01, "sd {sd}");
    assert_eq!(result["stats"]["n"], 6);
    assert_eq!(result["noise"]["source"], "this run");
    assert_ne!(result["verdict"]["decision"], "ACCEPT");

    let log = fs::read_to_string(fx.path("aa/LOG-entry.md")).unwrap();
    assert!(
        log.starts_with("## ") && log.contains(" CT aa fake quick A/A: noise "),
        "{log}"
    );
    assert!(
        log.contains("identical on all 14 arms") && log.contains(FAKE_ENV),
        "{log}"
    );

    // the noise feeds an A/B run, which accepts a clear 5% win
    let (out, ab) = fx.prepare_and_run(
        "ab",
        &["--exp", "fast.sh", "--noise-from", fx.path("aa").to_str().unwrap()],
        &[],
    );
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(ab["plan"]["mode"], "A/B");
    let mean = ab["stats"]["mean"].as_f64().unwrap();
    assert!((1.04..1.06).contains(&mean), "mean {mean}");
    assert_eq!(ab["verdict"]["decision"], "ACCEPT", "{}", ab["verdict"]);
    assert!(stdout(&out).contains("ACCEPT"));
    let log = fs::read_to_string(fx.path("ab/LOG-entry.md")).unwrap();
    assert!(
        log.contains("fake quick: ACCEPT, x1.05") && log.contains("Decision: <fill in>"),
        "{log}"
    );

    // a re-run of a finished directory is refused
    let again = fx.gpu_ab(&["run", fx.path("ab").to_str().unwrap()], &[]);
    assert!(!again.status.success());
}

#[test]
fn adapter_names_resolve_through_the_config_dir_or_a_pattern() {
    let fx = Fixture::new();
    let adapter = fs::read_to_string(fx.path("adapter.toml")).unwrap();
    let prepare = |label: &str, env: &[(&str, &str)]| {
        let run_dir = fx.path(label);
        let run_arg = run_dir.to_str().unwrap();
        let args = [
            "prepare",
            "--adapter",
            "fake",
            "--label",
            label,
            "--ctl",
            "ctl.sh",
            "--aa",
            "--run-dir",
            run_arg,
        ];
        let out = fx.gpu_ab(&args, env);
        let plan = fs::read_to_string(run_dir.join("plan.json")).ok();
        (out, plan.map(|p| serde_json::from_str::<Value>(&p).unwrap()))
    };

    // nothing configured: the default location is empty, and the error names the fix
    let config = fx.path("config");
    let (out, _) = prepare("none", &[("XDG_CONFIG_HOME", config.to_str().unwrap())]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("gpu-ab/adapters/fake.toml not found"), "{stderr}");
    assert!(stderr.contains("GPU_AB_ADAPTERS"), "{stderr}");

    let default = config.join("gpu-ab/adapters/fake.toml");
    fs::create_dir_all(default.parent().unwrap()).unwrap();
    fs::write(&default, &adapter).unwrap();
    let (out, plan) = prepare("xdg", &[("XDG_CONFIG_HOME", config.to_str().unwrap())]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(plan.unwrap()["adapter_path"], default.to_str().unwrap());

    // a pattern can point into any layout, such as one folder per repo
    let nested = fx.path("ops/fake/hill-climb/gpu-ab.toml");
    fs::create_dir_all(nested.parent().unwrap()).unwrap();
    fs::write(&nested, &adapter).unwrap();
    let pattern = fx.path("ops/{name}/hill-climb/gpu-ab.toml");
    let (out, plan) = prepare("pattern", &[("GPU_AB_ADAPTERS", pattern.to_str().unwrap())]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(plan.unwrap()["adapter_path"], nested.to_str().unwrap());
}

#[test]
fn wrong_output_is_refused_without_a_verdict() {
    let fx = Fixture::new();
    let (out, result) = fx.prepare_and_run("broken", &["--exp", "broken.sh", "--noise", "0.2%"], &[]);
    assert!(!out.status.success(), "a broken experiment must fail the job");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("output fingerprint") && err.contains("01-w-exp"), "{err}");

    assert_eq!(result["status"], "failed");
    assert!(result["verdict"].is_null() && result["stats"].is_null());
    assert_eq!(
        result["arms"].as_array().unwrap().len(),
        1,
        "only the first control arm passed"
    );
    let log = fs::read_to_string(fx.path("broken/LOG-entry.md")).unwrap();
    assert!(log.contains("refused") && !log.contains("ACCEPT"), "{log}");
}

#[test]
fn failing_gate_is_refused_even_without_a_hash() {
    let fx = Fixture::new();
    let adapter = fs::read_to_string(fx.path("adapter.toml")).unwrap();
    let gate_only = adapter.replace(
        r#"hash = ["sh", "-c", "sha256sum {out}/result.txt | cut -c1-64"], "#,
        "",
    );
    assert_ne!(adapter, gate_only);
    fs::write(fx.path("adapter.toml"), gate_only).unwrap();

    let (out, result) = fx.prepare_and_run("gate", &["--exp", "broken.sh"], &[]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("gate"));
    assert_eq!(result["status"], "failed");
}

#[test]
fn other_gpu_process_invalidates_arms() {
    let fx = Fixture::new();
    fx.use_gpu(&["ctl.sh", "fast.sh"]);
    let (out, result) = fx.prepare_and_run(
        "busy",
        &["--exp", "fast.sh", "--noise", "0.2%"],
        &[("FAKE_GPU_BUSY", "1")],
    );
    assert!(out.status.success());
    let arms = result["arms"].as_array().unwrap();
    assert!(arms.iter().all(|a| a["valid"] == false));
    let reason = arms[0]["invalid_reason"].as_str().unwrap();
    assert!(reason.contains(FAKE_BUSY), "{reason}");
    assert_eq!(result["verdict"]["decision"], "INCONCLUSIVE");
    assert!(
        result["verdict"]["reason"]
            .as_str()
            .unwrap()
            .contains("fewer than 2 valid pairs")
    );
}

#[cfg(target_os = "macos")]
#[test]
fn unmeasurable_gpu_use_invalidates_arms() {
    let fx = Fixture::new();
    fx.use_gpu(&["ctl.sh"]);
    let (out, result) = fx.prepare_and_run("drop", &["--aa"], &[("FAKE_GPU_DROP", "1")]);
    assert!(out.status.success());
    let arms = result["arms"].as_array().unwrap();
    assert!(arms.iter().all(|a| a["valid"] == false));
    let reason = arms[0]["invalid_reason"].as_str().unwrap();
    assert!(reason.contains("can't be measured: 5151 queue closer"), "{reason}");
}

#[cfg(target_os = "macos")]
#[test]
fn other_gpu_users_count_only_against_arms_that_used_the_gpu() {
    let fx = Fixture::new();
    // a CPU-only control against an experiment that uses the GPU, while another process
    // keeps the GPU busy: only the experiment's arms can be slowed by it
    fx.set_work(&["ctl.sh"], Work::Cpu);
    fx.use_gpu(&["fast.sh"]);
    let (out, result) = fx.prepare_and_run(
        "mixed",
        &["--exp", "fast.sh", "--noise", "0.2%"],
        &[("FAKE_GPU_BUSY", "1")],
    );
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    for arm in result["arms"].as_array().unwrap() {
        let apple = &arm["apple"];
        assert!(apple["other_share"].as_f64().unwrap() >= 0.01, "{apple}");
        let own = apple["own_share"].as_f64().unwrap();
        if arm["slot"]["arm"] == "control" {
            assert_eq!(own, 0.0, "{arm}");
            assert_eq!(arm["valid"], true, "{arm}");
        } else {
            assert!(own > 0.01, "the experiment's own GPU time was read: {apple}");
            assert!(arm["invalid_reason"].as_str().unwrap().contains(FAKE_BUSY), "{arm}");
        }
    }
    assert_eq!(result["verdict"]["decision"], "INCONCLUSIVE");

    // with neither arm on the GPU, the same busy GPU leaves every arm valid, and the log
    // says why
    let fx = Fixture::new();
    fx.set_work(&["ctl.sh", "fast.sh"], Work::Cpu);
    let (out, result) = fx.prepare_and_run(
        "cpu",
        &["--exp", "fast.sh", "--noise", "0.2%"],
        &[("FAKE_GPU_BUSY", "1")],
    );
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let arms = result["arms"].as_array().unwrap();
    assert!(arms.iter().all(|a| a["valid"] == true), "{}", result["arms"]);
    assert_eq!(result["verdict"]["decision"], "ACCEPT", "{}", result["verdict"]);
    let log = fs::read_to_string(fx.path("cpu/LOG-entry.md")).unwrap();
    assert!(
        log.contains("14 arms used no GPU, so other GPU users didn't count against them"),
        "{log}"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn arm_too_short_to_sample_is_checked_as_a_gpu_arm() {
    // an arm that ends before the first reading of its own clients can't show it used no
    // GPU, so other GPU users still count against it
    let fx = Fixture::new();
    let (out, result) = fx.prepare_and_run("short", &["--aa"], &[("FAKE_GPU_BUSY", "1")]);
    assert!(out.status.success());
    let arms = result["arms"].as_array().unwrap();
    assert!(
        arms.iter()
            .all(|a| a["valid"] == false && a["apple"]["own_share"].is_null())
    );
}
