# gpu-ab

`gpu-ab` decides whether a GPU performance change is a real win. It times two frozen bench binaries, a control and an experiment, in ABBA-ordered pairs in one sitting on one GPU. Every arm's output is checked against the control's, and GPU state is recorded around each arm. The per-pair ratios then go through a fixed acceptance rule, [the verdict](#the-verdict): ACCEPT, REJECT or INCONCLUSIVE.

It does one thing: a paired timing comparison of two binaries on the repo's own bench. It has no per-layer gates, reference math, locking or plugins. Profiling, parity suites and guard workloads stay in each repo's own tooling.

## Install

On an x86_64 Linux NVIDIA host or an Apple silicon Mac, install or update the latest release into `~/.local/bin`:

```sh
curl -fsSL https://github.com/praveenperera/gpu-ab/releases/latest/download/install.sh | bash
```

`install.sh` uses `gh` when it is logged in, and plain `curl` otherwise; set `GPU_AB_INSTALL_DIR` to install elsewhere. From a checkout, `just install` does the same, and `just release local` builds the checkout and installs it there instead. `~/.local/bin` must come before any older `cargo install` copy in `~/.cargo/bin` on `PATH`; `install.sh` warns when it doesn't.

The workspace has two crates:

- `gpu-ab-core` (library `gpu_ab`): adapters, scheduling, statistics, the verdict rule, records and the job spec. Other tools can reuse it.
- `gpu-ab` (binary): a thin CLI over it.

## How a run works

1. **Prepare.** `gpu-ab prepare` creates a run directory and copies the control and experiment binaries (and `gpu-ab` itself) into `bin/`, recording their sha256. It then writes `plan.json`, and `job.json` for [queue users](#queue-integration), and prints the command that runs it. Only the copies run, so a rebuild before the run starts can't change what it measures.
2. **Run.** On the GPU host, with nothing else using the GPU, run the printed `<run>/bin/gpu-ab run <run>`, directly or as the only step of a queue job. It runs every arm back to back: by default 2 uncounted warmup arms (control, then experiment), then 6 pairs in ABBA order (`AB BA AB BA AB BA`). For each arm it:
   - records GPU state and the host load average before and after the arm: on Linux, `nvidia-smi` clocks, temperature, power, utilization, throttle reasons and compute processes; on macOS, the readings in [Apple silicon](#apple-silicon);
   - runs the bench in a fresh process with the adapter's argv, env and working directory, and fails the run if it exits nonzero;
   - checks the output: the hash command's output must equal the first control arm's, and the gate command, if any, must exit 0. A mismatch stops the run with an error naming the arm, and no verdict is written;
   - reads the metric from the arm's files;
   - marks the arm invalid if any compute process was on the GPU before or after it, or, on macOS, if other processes used the GPU for at least 1% of an arm whose bench used the GPU itself. An arm whose bench used none can't be slowed by other GPU users, so on macOS their use is recorded for it but doesn't invalidate it (see [Other GPU users](#apple-silicon)). Pairs with an invalid arm are left out of the statistics.
3. **Decide.** It writes `result.json` and `LOG-entry.md`, a ready-to-paste Markdown entry for an experiment log, and prints the verdict. Lines it can't know (hypothesis, ceiling, guards, cause, decision) are left as `<fill in>`.

### A/A noise

An A/A run (`--aa`) times the control against a second copy of itself to measure paired noise. Its noise figure is the standard deviation of the per-pair ratios. Pass it to later A/B runs with `--noise-from <aa run dir>`, or as a number with `--noise 0.2%`. `--noise-from` takes only a completed A/A run with at least 6 valid pairs, on the same adapter and bench.

### The verdict

Each pair's ratio is a speedup, above 1 when the experiment wins: control ÷ experiment for lower-is-better metrics such as wall time, and experiment ÷ control for higher-is-better ones such as RTFx. Over the valid pairs (a pair is invalid when either of its arms is, by the rule in step 2 of [How a run works](#how-a-run-works)), gpu-ab takes the mean and a 95% t-interval (df = pairs − 1). With `bar = 1 + 2 × noise`:

| Verdict | Rule |
|---|---|
| ACCEPT | interval's lower end above 1, and mean at least `bar` |
| REJECT | interval's upper end below `bar`: the data rule out a gain that clears it, including any significant slowdown |
| INCONCLUSIVE | anything else, or fewer than 6 valid pairs, or no noise figure (then only a slowdown with the whole interval below 1 is REJECT) |

A/A runs use their own spread as the noise when none is supplied. They should never come out ACCEPT.

A verdict is one run's evidence. Before keeping a change, re-confirm an ACCEPT with a fresh run, and run the full accuracy suite and any guard workloads, which gpu-ab doesn't.

## Example

```sh
# A/A: paired noise for speakrs Cuda mode
gpu-ab prepare --adapter speakrs --bench sg --label aa \
  --ctl _scratch/climb/bin/speakrs-bm-ctl --ctl-src . --aa \
  --run-dir _scratch/climb/runs/aa
_scratch/climb/runs/aa/bin/gpu-ab run _scratch/climb/runs/aa

# A/B against that noise
gpu-ab prepare --adapter speakrs --bench sg --label exp07 \
  --ctl _scratch/climb/bin/speakrs-bm-ctl --ctl-src . \
  --exp _scratch/climb/bin/speakrs-bm-exp07 --exp-src ../speakrs.wt/exp07 \
  --noise-from _scratch/climb/runs/aa --run-dir _scratch/climb/runs/exp07
_scratch/climb/runs/exp07/bin/gpu-ab run _scratch/climb/runs/exp07

# afterwards: the entry is in runs/exp07/LOG-entry.md; re-decide with another noise figure
gpu-ab report _scratch/climb/runs/exp07 --noise 0.15%
```

A real A/A run of [scriptrs](https://github.com/avencera/scriptrs) LibriSpeech b1 on an RTX 5090 (queue job `0657f6e7`, 2026-10-30), as its log shows it:

```
[ 1/14] 00-w-ctl   LS b1 warm wall 12.8511 s  wall 28.6s
[ 2/14] 01-w-exp   LS b1 warm wall 12.6951 s  wall 28.6s
[ 3/14] 02-p0-ctl  LS b1 warm wall 12.7402 s  wall 28.5s
...
[14/14] 13-p5-ctl  LS b1 warm wall 12.7618 s  wall 28.1s

REJECT: 95% CI x0.9932-1.0007 lies below the bar x1.0072 (2x A/A noise)

## 2026-10-30 15:17 CT aa1 scriptrs ls-b1 A/A: noise 0.360%, REJECT
- Base 895bbe9f; ctl bin a172b038, exp bin a172b038
- A/A: gpu0 job 0657f6e7, 6 valid of 6 pairs ABBA (+2 warmup), mean x0.9970, 95% CI x0.9932-1.0007, sd 0.360%; A/A noise 0.360% (this run); REJECT: 95% CI x0.9932-1.0007 lies below the bar x1.0072 (2x A/A noise)
- Metric: LS b1 warm wall (lower is better), median ctl 12.7373 s, exp 12.7699 s; 0 invalid arms
- Accuracy: output fingerprint 99d15f9a identical on all 14 arms
- Env: NVIDIA GeForce RTX 5090, driver 595.99.02, CUDA 13.2; SM 2895-2917 MHz after arms; 49-62 C; no thermal or hardware slowdown; host load <= 6.63
- Record: <run>/result.json
```

The same control against a build whose transcripts were uppercased (job `2c54a3a9`) stopped at the first experiment arm, and the job failed:

```
[ 1/14] 00-w-ctl   LS b1 warm wall 12.7290 s  wall 28.2s
gpu-ab: run refused: arm 01-w-exp (exp): output fingerprint 0a798883e8eb differs from the control's 99d15f9a6f85 (arm 00-w-ctl); refusing to time a run whose output changed. Compare <run>/arms/01-w-exp/out with the control arm's out/
```

`--adapter` takes a file path, or a name looked up through `GPU_AB_ADAPTERS`, a path pattern in which `{name}` stands for the name. The default is `~/.config/gpu-ab/adapters/{name}.toml` (under `$XDG_CONFIG_HOME` when that is set); a pattern such as `~/ops/{name}/gpu-ab.toml` keeps each adapter beside other per-repo files. Other options: `--pairs`, `--warmup`, `--ctl-env KEY=VALUE` and `--exp-env KEY=VALUE` for per-arm environment, and `--resource`, `--priority` and `--timeout` for `job.json`. See `gpu-ab prepare --help`.

A run directory runs once. `gpu-ab run` refuses a directory that already has arms; prepare a new one instead of re-running a settled case.

## Queue integration

gpu-ab never talks to a scheduler, and needs none: the run is one command. Whatever runs it must give it the GPU to itself for the whole run, since every arm runs back to back so that pairs share one GPU state, and must not stop and restart it, since a resumed run would mix GPU states.

For the homebased GPU queue, `prepare` writes `job.json`, a one-step job spec that runs the frozen `gpu-ab run <run>` with `preempt` set to wait, and prints its submit command, `homebased --json resource job submit --job-id <uuid> --spec <run>/job.json`, with a job id chosen up front so a lost reply can be retried. `--resource` (default `gpu0`), `--priority` (default `medium`) and `--timeout` (an output-inactivity reminder, default `1h`) fill it in. When `HOMEBASED_JOB_ID` and `HOMEBASED_RESOURCE` are set, `result.json` records them. Other schedulers can ignore `job.json` and run the printed command as their job's only step.

## Apple silicon

On macOS, gpu-ab reads the GPU without root. Before and after every arm it records:

- every Metal process's cumulative GPU time, from `ioreg`'s per-client `accumulatedGPUTime`, and the instantaneous `Device Utilization %`;
- GPU performance-state residency and GPU and Neural Engine energy, from IOReport, the counters behind `powermetrics`;
- the thermal pressure level (`notifyutil -g com.apple.system.thermalpressurelevel`) and the power mode (`pmset -g`).

`result.json` keeps the raw readings in each snapshot's `apple` field, and each arm's `apple` field holds the difference: other processes' GPU time and its share of the arm's wall time, WindowServer's share, the GPU's active share and mean frequency (from the P-state table in `ioreg`'s power manager), and GPU and Neural Engine power. The `Env` line of the log entry summarizes them across arms.

**Other GPU users.** A GPU queue can't see iOS simulators, browsers or other projects' Core ML jobs, and CPU load doesn't show them: one pushed a Core ML call from 168 to 250 ms. An arm whose bench used the GPU is invalid when processes other than the bench together used the GPU for at least 1% of the time between the `ioreg` readings around it, which is the arm plus about 50 ms.

The rule guards GPU arms only. While each arm runs, gpu-ab also reads the Metal clients of the bench and its descendants, after 100 ms and then at doubling intervals up to once a second, about 20 ms of one core per reading in both arms. These readings are needed because a bench's clients close when it exits, before the reading after the arm. The bench's own GPU time, as a share of the same window, goes in the arm's `apple.own_share`. Under 0.1% (`OWN_GPU_IDLE_SHARE`), the arm used no GPU, so it can't be slowed by other GPU users: their share is still recorded, but the arm stays valid, and the log's `Env` line counts such arms. This lets a CPU-only arm, such as a CPU baseline against a GPU port, run on a desktop Mac whose apps use the GPU now and then. An arm that ends before the first reading has no `own_share` and is checked as a GPU arm. GPU work in the bench's final second can be missed, which can't exempt a GPU arm whose work spans more than that. On NVIDIA hosts, any compute process before or after an arm still invalidates it, whatever the bench used. WindowServer is recorded but exempt, because it composites the display all the time and the apps that keep it busy have GPU clients of their own. GPU time is counted per Metal client, keyed by its registry id, so a reused pid can't hide anything. A client's time is the sum over its open command queues, so closing a queue makes it go down; when that happens to a client that submitted work during the arm, how much it used can't be known, and the arm is invalid too. Apart from the bench's own clients, which the readings during the arm see, a client that closes before the second reading isn't seen, so another process that exits during the arm is missed.

**Noise.** Run-to-run medians drift 2-7% on the Mac, so measure A/A noise there before trusting the bar; pairing in one job cancels most of the drift. Each arm already runs in a fresh process. Thermal pressure above nominal and the power mode are recorded but never invalidate an arm, as NVIDIA throttle reasons are.

The [sam3rs Metal example adapter](docs/adapters/sam3rs-metal.toml) has its own name, `sam3rs-metal`, so its noise figures never apply to the CUDA benches; pass its path to `--adapter`.

## Adapter format

One TOML file per bench repo. It holds absolute, machine-specific paths, so keep it outside the bench repo, where `--adapter` or `GPU_AB_ADAPTERS` can find it. [docs/adapters/](docs/adapters/) has a [commented template](docs/adapters/example.toml) and real examples for [speakrs](docs/adapters/speakrs.toml), [scriptrs](docs/adapters/scriptrs.toml), [sam3rs](docs/adapters/sam3rs.toml) and [sam3rs on Metal](docs/adapters/sam3rs-metal.toml); replace their `/path/to/` paths with your own.

```toml
name = "scriptrs"                       # used in job names and log entries
repo = "/path/to/scriptrs"              # absolute; {repo}, and the default working directory

[bench.ls-b1]                           # one table per bench, picked with --bench
description = "LibriSpeech test-clean b1 FP16 v3, warm pass wall, identical transcripts"
argv = ["{bin}", "--models", "/path/to/models/scriptrs-native", "--version", "v3",
        "--repeats", "2", "--out", "{out}/ls.json"]   # no shell; must use {bin}
# cwd = "{repo}"                        # optional
# env = { RUST_LOG = "info" }           # optional, on top of the inherited environment

# the timed number: name, unit, which direction is better, and where to find it
metric = { name = "LS b1 warm wall", unit = "s", better = "lower", file = "{out}/ls.json", json = "/walls_s/1" }

# the output check: hash, gate, or both
fingerprint = { hash = ["sh", "-c", "sha256sum < {out}/ls.texts.json"] }
```

**Placeholders**, usable in `argv`, `cwd`, `env` values, `metric.file` and the fingerprint commands:

| Placeholder | Expands to |
|---|---|
| `{bin}` | the frozen binary for this arm |
| `{out}` | an empty per-arm output directory |
| `{arm_dir}` | the per-arm directory holding `stdout.txt`, `stderr.txt` and `out/` |
| `{stdout}` | the arm's captured stdout |
| `{ctl_out}` | the reference control arm's `out/`, for gates that compare |
| `{run_dir}` | the run directory |
| `{repo}` | the adapter's `repo` |

Only `{name}` with lowercase letters and underscores is a placeholder, and `${NAME}` is left alone, so shell snippets pass through. An unknown placeholder fails at prepare time.

**Metric.** `file` defaults to `{stdout}` and may be a glob that matches exactly one file. Read the value with either `regex`, whose first capture group is the number, or `json`, a JSON pointer such as `/implementations/cuda/measurement/time_seconds`. `better` is `lower` or `higher`. The value must be a positive, finite number.

**Fingerprint.**

- `hash`: a command whose trimmed stdout identifies the output. Every arm must match the first control arm. Keep paths out of its output: `sha256sum FILE` prints the per-arm path, so use `sha256sum < FILE` or `cat ... | sha256sum`. Empty output, or the sha256 of empty input (a glob that matched nothing), is an error.
- `gate`: a command that must exit 0, typically comparing `{out}` with `{ctl_out}` against accuracy limits. Use a gate rather than a hash for changes to arithmetic, where outputs may legitimately differ within limits. The speakrs example's `*-der` benches use one.

**What is frozen.** Besides itself, gpu-ab copies only the files passed as `--ctl` and `--exp`, and the sha256 in `plan.json` covers only those files. When `{bin}` is a launcher script, such as one that starts a model server and drives a load generator against it, the script is frozen but the programs, models and configs it calls are not. Point it at copies that stay unchanged for the whole run, such as read-only or content-addressed paths, and have it write their hashes or digests into `{out}` so every arm records what it actually ran. Recording a version alone doesn't stop a file from changing between arms.

## Run directory

```
plan.json         resolved bench, frozen binaries with sha256 and source commits, settings
job.json          optional homebased queue spec
bin/              control, experiment, gpu-ab
arms/NN-p<k>-ctl/ stdout.txt, stderr.txt, fingerprint.txt, gate.txt, out/
result.json       environment, every arm with GPU snapshots, pairs, statistics, noise, verdict
LOG-entry.md      draft LOG.md entry
```

`result.json` is rewritten after every arm while the run is in progress (status `running`), so a killed job leaves a partial record. A refused run has status `failed`, the error, and no statistics.

## Development

```sh
just ci   # fmt check, clippy, tests; what CI runs
```

To release: commit, `just bump patch` (or `minor`, `major`), commit the bump, push, then `just release` and `just release-watch`. The Release workflow publishes a static `x86_64-unknown-linux-musl` binary, an `aarch64-apple-darwin` binary and `install.sh`.

[AGENTS.md](AGENTS.md) lists what gpu-ab deliberately leaves out and where those needs belong instead.

### Tests

Unit tests cover the t-interval, the verdict rule, ABBA order and placeholder expansion. `crates/gpu-ab/tests/end_to_end.rs` runs the CLI against a fake bench script and a fake `nvidia-smi` (Linux) or `ioreg` (macOS), with no GPU: an A/A run, an A/B run that accepts a clear win, a broken experiment refused by its hash, one refused by its gate, and a busy GPU that invalidates every arm. On macOS they also check that another GPU user invalidates only the arms whose bench used the GPU itself, through a child process as a real bench's worker does, and that an arm too short to read is still checked. CI runs them on both.
