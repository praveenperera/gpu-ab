# Scope

gpu-ab does one thing: a paired timing comparison of two frozen binaries on a repo's own bench, with every arm's output checked, ending in ACCEPT, REJECT or INCONCLUSIVE under the rule in the README's "The verdict". Its value is that the procedure is fixed, so a verdict from one run means the same as a verdict from any other. Keep it narrow.

These are deliberately out of scope. When a hill-climb needs one, put it where the table says, not in gpu-ab:

| Wanted | Where it belongs |
|---|---|
| Several benches, guard workloads or variants in one job | Separate gpu-ab runs as separate steps of one queue job |
| Building the binaries | The caller's build step; freezing only means something if building stays outside |
| Submitting to a queue, waiting, retrying | The scheduler, such as homebased; gpu-ab only writes `job.json` and prints the commands |
| Per-layer timing, nsys, ncu, kernel counters | Each bench repo's own profiling helpers |
| Early stopping, resuming a killed run, extra pairs after the fact | Prepare a fresh run; a resumed run mixes GPU states |
| Re-confirming a winner | A second run |
| Accuracy suites, tolerances, reference math | The adapter's `gate` script |
| Analysis in Python or another language | Read `result.json`; no language bindings |
| Repo-specific logic | The repo's adapter TOML and gate scripts, never code in gpu-ab |

Changing the procedure needs Praveen's approval before any code: the ABBA schedule, warmup, `MIN_PAIRS`, the pair statistics, the verdict rule, freezing, the output checks, or what makes an arm invalid. The README's "How a run works" and "The verdict" document it and must change in the same change. Praveen's private GPU hill-climb playbook restates the rule (section 4 step 5), so say in the change that it needs the same update. Bug fixes that keep the procedure as documented don't need approval.

If a feature seems to need a new module, a plugin point, a daemon or a config layer, stop and ask instead.

# Adapters

Live adapters hold machine-specific paths, so they stay outside this repo; `GPU_AB_ADAPTERS` points at them. The files in `docs/adapters/` are public examples with `/path/to/` placeholders. When a live adapter they mirror changes in a way the README describes, update the example too, and never copy private paths, gate script locations or machine names into it.

# Checks

`just ci` runs fmt, clippy and the tests, as CI does. The end-to-end tests use a fake bench and a fake `nvidia-smi` on Linux or `ioreg` on macOS, so they need no GPU; CI runs them on both. Anything that does need a GPU goes through the homebased queue.

# Release

Commit, `just bump <part>`, commit the bump, push, `just release`, then `just release-watch`. The Release workflow builds a static x86_64 Linux binary and an arm64 macOS one and publishes them with `install.sh`.
