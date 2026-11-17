//! Per-repo adapter files: how to run one bench arm, read its metric and fingerprint its
//! output
//!
//! An adapter is a TOML file with a `repo` checkout and one `[bench.<name>]` table per
//! bench. Strings in `argv`, `cwd`, `env`, `metric.file` and the fingerprint commands may
//! use the placeholders listed in [`PLACEHOLDERS`]

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use eyre::{Result, WrapErr, bail, eyre};
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Placeholders that adapter strings may use, with what each expands to
pub const PLACEHOLDERS: &[(&str, &str)] = &[
    ("bin", "frozen binary for this arm"),
    ("out", "empty per-arm output directory"),
    ("arm_dir", "per-arm directory holding stdout.txt, stderr.txt and out/"),
    ("stdout", "the arm's captured stdout file"),
    ("ctl_out", "output directory of the reference control arm, for gates"),
    ("run_dir", "the run directory"),
    ("repo", "the adapter's repo checkout"),
];

/// One repo's adapter file
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Adapter {
    /// Short repo name used in job names and log entries, such as `speakrs`
    pub name: String,
    /// Absolute path of the repo checkout, available as `{repo}`
    pub repo: PathBuf,
    /// Benches by name
    pub bench: BTreeMap<String, Bench>,
}

/// How to run and score one bench arm
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bench {
    /// One line shown in log entries
    #[serde(default)]
    pub description: Option<String>,
    /// Command for one arm, run without a shell; must reference `{bin}`
    pub argv: Vec<String>,
    /// Working directory, `{repo}` when unset
    #[serde(default)]
    pub cwd: Option<String>,
    /// Extra environment for every arm, on top of the inherited environment
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub metric: MetricSpec,
    pub fingerprint: FingerprintSpec,
}

/// Where to read the timed metric from after an arm exits
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricSpec {
    /// Name shown in reports, such as `wall` or `RTFx`
    pub name: String,
    /// Unit shown in reports, such as `s` or `ms`
    #[serde(default)]
    pub unit: Option<String>,
    /// Whether a smaller or a larger value is a win
    pub better: Better,
    /// File to read, possibly a glob that must match exactly one file
    #[serde(default = "default_metric_file")]
    pub file: String,
    /// Regex whose first capture group is the value
    #[serde(default)]
    pub regex: Option<String>,
    /// JSON pointer (RFC 6901) to a number in the file
    #[serde(default)]
    pub json: Option<String>,
}

fn default_metric_file() -> String {
    "{stdout}".to_owned()
}

/// Direction of improvement for a metric
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Better {
    Lower,
    Higher,
}

impl Better {
    /// Per-pair speed ratio, above 1 when the experiment wins
    pub fn ratio(self, control: f64, experiment: f64) -> f64 {
        match self {
            Better::Lower => control / experiment,
            Better::Higher => experiment / control,
        }
    }
}

/// How each arm's output is checked against the control's
///
/// With `hash`, the command's trimmed stdout must equal the reference control arm's. With
/// `gate`, the command must exit 0. Both may be set; at least one must be
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FingerprintSpec {
    #[serde(default)]
    pub hash: Option<Vec<String>>,
    #[serde(default)]
    pub gate: Option<Vec<String>>,
}

impl Adapter {
    /// Reads and validates an adapter file
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).wrap_err_with(|| format!("reading adapter {}", path.display()))?;
        let adapter: Adapter = toml::from_str(&text).wrap_err_with(|| format!("parsing adapter {}", path.display()))?;
        adapter
            .validate()
            .wrap_err_with(|| format!("invalid adapter {}", path.display()))?;
        Ok(adapter)
    }

    /// Picks a bench by name, or the only one when `name` is `None`
    pub fn bench(&self, name: Option<&str>) -> Result<(&str, &Bench)> {
        match name {
            Some(name) => self
                .bench
                .get_key_value(name)
                .map(|(k, v)| (k.as_str(), v))
                .ok_or_else(|| {
                    eyre!(
                        "no bench {name:?} in adapter {}; have {:?}",
                        self.name,
                        self.bench.keys()
                    )
                }),
            None if self.bench.len() == 1 => {
                let (k, v) = self.bench.iter().next().expect("one bench");
                Ok((k.as_str(), v))
            }
            None => bail!(
                "adapter {} has several benches, pick one with --bench: {:?}",
                self.name,
                self.bench.keys()
            ),
        }
    }

    fn validate(&self) -> Result<()> {
        if !self.repo.is_absolute() {
            bail!("repo must be an absolute path, got {}", self.repo.display());
        }

        if self.bench.is_empty() {
            bail!("no [bench.<name>] tables");
        }

        for (name, bench) in &self.bench {
            bench.validate().wrap_err_with(|| format!("bench {name}"))?;
        }

        Ok(())
    }
}

impl Bench {
    fn validate(&self) -> Result<()> {
        if self.argv.is_empty() || self.argv[0].is_empty() {
            bail!("argv must name a program");
        }

        if !self.argv.iter().any(|a| a.contains("{bin}")) {
            bail!("argv must run the frozen binary through {{bin}}");
        }

        match (&self.metric.regex, &self.metric.json) {
            (Some(re), None) => {
                let re = Regex::new(re).wrap_err("metric.regex")?;
                if re.captures_len() < 2 {
                    bail!("metric.regex needs a capture group for the value");
                }
            }
            (None, Some(ptr)) if ptr.is_empty() || ptr.starts_with('/') => {}
            (None, Some(ptr)) => bail!("metric.json must be a JSON pointer starting with '/', got {ptr:?}"),
            _ => bail!("set exactly one of metric.regex and metric.json"),
        }

        let fp = &self.fingerprint;
        if fp.hash.is_none() && fp.gate.is_none() {
            bail!("fingerprint needs a hash command, a gate command or both");
        }

        for cmd in [&fp.hash, &fp.gate].into_iter().flatten() {
            if cmd.is_empty() || cmd[0].is_empty() {
                bail!("fingerprint commands must name a program");
            }
        }

        // expand every templated string once so typos in placeholders fail at prepare time
        let vars = Vars::dummy();
        let templated = self
            .argv
            .iter()
            .chain(self.cwd.iter())
            .chain(self.env.values())
            .chain([&self.metric.file])
            .chain(fp.hash.iter().flatten())
            .chain(fp.gate.iter().flatten());
        for s in templated {
            vars.expand(s)?;
        }

        Ok(())
    }
}

/// Placeholder values for one arm
#[derive(Debug, Clone, Default)]
pub struct Vars(BTreeMap<&'static str, String>);

impl Vars {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a placeholder; panics on a name missing from [`PLACEHOLDERS`]
    pub fn set(&mut self, name: &'static str, value: impl Into<String>) -> &mut Self {
        assert!(
            PLACEHOLDERS.iter().any(|(n, _)| *n == name),
            "unknown placeholder {name}"
        );
        self.0.insert(name, value.into());
        self
    }

    /// Checks that a string expands, with every placeholder known
    pub fn check(s: &str) -> Result<()> {
        Self::dummy().expand(s).map(drop)
    }

    fn dummy() -> Self {
        let mut vars = Self::new();
        for (name, _) in PLACEHOLDERS {
            vars.set(name, "x");
        }

        vars
    }

    /// Replaces `{name}` placeholders
    ///
    /// Only `{` followed by lowercase letters or underscores and `}` is a placeholder, and
    /// `${name}` is left alone, so shell snippets such as `${HOME}` or awk's `{print $1}`
    /// pass through unchanged. An unknown or unset placeholder is an error
    pub fn expand(&self, s: &str) -> Result<String> {
        let mut out = String::with_capacity(s.len());
        let mut rest = s;
        while let Some(open) = rest.find('{') {
            let (head, tail) = rest.split_at(open);
            out.push_str(head);
            let name_len = tail[1..]
                .find(|c: char| !(c.is_ascii_lowercase() || c == '_'))
                .unwrap_or(tail.len() - 1);
            let name = &tail[1..1 + name_len];
            let closes = tail[1 + name_len..].starts_with('}');
            if name.is_empty() || !closes || head.ends_with('$') {
                out.push('{');
                rest = &tail[1..];
                continue;
            }

            let value = self.0.get(name).ok_or_else(|| {
                let known: Vec<&str> = PLACEHOLDERS.iter().map(|(n, _)| *n).collect();
                eyre!("unknown placeholder {{{name}}} in {s:?}; known: {known:?}")
            })?;
            out.push_str(value);
            rest = &tail[name_len + 2..];
        }

        out.push_str(rest);
        Ok(out)
    }

    /// Expands every element of an argv
    pub fn expand_all(&self, argv: &[String]) -> Result<Vec<String>> {
        argv.iter().map(|a| self.expand(a)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_keeps_shell_braces() {
        let mut vars = Vars::new();
        vars.set("out", "/r/a");
        let got = vars.expand("cat {out}/*.rttm ${HOME} | awk '{print $1}' {}").unwrap();
        assert_eq!(got, "cat /r/a/*.rttm ${HOME} | awk '{print $1}' {}");
        assert!(vars.expand("{outt}").is_err());
    }
}
