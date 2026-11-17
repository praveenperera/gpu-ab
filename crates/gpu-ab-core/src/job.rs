//! Homebased queue job spec for a prepared run
//!
//! gpu-ab doesn't talk to the queue. It writes a one-step spec that runs every arm back
//! to back inside a single job, and prints the submit command

use std::path::Path;

use serde_json::{Value, json};

use crate::plan::{Mode, Plan};

pub const JOB_FILE: &str = "job.json";

/// Spec for `homebased resource job submit`, following `homebased resource schema`
pub fn spec(plan: &Plan, run_dir: &Path) -> Value {
    let mode = match plan.mode {
        Mode::AA => "A/A",
        Mode::AB => "A/B",
    };

    // the schema caps names at 120 characters
    let name: String = format!("gpu-ab {} {} {} {mode}", plan.adapter_name, plan.bench_name, plan.label)
        .chars()
        .take(120)
        .collect();

    json!({
        "api_version": 1,
        "name": name,
        "cwd": run_dir,
        "priority": plan.job.priority,
        // a timing run must never be cut short and restarted
        "preempt": {"mode": "wait"},
        "resource": plan.job.resource,
        "timeout": plan.job.timeout,
        "steps": [{"type": "task", "command": [plan.runner.path, "run", run_dir]}],
    })
}

/// Shell command that submits the run's job with its saved id
pub fn submit_command(plan: &Plan, run_dir: &Path) -> String {
    format!(
        "homebased --json resource job submit --job-id {} --spec {}",
        plan.job.id,
        shell_quote(&run_dir.join(JOB_FILE).display().to_string())
    )
}

/// Quotes `word` for a POSIX shell, leaving words made only of safe characters as they are
///
/// Printed commands are meant to be pasted, so a run directory with spaces or quotes must
/// stay one argument
pub fn shell_quote(word: &str) -> String {
    let safe = |c: char| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c);
    if !word.is_empty() && word.chars().all(safe) {
        return word.to_string();
    }

    format!("'{}'", word.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::shell_quote;

    #[test]
    fn quoted_paths_stay_one_shell_word() {
        assert_eq!(shell_quote("/runs/exp07"), "/runs/exp07");
        assert_eq!(shell_quote("/Users/Jane Doe/runs"), "'/Users/Jane Doe/runs'");
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
        assert_eq!(shell_quote(""), "''");
    }
}
