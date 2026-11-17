//! Timestamps in US Central time

use chrono::Utc;
use chrono_tz::America::Chicago;

/// Current time as RFC 3339 with the Central offset, for records
pub fn now_ct() -> String {
    Utc::now()
        .with_timezone(&Chicago)
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, false)
}

/// Current time as `YYYY-MM-DD HH:MM CT`, for log headings
pub fn heading_ct() -> String {
    format!("{} CT", Utc::now().with_timezone(&Chicago).format("%Y-%m-%d %H:%M"))
}

/// An RFC 3339 time as `YYYY-MM-DD HH:MM CT`
pub fn heading_from(rfc3339: &str) -> Option<String> {
    let t = chrono::DateTime::parse_from_rfc3339(rfc3339).ok()?;
    Some(format!("{} CT", t.with_timezone(&Chicago).format("%Y-%m-%d %H:%M")))
}

/// Current time as `YYYYMMDD-HHMM` in Central time, for directory names
pub fn stamp_ct() -> String {
    Utc::now().with_timezone(&Chicago).format("%Y%m%d-%H%M").to_string()
}
