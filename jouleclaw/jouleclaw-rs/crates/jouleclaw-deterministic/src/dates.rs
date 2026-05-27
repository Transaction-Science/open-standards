//! Date-arithmetic primitives. All dates are parsed in ISO-8601
//! `YYYY-MM-DD` form. Internally uses [`chrono::NaiveDate`] so leap
//! years and Gregorian calendar quirks are correct.

use chrono::{Datelike, Duration, NaiveDate, Weekday};
use jouleclaw_cascade::LawfulPrimitive;
use std::sync::Arc;

pub fn primitives() -> Vec<Arc<dyn LawfulPrimitive>> {
    vec![
        Arc::new(DayOfWeek),
        Arc::new(DaysBetween),
        Arc::new(IsLeapYear),
        Arc::new(IsoToWeekday),
        Arc::new(AddDays),
    ]
}

fn strip_prefix_ci<'a>(q: &'a str, prefix: &str) -> Option<&'a str> {
    let q = q.trim();
    if q.len() < prefix.len() {
        return None;
    }
    let (head, tail) = q.split_at(prefix.len());
    if !head.eq_ignore_ascii_case(prefix) {
        return None;
    }
    let rest = tail.strip_prefix(|c: char| c.is_whitespace())?;
    Some(rest.trim())
}

fn parse_date(s: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()
}

fn weekday_name(w: Weekday) -> &'static str {
    match w {
        Weekday::Mon => "Monday",
        Weekday::Tue => "Tuesday",
        Weekday::Wed => "Wednesday",
        Weekday::Thu => "Thursday",
        Weekday::Fri => "Friday",
        Weekday::Sat => "Saturday",
        Weekday::Sun => "Sunday",
    }
}

// ---- day of week --------------------------------------------------------

pub struct DayOfWeek;
impl LawfulPrimitive for DayOfWeek {
    fn id(&self) -> &str {
        "lawful:dates:day-of-week"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "day of week")?;
        let d = parse_date(rest)?;
        Some(weekday_name(d.weekday()).to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        100
    }
}

// ---- days between -------------------------------------------------------

pub struct DaysBetween;
impl LawfulPrimitive for DaysBetween {
    fn id(&self) -> &str {
        "lawful:dates:days-between"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "days between")?;
        // Expect "<date-a> and <date-b>"
        let lower = rest.to_ascii_lowercase();
        let idx = lower.find(" and ")?;
        let a_str = &rest[..idx];
        let b_str = &rest[idx + " and ".len()..];
        let a = parse_date(a_str)?;
        let b = parse_date(b_str)?;
        let delta = (b - a).num_days();
        Some(delta.abs().to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        110
    }
}

// ---- is leap year -------------------------------------------------------

pub struct IsLeapYear;
impl LawfulPrimitive for IsLeapYear {
    fn id(&self) -> &str {
        "lawful:dates:is-leap-year"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "is leap year")?;
        let y: i32 = rest.trim().parse().ok()?;
        // Same rule chrono uses.
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        Some(if leap { "true".into() } else { "false".into() })
    }
    fn declared_cost_uj(&self) -> u64 {
        80
    }
}

// ---- iso weekday number -------------------------------------------------

pub struct IsoToWeekday;
impl LawfulPrimitive for IsoToWeekday {
    fn id(&self) -> &str {
        "lawful:dates:iso-weekday"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "iso weekday")?;
        let d = parse_date(rest)?;
        // ISO: Monday = 1 .. Sunday = 7
        Some(d.weekday().number_from_monday().to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        100
    }
}

// ---- add days -----------------------------------------------------------

pub struct AddDays;
impl LawfulPrimitive for AddDays {
    fn id(&self) -> &str {
        "lawful:dates:add-days"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "add")?;
        // "<N> days to <YYYY-MM-DD>"
        let lower = rest.to_ascii_lowercase();
        let days_idx = lower.find(" days to ")?;
        let n_str = &rest[..days_idx];
        let date_str = &rest[days_idx + " days to ".len()..];
        let n: i64 = n_str.trim().parse().ok()?;
        let d = parse_date(date_str)?;
        let result = d.checked_add_signed(Duration::days(n))?;
        Some(result.format("%Y-%m-%d").to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        110
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_of_week_works() {
        // 2024-01-01 was a Monday.
        assert_eq!(DayOfWeek.try_resolve("day of week 2024-01-01").as_deref(), Some("Monday"));
        // 2000-01-01 was a Saturday.
        assert_eq!(DayOfWeek.try_resolve("day of week 2000-01-01").as_deref(), Some("Saturday"));
    }

    #[test]
    fn days_between_works() {
        assert_eq!(
            DaysBetween.try_resolve("days between 2024-01-01 and 2024-01-31").as_deref(),
            Some("30")
        );
        assert_eq!(
            DaysBetween.try_resolve("days between 2024-12-31 and 2024-01-01").as_deref(),
            Some("365")
        );
    }

    #[test]
    fn is_leap_year_works() {
        assert_eq!(IsLeapYear.try_resolve("is leap year 2024").as_deref(), Some("true"));
        assert_eq!(IsLeapYear.try_resolve("is leap year 2023").as_deref(), Some("false"));
        assert_eq!(IsLeapYear.try_resolve("is leap year 2000").as_deref(), Some("true"));
        assert_eq!(IsLeapYear.try_resolve("is leap year 1900").as_deref(), Some("false"));
    }

    #[test]
    fn iso_weekday_works() {
        assert_eq!(IsoToWeekday.try_resolve("iso weekday 2024-01-01").as_deref(), Some("1"));
        assert_eq!(IsoToWeekday.try_resolve("iso weekday 2024-01-07").as_deref(), Some("7"));
    }

    #[test]
    fn add_days_works() {
        assert_eq!(
            AddDays.try_resolve("add 7 days to 2024-01-01").as_deref(),
            Some("2024-01-08")
        );
        assert_eq!(
            AddDays.try_resolve("add -1 days to 2024-01-01").as_deref(),
            Some("2023-12-31")
        );
    }

    #[test]
    fn malformed_returns_none() {
        assert!(DayOfWeek.try_resolve("day of week not-a-date").is_none());
        assert!(DaysBetween.try_resolve("days between 2024-01-01 to 2024-01-31").is_none());
        assert!(IsLeapYear.try_resolve("is leap year xyz").is_none());
        assert!(AddDays.try_resolve("add many days to 2024-01-01").is_none());
        assert!(DayOfWeek.try_resolve("what day is it").is_none());
    }

    #[test]
    fn category_count() {
        assert_eq!(primitives().len(), 5);
    }
}
