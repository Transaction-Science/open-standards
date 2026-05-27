//! Bulk-load LUT entries from TOML or CSV.
//!
//! TOML schema (file-level):
//! ```toml
//! [[entries]]
//! input      = "gcd 12 8"
//! output     = "4"
//! source_tag = "lawful:gcd"
//! cost_uj    = 1            # optional
//!
//! [[entries]]
//! input      = "gcd 100 75"
//! output     = "25"
//! source_tag = "lawful:gcd"
//! ```
//!
//! CSV schema: header row `input,output,source_tag,cost_uj` with
//! `cost_uj` optional (empty cell or column omitted both fall back to
//! [`crate::lut::DEFAULT_DECLARED_COST_UJ`]).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::lut::{Lut, DEFAULT_DECLARED_COST_UJ};
use crate::types::LutError;

/// On-disk shape of a single TOML LUT entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TomlEntry {
    pub input: String,
    pub output: String,
    #[serde(default)]
    pub cost_uj: Option<u64>,
    pub source_tag: String,
}

/// On-disk shape of a TOML LUT file: `[[entries]]` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TomlFile {
    #[serde(default)]
    pub entries: Vec<TomlEntry>,
}

impl Lut {
    /// Bulk-load entries from a TOML file. Returns a fresh [`Lut`]
    /// containing exactly the entries declared.
    pub fn load_toml<P: AsRef<Path>>(path: P) -> Result<Self, LutError> {
        let text = std::fs::read_to_string(path)?;
        let parsed: TomlFile = toml::from_str(&text)?;
        let mut lut = Lut::new();
        for entry in parsed.entries {
            let cost = entry.cost_uj.unwrap_or(DEFAULT_DECLARED_COST_UJ);
            lut.register_with_cost(&entry.input, entry.output, cost, entry.source_tag);
        }
        Ok(lut)
    }

    /// Bulk-load entries from a CSV file with header row
    /// `input,output,source_tag,cost_uj`. The `cost_uj` column may be
    /// omitted entirely, or present but empty on a per-row basis.
    pub fn load_csv<P: AsRef<Path>>(path: P) -> Result<Self, LutError> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(path)?;

        // Resolve column indices from the header so the CSV can be
        // written in either `input,output,source_tag,cost_uj` or
        // `input,output,source_tag` (cost_uj omitted) form.
        let headers = rdr.headers()?.clone();
        let idx_input = column_index(&headers, "input")
            .ok_or(LutError::CsvMissingColumn("input"))?;
        let idx_output = column_index(&headers, "output")
            .ok_or(LutError::CsvMissingColumn("output"))?;
        let idx_source = column_index(&headers, "source_tag")
            .ok_or(LutError::CsvMissingColumn("source_tag"))?;
        let idx_cost = column_index(&headers, "cost_uj");

        let mut lut = Lut::new();
        for record in rdr.records() {
            let record = record?;
            let input = record.get(idx_input).unwrap_or("");
            let output = record.get(idx_output).unwrap_or("");
            let source = record.get(idx_source).unwrap_or("");
            let cost = if let Some(i) = idx_cost {
                match record.get(i) {
                    None | Some("") => DEFAULT_DECLARED_COST_UJ,
                    Some(raw) => raw
                        .parse::<u64>()
                        .map_err(|_| LutError::CsvBadCost(raw.to_string()))?,
                }
            } else {
                DEFAULT_DECLARED_COST_UJ
            };
            lut.register_with_cost(input, output.as_bytes().to_vec(), cost, source);
        }
        Ok(lut)
    }
}

fn column_index(headers: &csv::StringRecord, name: &str) -> Option<usize> {
    headers.iter().position(|h| h == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tempfile_with(name: &str, body: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("jouleclaw-lut-test-{}-{}", std::process::id(), name));
        let mut f = std::fs::File::create(&path).expect("create tempfile");
        f.write_all(body.as_bytes()).expect("write tempfile");
        path
    }

    #[test]
    fn toml_round_trip() {
        let toml_body = r#"
[[entries]]
input = "gcd 12 8"
output = "4"
source_tag = "lawful:gcd"
cost_uj = 7

[[entries]]
input = "  Greet  "
output = "hello"
source_tag = "smartbyte:greeting"
"#;
        let path = tempfile_with("round_trip.toml", toml_body);
        let lut = Lut::load_toml(&path).expect("load toml");
        let _ = std::fs::remove_file(&path);

        assert_eq!(lut.len(), 2);
        let h = lut.try_lookup("gcd 12 8").expect("gcd hits");
        assert_eq!(h.output, b"4");
        assert_eq!(h.declared_cost_uj, 7);
        assert_eq!(h.source_tag, "lawful:gcd");
        // Normalisation: `"  Greet  "` and `"greet"` collide.
        let g = lut.try_lookup("greet").expect("greet hits");
        assert_eq!(g.output, b"hello");
        assert_eq!(g.declared_cost_uj, DEFAULT_DECLARED_COST_UJ);
    }

    #[test]
    fn toml_empty_file_loads() {
        let path = tempfile_with("empty.toml", "");
        let lut = Lut::load_toml(&path).expect("empty toml loads");
        let _ = std::fs::remove_file(&path);
        assert!(lut.is_empty());
    }

    #[test]
    fn csv_round_trip_with_cost() {
        let body = "input,output,source_tag,cost_uj\n\
                    gcd 12 8,4,lawful:gcd,7\n\
                    greet,hello,smartbyte:greeting,\n";
        let path = tempfile_with("round_trip.csv", body);
        let lut = Lut::load_csv(&path).expect("load csv");
        let _ = std::fs::remove_file(&path);

        assert_eq!(lut.len(), 2);
        let h = lut.try_lookup("gcd 12 8").expect("gcd hits");
        assert_eq!(h.output, b"4");
        assert_eq!(h.declared_cost_uj, 7);
        let g = lut.try_lookup("greet").expect("greet hits");
        assert_eq!(g.output, b"hello");
        assert_eq!(g.declared_cost_uj, DEFAULT_DECLARED_COST_UJ);
    }

    #[test]
    fn csv_without_cost_column() {
        let body = "input,output,source_tag\n\
                    a,1,src\n\
                    b,2,src\n";
        let path = tempfile_with("no_cost.csv", body);
        let lut = Lut::load_csv(&path).expect("load csv without cost_uj");
        let _ = std::fs::remove_file(&path);
        assert_eq!(lut.len(), 2);
        let a = lut.try_lookup("a").expect("a hits");
        assert_eq!(a.declared_cost_uj, DEFAULT_DECLARED_COST_UJ);
    }

    #[test]
    fn csv_missing_required_column_errors() {
        let body = "input,output\n\
                    a,1\n";
        let path = tempfile_with("missing.csv", body);
        let res = Lut::load_csv(&path);
        let _ = std::fs::remove_file(&path);
        match res {
            Err(LutError::CsvMissingColumn("source_tag")) => {}
            other => panic!("expected CsvMissingColumn(source_tag), got {:?}", other),
        }
    }

    #[test]
    fn csv_bad_cost_errors() {
        let body = "input,output,source_tag,cost_uj\n\
                    a,1,src,not-a-number\n";
        let path = tempfile_with("bad_cost.csv", body);
        let res = Lut::load_csv(&path);
        let _ = std::fs::remove_file(&path);
        match res {
            Err(LutError::CsvBadCost(s)) => assert_eq!(s, "not-a-number"),
            other => panic!("expected CsvBadCost, got {:?}", other),
        }
    }
}
