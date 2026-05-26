//! Persistent calibration. Survives runtime restart.
//!
//! Format mirrors the disk-history layout:
//!
//!   header  8 bytes  "JCAL0001"
//!   for each cell:
//!     record_len  u32 BE  (length of the following record)
//!     cell_id     u16 BE
//!     samples     u64 BE
//!     ratio_sum   f64 BE
//!     max_ratio   f64 BE
//!     violations  u64 BE
//!     total_est   f64 BE
//!     total_act   f64 BE
//!
//! Reading and writing are simple linear passes; the file is small
//! (~50 bytes per cell, ~10 KB for full cascade coverage of the
//! 8,000 cells).

use crate::calibration::{CalibrationReport, TierCalibration};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"JCAL0001";
const RECORD_BYTES: usize = 2 + 8 + 8 + 8 + 8 + 8 + 8;  // 50 bytes

#[derive(Debug)]
pub enum DiskCalibrationError {
    Io(std::io::Error),
    BadMagic,
    Truncated,
    BadRecord(String),
}

impl std::fmt::Display for DiskCalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {}", e),
            Self::BadMagic => write!(f, "bad magic — not a JCAL0001 file"),
            Self::Truncated => write!(f, "truncated calibration file"),
            Self::BadRecord(s) => write!(f, "bad record: {}", s),
        }
    }
}

impl std::error::Error for DiskCalibrationError {}

impl From<std::io::Error> for DiskCalibrationError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

/// Write a full calibration report to disk. Overwrites the file.
/// The report's `per_cell` map is the source of truth; `per_tier`
/// is NOT persisted (it can be reconstructed if `TierId` semantics
/// are stable, but we prefer keeping the on-disk format keyed on
/// the stable coordinate cell IDs).
pub fn save<P: AsRef<Path>>(
    path: P,
    report: &CalibrationReport,
) -> Result<(), DiskCalibrationError> {
    let mut f = OpenOptions::new()
        .create(true).write(true).truncate(true)
        .open(path)?;
    f.write_all(MAGIC)?;

    let mut cells: Vec<_> = report.per_cell.iter().collect();
    cells.sort_by_key(|(c, _)| *c);

    for (cell_id, cal) in cells {
        // Build the record bytes.
        let mut rec = Vec::with_capacity(RECORD_BYTES);
        rec.extend_from_slice(&cell_id.to_be_bytes());
        rec.extend_from_slice(&cal.samples.to_be_bytes());
        rec.extend_from_slice(&cal.ratio_sum.to_be_bytes());
        rec.extend_from_slice(&cal.max_ratio.to_be_bytes());
        rec.extend_from_slice(&cal.budget_violations.to_be_bytes());
        rec.extend_from_slice(&cal.total_estimated.to_be_bytes());
        rec.extend_from_slice(&cal.total_actual.to_be_bytes());

        // Length-prefix.
        f.write_all(&(rec.len() as u32).to_be_bytes())?;
        f.write_all(&rec)?;
    }
    f.flush()?;
    Ok(())
}

/// Load a calibration report from disk. Returns an empty report if
/// the file doesn't exist (first-run case).
pub fn load<P: AsRef<Path>>(
    path: P,
) -> Result<CalibrationReport, DiskCalibrationError> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(CalibrationReport::default());
    }

    let mut f = File::open(path)?;

    // Magic.
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(DiskCalibrationError::BadMagic);
    }

    let mut report = CalibrationReport::default();

    loop {
        let mut len_buf = [0u8; 4];
        match f.read_exact(&mut len_buf) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_be_bytes(len_buf) as usize;
        if len != RECORD_BYTES {
            return Err(DiskCalibrationError::BadRecord(
                format!("expected {} bytes, got {}", RECORD_BYTES, len)));
        }

        let mut rec = vec![0u8; len];
        f.read_exact(&mut rec).map_err(|_| DiskCalibrationError::Truncated)?;

        let cell_id = u16::from_be_bytes([rec[0], rec[1]]);
        let samples = u64::from_be_bytes(rec[2..10].try_into().unwrap());
        let ratio_sum = f64::from_be_bytes(rec[10..18].try_into().unwrap());
        let max_ratio = f64::from_be_bytes(rec[18..26].try_into().unwrap());
        let violations = u64::from_be_bytes(rec[26..34].try_into().unwrap());
        let total_est = f64::from_be_bytes(rec[34..42].try_into().unwrap());
        let total_act = f64::from_be_bytes(rec[42..50].try_into().unwrap());

        let cal = TierCalibration {
            samples,
            ratio_sum,
            max_ratio,
            budget_violations: violations,
            total_estimated: total_est,
            total_actual: total_act,
        };
        report.per_cell.insert(cell_id, cal);
    }

    Ok(report)
}

/// A `PersistentCalibration` is a `CalibrationReport` plus a file
/// path. It loads on construction and saves explicitly. Use this
/// when you want calibration data to survive across runtime restarts.
pub struct PersistentCalibration {
    pub report: CalibrationReport,
    pub path: PathBuf,
}

impl PersistentCalibration {
    /// Open a calibration file. If the file exists, load it; if not,
    /// start fresh. Either way the file is created on first save.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, DiskCalibrationError> {
        let report = load(&path)?;
        Ok(Self { report, path: path.as_ref().to_path_buf() })
    }

    pub fn save(&self) -> Result<(), DiskCalibrationError> {
        save(&self.path, &self.report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TierId, L1Primitive};
    use crate::coord::prebuilt;

    fn tmpfile(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("joule-{}-{}.cal",
            name, std::process::id()))
    }

    #[test]
    fn save_and_load_roundtrip() {
        let path = tmpfile("roundtrip");
        let _ = std::fs::remove_file(&path);

        let mut original = CalibrationReport::default();
        let l1 = prebuilt::l1_execute();
        let l4 = prebuilt::l4_frontier_model();
        for _ in 0..10 {
            original.record_with_coord(
                TierId::L1(L1Primitive::Execute), &l1, 1e-9, 1.2e-9);
        }
        for _ in 0..5 {
            original.record_with_coord(
                TierId::L4(crate::types::L4ModelId(0)), &l4, 0.5, 1.0);
        }

        save(&path, &original).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.per_cell.len(), 2);
        let l1_cell = loaded.per_cell.get(&l1.cell_id()).unwrap();
        assert_eq!(l1_cell.samples, 10);
        assert!((l1_cell.mean_ratio() - 1.2).abs() < 1e-9);

        let l4_cell = loaded.per_cell.get(&l4.cell_id()).unwrap();
        assert_eq!(l4_cell.samples, 5);
        assert!((l4_cell.mean_ratio() - 2.0).abs() < 1e-9);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_missing_file_returns_empty() {
        let path = tmpfile("missing");
        let _ = std::fs::remove_file(&path);
        let report = load(&path).unwrap();
        assert!(report.per_cell.is_empty());
    }

    #[test]
    fn load_bad_magic_errors() {
        let path = tmpfile("bad_magic");
        std::fs::write(&path, b"NOTJOULE").unwrap();
        match load(&path) {
            Err(DiskCalibrationError::BadMagic) => {}
            other => panic!("expected BadMagic, got {:?}", other),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn persistent_calibration_lifecycle() {
        let path = tmpfile("lifecycle");
        let _ = std::fs::remove_file(&path);

        // Open empty, record, save.
        {
            let mut pcal = PersistentCalibration::open(&path).unwrap();
            assert!(pcal.report.per_cell.is_empty());
            let l1 = prebuilt::l1_execute();
            for _ in 0..3 {
                pcal.report.record_with_coord(
                    TierId::L1(L1Primitive::Execute), &l1, 1e-9, 1.5e-9);
            }
            pcal.save().unwrap();
        }

        // Reopen, expect data present.
        {
            let pcal = PersistentCalibration::open(&path).unwrap();
            let l1 = prebuilt::l1_execute();
            assert_eq!(pcal.report.per_cell.len(), 1);
            let mu = pcal.report.learned_mu(&l1);
            assert!((mu - 1.5).abs() < 1e-9, "loaded μ = {}", mu);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_report_writes_just_the_magic() {
        let path = tmpfile("empty");
        let _ = std::fs::remove_file(&path);

        let empty = CalibrationReport::default();
        save(&path, &empty).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[..], MAGIC);
        let _ = std::fs::remove_file(&path);
    }
}
