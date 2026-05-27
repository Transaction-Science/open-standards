//! `DiskHistory` — append-only log file backing for the history layer.
//!
//! Format:
//!   header:  16 bytes magic+version
//!   record:  [u64 record_len BE] [record_bytes]
//!
//! Record layout (manually serialized, no external dep):
//!   [u8  input_variant]      0=Text, 1=Structured, 2=Binary
//!   [u32 input_len BE]
//!   [input_bytes]
//!   [16 bytes session_id]
//!   [32 bytes context_fingerprint]
//!   [u8  output_kind]        0=Text, 1=Structured, 2=Refused
//!   [u32 output_payload_len BE]
//!   [output_payload bytes]
//!   [u8  tier_family]        0=L0, 1=L1, 2=L2, 3=L3, 4=L4
//!   [u32 tier_data BE]       L1 primitive variant, or L2/L3/L4 model id
//!   [f64 joules]
//!   [f32 confidence]
//!   [u64 timestamp_secs BE]
//!   [u32 embedding_dim BE]
//!   [embedding_dim × f32 BE]
//!
//! On open, the file is read sequentially and the in-memory map
//! populated. Subsequent records append to the end. The format is
//! intentionally low-tech — a developer can dump it with hexdump.
//!
//! Two design properties matter:
//!   1. Append-only — every write is a syscall to extend the file, no
//!      seek-back. Crash-safety is "what was written before the crash
//!      is recoverable; what wasn't is lost." Acceptable for L0 cache.
//!   2. Same key-space as in-memory — the entry key from `key_for(q)`
//!      addresses both backends identically. A disk-backed instance
//!      and an in-memory one given the same queries record identical
//!      keys.

use jouleclaw_cascade::*;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write, BufWriter};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"JOULEHX1";  // "Joule history v1"
const VERSION: u32 = 1;

pub struct DiskHistory {
    path: PathBuf,
    file: BufWriter<File>,
    /// In-memory index for O(1) lookups. Populated from the file at
    /// open and updated on every `record()`.
    index: HashMap<EntryKey, HistoryEntry>,
    stats: HistoryStats,
    c_lookup: f64,
    c_hash_per_byte: f64,
}

impl DiskHistory {
    /// Open (or create) a history log at `path`. If the file exists,
    /// its records are loaded into the in-memory index.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, HistoryError> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .read(true).write(true).create(true)
            .open(&path)?;

        let mut index = HashMap::new();
        let mut stats = HistoryStats::default();
        let file_len = file.metadata()?.len();

        if file_len == 0 {
            // Fresh file: write the magic header.
            file.write_all(MAGIC)?;
            file.write_all(&VERSION.to_be_bytes())?;
            file.write_all(&[0u8; 4])?;  // padding to 16 bytes
        } else if file_len < 16 {
            return Err(HistoryError::Corrupt("file too short for header".into()));
        } else {
            // Existing file: verify header and read records.
            file.seek(SeekFrom::Start(0))?;
            let mut header = [0u8; 16];
            file.read_exact(&mut header)?;
            if &header[..8] != MAGIC {
                return Err(HistoryError::Corrupt("bad magic".into()));
            }
            let version = u32::from_be_bytes(header[8..12].try_into().unwrap());
            if version != VERSION {
                return Err(HistoryError::Corrupt(
                    format!("unsupported version {}", version)
                ));
            }
            // Read records until EOF.
            loop {
                let mut len_buf = [0u8; 8];
                match file.read_exact(&mut len_buf) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(HistoryError::Io(e)),
                }
                let record_len = u64::from_be_bytes(len_buf);
                let mut bytes = vec![0u8; record_len as usize];
                file.read_exact(&mut bytes)?;
                let entry = decode_record(&bytes)?;
                stats.joules_recorded += entry.answer.joules_spent;
                if !index.contains_key(&entry.key) {
                    stats.writes += 1;
                    stats.entry_count += 1;
                }
                index.insert(entry.key, entry);
            }
            // Seek to end for further appends.
            file.seek(SeekFrom::End(0))?;
        }

        Ok(Self {
            path,
            file: BufWriter::new(file),
            index,
            stats,
            c_lookup: 6e-9,
            c_hash_per_byte: 5e-11,
        })
    }

    pub fn path(&self) -> &Path { &self.path }
    pub fn len(&self) -> usize { self.index.len() }
    pub fn is_empty(&self) -> bool { self.index.is_empty() }

    /// Force a flush to disk. Useful in tests, or before a clean
    /// shutdown.
    pub fn flush(&mut self) -> Result<(), HistoryError> {
        self.file.flush()?;
        Ok(())
    }

    pub fn entries(&self) -> impl Iterator<Item = &HistoryEntry> {
        self.index.values()
    }
}

impl crate::semantic::IndexedHistory for DiskHistory {
    fn iter_entries(&self) -> Box<dyn Iterator<Item = HistoryEntry> + '_> {
        Box::new(self.index.values().cloned())
    }

    fn set_embedding(&mut self, key: &EntryKey, embedding: Vec<f32>)
        -> Result<(), HistoryError>
    {
        // Update the in-memory index. Persisting to disk via an
        // append-only log would require a separate "patch record"
        // format; for R5 the embedding is recomputed on next open.
        // (Future: dedicated embedding side-file.)
        if let Some(e) = self.index.get_mut(key) {
            e.embedding = embedding;
        }
        Ok(())
    }
}

impl HistoryLayer for DiskHistory {
    fn lookup_exact(&mut self, key: &EntryKey) -> Result<Option<HistoryAnswer>, HistoryError> {
        self.stats.total_lookups += 1;
        match self.index.get(key) {
            Some(e) => {
                self.stats.hits += 1;
                Ok(Some(e.answer.clone()))
            }
            None => {
                self.stats.misses += 1;
                Ok(None)
            }
        }
    }

    fn record(&mut self, q: &Query, a: &Answer) -> Result<EntryKey, HistoryError> {
        let key = key_for(q);
        let entry = HistoryEntry {
            key,
            query_input: q.input.clone(),
            query_context: q.context,
            answer: answer_to_history(a),
            timestamp_secs: now_secs(),
            embedding: Vec::new(),
        };
        // Write to disk.
        let bytes = encode_record(&entry);
        self.file.write_all(&(bytes.len() as u64).to_be_bytes())?;
        self.file.write_all(&bytes)?;
        self.file.flush()?;
        // Update in-memory index.
        let already = self.index.contains_key(&key);
        self.stats.joules_recorded += a.joules_spent;
        if !already {
            self.stats.writes += 1;
            self.stats.entry_count += 1;
        }
        self.index.insert(key, entry);
        Ok(key)
    }

    fn estimate_lookup_cost(&self, q: &Query) -> f64 {
        let len = match &q.input {
            QueryInput::Text(s) => s.len(),
            QueryInput::Structured(b) | QueryInput::Binary(b) => b.len(),
            QueryInput::Image(b) | QueryInput::Audio(b) => b.len(),
            QueryInput::Multimodal { text, images, audio } => {
                text.len()
                    + images.iter().map(|v| v.len()).sum::<usize>()
                    + audio.iter().map(|v| v.len()).sum::<usize>()
            }
        };
        self.c_lookup + self.c_hash_per_byte * (len as f64)
    }

    fn stats(&self) -> &HistoryStats { &self.stats }
}

// ============================================================
// Record encoding / decoding
// ============================================================

fn encode_record(e: &HistoryEntry) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    // input
    match &e.query_input {
        QueryInput::Text(s) => {
            out.push(0u8);
            out.extend_from_slice(&(s.len() as u32).to_be_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        QueryInput::Structured(b) => {
            out.push(1u8);
            out.extend_from_slice(&(b.len() as u32).to_be_bytes());
            out.extend_from_slice(b);
        }
        QueryInput::Binary(b) => {
            out.push(2u8);
            out.extend_from_slice(&(b.len() as u32).to_be_bytes());
            out.extend_from_slice(b);
        }
        QueryInput::Image(b) => {
            out.push(3u8);
            out.extend_from_slice(&(b.len() as u32).to_be_bytes());
            out.extend_from_slice(b);
        }
        QueryInput::Audio(b) => {
            out.push(4u8);
            out.extend_from_slice(&(b.len() as u32).to_be_bytes());
            out.extend_from_slice(b);
        }
        QueryInput::Multimodal { text, images, audio } => {
            // Build the inner payload first so we can emit its length.
            // Payload layout:
            //   u32 text_len | text_bytes
            //   u32 n_images | (u32 img_len | img_bytes) ...
            //   u32 n_audio  | (u32 a_len   | a_bytes  ) ...
            let mut payload = Vec::new();
            payload.extend_from_slice(&(text.len() as u32).to_be_bytes());
            payload.extend_from_slice(text.as_bytes());
            payload.extend_from_slice(&(images.len() as u32).to_be_bytes());
            for img in images {
                payload.extend_from_slice(&(img.len() as u32).to_be_bytes());
                payload.extend_from_slice(img);
            }
            payload.extend_from_slice(&(audio.len() as u32).to_be_bytes());
            for clip in audio {
                payload.extend_from_slice(&(clip.len() as u32).to_be_bytes());
                payload.extend_from_slice(clip);
            }
            out.push(5u8);
            out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
            out.extend_from_slice(&payload);
        }
    }
    // context
    out.extend_from_slice(&e.query_context.session_id.0);
    out.extend_from_slice(&e.query_context.history_fingerprint.0);
    // output
    match &e.answer.output {
        AnswerOutput::Text(s) => {
            out.push(0u8);
            out.extend_from_slice(&(s.len() as u32).to_be_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        AnswerOutput::Structured(b) => {
            out.push(1u8);
            out.extend_from_slice(&(b.len() as u32).to_be_bytes());
            out.extend_from_slice(b);
        }
        AnswerOutput::Refused(r) => {
            out.push(2u8);
            // Refusals serialize as: variant byte + optional payload.
            match r {
                RefusalReason::Inapplicable => {
                    out.push(0u8);
                    out.extend_from_slice(&0u32.to_be_bytes());
                }
                RefusalReason::LowConfidence(c) => {
                    out.push(1u8);
                    out.extend_from_slice(&c.to_be_bytes());
                }
                RefusalReason::TierSpecific(s) => {
                    out.push(2u8);
                    out.extend_from_slice(&(s.len() as u32).to_be_bytes());
                    out.extend_from_slice(s.as_bytes());
                }
            }
        }
    }
    // tier family + data
    let (family, data) = encode_tier(&e.answer.originating_tier);
    out.push(family);
    out.extend_from_slice(&data.to_be_bytes());
    // joules, confidence, timestamp
    out.extend_from_slice(&e.answer.joules_spent.to_be_bytes());
    out.extend_from_slice(&e.answer.confidence.to_be_bytes());
    out.extend_from_slice(&e.timestamp_secs.to_be_bytes());
    // embedding
    out.extend_from_slice(&(e.embedding.len() as u32).to_be_bytes());
    for &v in &e.embedding {
        out.extend_from_slice(&v.to_be_bytes());
    }
    out
}

fn decode_record(bytes: &[u8]) -> Result<HistoryEntry, HistoryError> {
    let mut p = Cursor::new(bytes);
    // input
    let input_variant = p.read_u8()?;
    let input_len = p.read_u32()? as usize;
    let input_bytes = p.read_n(input_len)?;
    let query_input = match input_variant {
        0 => QueryInput::Text(String::from_utf8(input_bytes.to_vec())
            .map_err(|e| HistoryError::Corrupt(format!("input not utf8: {}", e)))?),
        1 => QueryInput::Structured(input_bytes.to_vec()),
        2 => QueryInput::Binary(input_bytes.to_vec()),
        3 => QueryInput::Image(input_bytes.to_vec()),
        4 => QueryInput::Audio(input_bytes.to_vec()),
        5 => {
            // Parse the multimodal sub-payload from input_bytes.
            let mut sub = Cursor::new(input_bytes);
            let tlen = sub.read_u32()? as usize;
            let text = String::from_utf8(sub.read_n(tlen)?.to_vec())
                .map_err(|e| HistoryError::Corrupt(format!("multimodal text not utf8: {}", e)))?;
            let n_img = sub.read_u32()? as usize;
            let mut images = Vec::with_capacity(n_img);
            for _ in 0..n_img {
                let il = sub.read_u32()? as usize;
                images.push(sub.read_n(il)?.to_vec());
            }
            let n_audio = sub.read_u32()? as usize;
            let mut audio = Vec::with_capacity(n_audio);
            for _ in 0..n_audio {
                let al = sub.read_u32()? as usize;
                audio.push(sub.read_n(al)?.to_vec());
            }
            QueryInput::Multimodal { text, images, audio }
        }
        v => return Err(HistoryError::Corrupt(format!("bad input variant {}", v))),
    };
    // context
    let mut sid = [0u8; 16];
    sid.copy_from_slice(p.read_n(16)?);
    let mut fp = [0u8; 32];
    fp.copy_from_slice(p.read_n(32)?);
    let query_context = ContextRef {
        session_id: SessionId(sid),
        history_fingerprint: ContextFingerprint(fp),
    };
    // output
    let output_kind = p.read_u8()?;
    let output = match output_kind {
        0 => {
            let n = p.read_u32()? as usize;
            let b = p.read_n(n)?;
            AnswerOutput::Text(String::from_utf8(b.to_vec())
                .map_err(|e| HistoryError::Corrupt(format!("output not utf8: {}", e)))?)
        }
        1 => {
            let n = p.read_u32()? as usize;
            let b = p.read_n(n)?;
            AnswerOutput::Structured(b.to_vec())
        }
        2 => {
            let r = p.read_u8()?;
            match r {
                0 => { let _ = p.read_u32()?; AnswerOutput::Refused(RefusalReason::Inapplicable) }
                1 => {
                    let c = p.read_u32()?;
                    AnswerOutput::Refused(RefusalReason::LowConfidence(c))
                }
                2 => {
                    let n = p.read_u32()? as usize;
                    let b = p.read_n(n)?;
                    AnswerOutput::Refused(RefusalReason::TierSpecific(
                        String::from_utf8(b.to_vec())
                            .map_err(|e| HistoryError::Corrupt(format!("refusal: {}", e)))?
                    ))
                }
                v => return Err(HistoryError::Corrupt(format!("bad refusal {}", v))),
            }
        }
        v => return Err(HistoryError::Corrupt(format!("bad output kind {}", v))),
    };
    // tier
    let tier_family = p.read_u8()?;
    let tier_data = p.read_u32()?;
    let originating_tier = decode_tier(tier_family, tier_data)?;
    // numeric fields
    let joules = p.read_f64()?;
    let confidence = p.read_f32()?;
    let timestamp = p.read_u64()?;
    // embedding
    let edim = p.read_u32()? as usize;
    let mut embedding = Vec::with_capacity(edim);
    for _ in 0..edim {
        embedding.push(p.read_f32()?);
    }

    Ok(HistoryEntry {
        key: {
            // Recompute the key from the decoded query — gives us
            // round-trip integrity for free.
            use jouleclaw_core::hash::Hasher256;
            let mut h = Hasher256::new();
            h.update(b"L0v1");
            match &query_input {
                QueryInput::Text(s) => { h.update(b"T:"); h.update(s.as_bytes()); }
                QueryInput::Structured(b) => { h.update(b"S:"); h.update(b); }
                QueryInput::Binary(b) => { h.update(b"B:"); h.update(b); }
                QueryInput::Image(b) => { h.update(b"I:"); h.update(b); }
                QueryInput::Audio(b) => { h.update(b"A:"); h.update(b); }
                QueryInput::Multimodal { text, images, audio } => {
                    h.update(b"M:");
                    h.update(text.as_bytes());
                    h.update(b"|i:");
                    for img in images {
                        h.update(&(img.len() as u64).to_le_bytes());
                        h.update(img);
                    }
                    h.update(b"|a:");
                    for clip in audio {
                        h.update(&(clip.len() as u64).to_le_bytes());
                        h.update(clip);
                    }
                }
            }
            h.update(b"|C:");
            h.update(&query_context.history_fingerprint.0);
            h.finalize()
        },
        query_input,
        query_context,
        answer: HistoryAnswer {
            output,
            originating_tier,
            joules_spent: joules,
            confidence,
        },
        timestamp_secs: timestamp,
        embedding,
    })
}

fn encode_tier(tier: &TierId) -> (u8, u32) {
    // The L0-L4 wire encoding is byte-stable for receipts (per SPEC §7).
    // The L0-L10 fractional tiers added in v0.2 collapse onto their
    // coarse family for on-disk history; the precise tier identity
    // is preserved in the receipt's `tier` field via `wire_tag`.
    use jouleclaw_cascade::JouleClass;
    match tier {
        TierId::L0 => (0, 0),
        TierId::L1(p) => (1, *p as u32),
        TierId::L2(m) => (2, m.0),
        TierId::L3(m) => (3, m.0),
        TierId::L4(m) => (4, m.0),
        // Fractional and meta tiers collapse to their coarse class for
        // disk encoding; round-trip identity is via receipt.wire_tag.
        other => match other.joule_class() {
            JouleClass::Cache => (0, 0),
            JouleClass::Lawful => (1, 0),
            JouleClass::Embed => (2, 0),
            JouleClass::Model => (3, 0),
            JouleClass::Wire => (4, 0),
            JouleClass::Meta => (5, 0),
        },
    }
}

fn decode_tier(family: u8, data: u32) -> Result<TierId, HistoryError> {
    match family {
        0 => Ok(TierId::L0),
        1 => Ok(TierId::L1(decode_l1_primitive(data)?)),
        2 => Ok(TierId::L2(L2ModelId(data))),
        3 => Ok(TierId::L3(L3ModelId(data))),
        4 => Ok(TierId::L4(L4ModelId(data))),
        v => Err(HistoryError::Corrupt(format!("bad tier family {}", v))),
    }
}

fn decode_l1_primitive(data: u32) -> Result<L1Primitive, HistoryError> {
    // Mirror of L1Primitive's variant order in cascade::types.
    match data {
        0 => Ok(L1Primitive::CacheLookup),
        1 => Ok(L1Primitive::Tokenize),
        2 => Ok(L1Primitive::Detokenize),
        3 => Ok(L1Primitive::Regex),
        4 => Ok(L1Primitive::Parse),
        5 => Ok(L1Primitive::TemplateFill),
        6 => Ok(L1Primitive::Retrieve),
        7 => Ok(L1Primitive::Execute),
        v => Err(HistoryError::Corrupt(format!("bad L1 primitive {}", v))),
    }
}

// Tiny cursor reader.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, pos: 0 } }
    fn read_n(&mut self, n: usize) -> Result<&'a [u8], HistoryError> {
        if self.pos + n > self.bytes.len() {
            return Err(HistoryError::Corrupt(
                format!("short read: pos {}, want {}, have {}",
                    self.pos, n, self.bytes.len())));
        }
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn read_u8(&mut self) -> Result<u8, HistoryError> {
        Ok(self.read_n(1)?[0])
    }
    fn read_u32(&mut self) -> Result<u32, HistoryError> {
        let b = self.read_n(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> Result<u64, HistoryError> {
        let b = self.read_n(8)?;
        Ok(u64::from_be_bytes(b.try_into().unwrap()))
    }
    fn read_f32(&mut self) -> Result<f32, HistoryError> {
        let b = self.read_n(4)?;
        Ok(f32::from_be_bytes(b.try_into().unwrap()))
    }
    fn read_f64(&mut self) -> Result<f64, HistoryError> {
        let b = self.read_n(8)?;
        Ok(f64::from_be_bytes(b.try_into().unwrap()))
    }
}
