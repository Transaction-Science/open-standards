//! Rust port of CompressAI's rANS decoder (compressai/cpp_exts/rans/
//! rans_interface.cpp + third_party/ryg_rans/rans64.h).
//!
//! Byte-exact compatible with the upstream encoder — the Rust native
//! sink and the browser JS port (wai-web/demo/rans.js) both decode the
//! same bitstream to the same int32 symbols. precision=16, bypass=4.

const PRECISION:        u32 = 16;
const BYPASS_PRECISION: u32 = 4;
const MAX_BYPASS_VAL:   u32 = (1 << BYPASS_PRECISION) - 1;
const RANS64_L:         u64 = 1u64 << 31;

pub struct RansDecoder<'a> {
    words: &'a [u8],  // big-byte view; we read u32 LE words at idx
    idx:   usize,     // word index (i.e., 4*idx is byte offset)
    state: u64,
}

impl<'a> RansDecoder<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, &'static str> {
        if bytes.len() % 4 != 0 {
            return Err("rANS bitstream length not /4");
        }
        if bytes.len() < 8 {
            return Err("rANS bitstream too short to seed 64-bit state");
        }
        let w0 = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as u64;
        let w1 = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as u64;
        Ok(Self { words: bytes, idx: 2, state: (w1 << 32) | w0 })
    }

    #[inline] fn renorm(&mut self) {
        if self.state < RANS64_L {
            let off = self.idx * 4;
            let next = u32::from_le_bytes(self.words[off..off + 4].try_into().unwrap()) as u64;
            self.state = (self.state << 32) | next;
            self.idx += 1;
        }
    }

    fn get_bits(&mut self, n: u32) -> u32 {
        let mask = (1u64 << n) - 1;
        let val  = (self.state & mask) as u32;
        self.state >>= n;
        self.renorm();
        val
    }

    /// Decode one symbol against `cdf[0..cdf_len]` with per-channel `offset`.
    pub fn decode(&mut self, cdf: &[i32], cdf_len: usize, offset: i32) -> i32 {
        let max_value = (cdf_len as i32) - 2;
        let cum_freq  = (self.state & ((1u64 << PRECISION) - 1)) as i32;
        // Binary search for the largest s such that cdf[s] <= cum_freq.
        let (mut lo, mut hi) = (1, cdf_len - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if cdf[mid] > cum_freq { hi = mid; } else { lo = mid + 1; }
        }
        let s = (lo - 1) as i32;
        let start = cdf[s as usize] as u64;
        let freq  = (cdf[(s + 1) as usize] - cdf[s as usize]) as u64;
        self.state = freq * (self.state >> PRECISION)
                   + (self.state & ((1u64 << PRECISION) - 1)) - start;
        self.renorm();

        let mut value = s;
        if value == max_value {
            let mut val = self.get_bits(BYPASS_PRECISION);
            let mut n_bypass = val;
            while val == MAX_BYPASS_VAL {
                val = self.get_bits(BYPASS_PRECISION);
                n_bypass += val;
            }
            let mut raw_val: u32 = 0;
            for j in 0..n_bypass {
                let v = self.get_bits(BYPASS_PRECISION);
                raw_val |= v << (j * BYPASS_PRECISION);
            }
            value = (raw_val >> 1) as i32;
            if (raw_val & 1) != 0 { value = -value - 1; }
            else                  { value += max_value; }
        }
        value + offset
    }
}
