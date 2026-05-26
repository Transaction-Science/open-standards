# Gemma 2 Bridge Plan (audit 2026-05-26)

Bridge SANA-WM's `encode_text` stub to a real Gemma-2-2b-it forward via pattern-lang's
`joule-loader-gguf::gemma4` module.

## Key finding

- **Struct name is `Gemma4`** (not `ModelF16` as the bridge note assumed).
- `ForwardOut.final_norm` (`gemma4.rs:128`) already exposes the last hidden state
  pre-LM-head — **no API change needed** for the readout side.
- Two additive `Gemma4Config` fields cover both Gemma-2 deltas; **zero behavior
  change for existing Gemma-4 callers** (defaults preserve current path).
- Tokenizer (`gemma_tokenizer.rs`) is HF `tokenizer.json`-driven; **works as-is**
  for Gemma-2 (shared 262144-entry vocab).

## Concrete action plan

1. **`gemma4.rs:27-45`** — append two `pub` fields to `Gemma4Config`:
   - `attn_softcap: f32` (default `0.0` → no-op)
   - `q_pre_attn_scalar: f32` (default `0.0` → existing `scaling = 1.0` path)

2. **`gemma4.rs:54-95`** — read optional GGUF keys:
   - `gemma.attention.attn_logit_softcapping`
   - `gemma.attention.query_pre_attn_scalar`
   - Defaults `0.0`.
   - Relax `gemma.rs:58-63` arch check to accept `Gemma2ForCausalLM`, OR add
     sibling `load_gemma2_dir` that bypasses PLE.

3. **`gemma4.rs:487` (prefill) and `gemma4.rs:864` (cached)** — after q-norm + RoPE:
   ```rust
   if cfg.q_pre_attn_scalar > 0.0 {
       let q_scale = 1.0 / cfg.q_pre_attn_scalar.sqrt();
       for x in qh.iter_mut() { *x *= q_scale; }
   }
   ```

4. **`gemma4.rs:557` and `gemma4.rs:901`** — after `sc[j] = acc` inner loop,
   before the max scan:
   ```rust
   if cfg.attn_softcap > 0.0 {
       let sc_lim = cfg.attn_softcap;
       for s in sc[j0..=i].iter_mut() {
           *s = sc_lim * (*s / sc_lim).tanh();
       }
   }
   ```
   Recompute `mx` from the rewritten `sc`.

5. **`final_norm` already public** on `ForwardOut` — no change. SANA-WM reads
   `out.final_norm`, reshapes to `[seq, 2304]`, pads/truncates to `[300, 2304]`,
   casts to f16.

6. **Skip PLE + per-layer-gate + layer_scalar for Gemma 2**:
   - PLE block at `gemma4.rs:652-690` and `gemma4.rs:935-946` is already gated by
     `c.ple_dim > 0` — preserved.
   - `layer_scalar` at `gemma4.rs:693-696` and `gemma4.rs:948-951` — guard with
     `if self.w.contains_key(...)`.
   - Both edits local and additive.

7. **Add `pub fn forward_hidden(&self, ids: &[u32]) -> Vec<f32>`** as thin wrapper
   that returns only `final_norm`. ~5 LOC.

8. **efficient-genai side**:
   - Add cross-workspace path dep in `Cargo.toml`:
     ```toml
     joule-loader-gguf = { path = "../pattern-lang/crates/joule-loader-gguf" }
     ```
   - Add fields to `SanaWmPipeline`: `text_tokenizer: GemmaTokenizer`, `text_gemma: Gemma4`.
   - Load both in `SanaWmPipeline::new` from a directory pointed by env var or constructor arg.
   - Rewrite `sana_wm.rs::encode_text`:
     ```rust
     let ids = self.text_tokenizer.encode(prompt, true);
     let ids = if ids.len() > 300 { &ids[..300] } else { ids.as_slice() };
     let hidden_f32 = self.text_gemma.forward_hidden(ids);
     // hidden_f32 has shape [seq, 2304]; pad to [300, 2304]
     let mut out = vec![half::f16::ZERO; 300 * 2304];
     for (i, v) in hidden_f32.iter().enumerate() {
         if i / 2304 >= 300 { break; }
         out[i] = half::f16::from_f32(*v);
     }
     Ok(out)
     ```

9. **Oracle gate** — add `examples/sana_wm_text_verify.rs`:
   - Loads `gemma-2-2b-it` from HF
   - Runs `forward_hidden("a man walking")`
   - Asserts cos ≥ 0.999 vs HF `transformers` dump of `model(input_ids).last_hidden_state`

## Effort estimate

| Step | Days |
|---|---|
| Pattern-lang config + dual-mode forward (steps 1-7) | 1.5 |
| HF Gemma-2 oracle dump + cosine gate (step 9) | 0.5 |
| efficient-genai wiring + cache + f16 reshape (step 8) | 0.5 |
| End-to-end SANA-WM run + cos-check vs Sana reference | 0.5 |
| **Total** | **~3 person-days** |

## Relevant files

- `/Users/dcharlot/data-share/vibe-coding/pattern-lang/crates/joule-loader-gguf/src/gemma4.rs`
- `/Users/dcharlot/data-share/vibe-coding/pattern-lang/crates/joule-loader-gguf/src/gemma_tokenizer.rs`
- `/Users/dcharlot/data-share/vibe-coding/pattern-lang/crates/joule-loader-gguf/src/gemma.rs`
- `/Users/dcharlot/data-share/vibe-coding/efficient-genai/src/inference/architecture/sana_wm.rs` (lines 541-560 = bridge target)
