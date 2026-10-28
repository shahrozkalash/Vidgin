# CHANGELOG

## Upgrade
- Added Farneback optical flow with NumPy FFT phase-correlation fallback to estimate global translation and residual subject movement, replacing the original frame-difference-only camera classification.
- Added MiDaS inverse-depth polarity and center/edge near/far foreground/background fields.
- Added deterministic luminance-derived lighting labels, overriding model guesses.
- Added a denser-frame expansion pass for captions below 40 words, with no synthetic padding.
- Added optional `--self-check` for independent-frame subject/action agreement.
- Added separate generated/reference manifest benchmark join, per-field exact/F1 and caption F1 quality gate.
- Reused persistent HTTP sessions and expanded synthetic self-tests.
- Preserved original Gemini raw rows, hash resume, atomic backup, and `caption_weak` handling.

## Round 2

- Inverted pan/tilt labels to camera-centric convention to match human descriptions.
- Added NumPy scale-registration zoom detector and synthetic center-crop zoom test.
- Added residual-motion subject mask, dilation, inverse-depth mask statistics, and subject-side field; tested moving off-center near block.
- Added editable deterministic JSON synonym map and normalization counter.
- Extended benchmark with synonym F1, camera confusion matrix, and seeded human review pack.
- Added SHA256 filename sharding (`--shard K/N`) and disjointness/exhaustiveness test.
- Added bounded backend retries, one malformed-JSON repair prompt, error categories, per-stage timing medians, and temporal image-position metadata.
- Preserved manifest byte-level skip semantics, backup, hash resume, and existing self-test checks.

## Round 3 — observation-grounded quality and readiness

- Changed VLM response schema to frame-positioned `observations`, `subject`, `action`, `setting`, `start_state`, and `end_state`. Composed scene change in code and added two marked few-shot examples to constrain style.
- Added lexical grounding flags and candidate selection with deterministic score and margin; self-check now retains the better candidate instead of merely flagging disagreement.
- Added configurable 2x upscaling, sampled-frame dedup with four-frame floor, per-clip base64 encoding cache, and optional OpenCV JPEG.
- Added `--eval-suite` with contract audit, paired Gemini benchmark, stratified chunk reporting, 10-pair human review pack, and explicit gate-by-gate readiness verdict. Stub provenance blocks readiness.
- Added read-only `--suggest-normalizations`, atomic `.caption_progress.json`, grounding and frame-count statistics, and five named Round 3 synthetic tests.
- Preserved Gemini byte identity, SHA-256 resume, `.bak` writes, sharding, measured motion/depth/lighting, and loopback-only backends.
- **Still unverified:** real Gemini-caption quality, human review outcome, and <=10-second T4 throughput.

## Round 4 — skeptic and evidence

- Added optional adversarial `--skeptic` and `--skeptic-model`: claim-level verification, one banned-claim rewrite, second verification, persistent failure flags.
- Added heuristic motion/zoom/depth confidence and conservative unknown/undetermined fallback; explicitly documented missing calibration and mask-coverage signals.
- Added optional middle temporal beat, secondary subjects, and vague-word lint without forced rewriting.
- Added blind randomized A/B pack, separate answer key, rating scorer, and optional eval-suite A/B reporting.
- Added manifest-level `--jobs` process pool with worker-specific sessions, progress PID and parent aggregation.
- Extended standalone self-test with named skeptic, confidence, composition, abpack and parallel cases.

## Round 5 — provenance, immutable identity, and operational audit

- Added explicit backend/model provenance on all new VLM rows and a conservative real-backend readiness gate rejecting legacy missing provenance and stub output.
- Added one-time `.gemini_baseline.json` raw-row SHA-256 snapshots and baseline-first verification; `.bak` remains a labeled fallback.
- Corrected benchmark field metrics to exclude blank reference fields from denominators, without changing caption F1 or readiness thresholds.
- Added append-only content-hash caption registry, durable per-manifest flush, malformed-tail tolerance, and `reused` counts; enforced single-writer mode for registry safety.
- Added configurable consecutive backend-failure circuit breaker and nonzero exit in sequential mode.
- Added local model/server preflight with a synthetic full-pipeline clip, stage timings, grounding and length reporting.
- Added and executed named synthetic tests: provenance, baseline, fieldna, registry, breaker, preflight. Retained Rounds 1–4 regression tests.

## Targeted HTTP backend bug fix (2026-10-28)

- Fixed `LocalCompatibleBackend.describe()` to accept pre-encoded base64 image strings from `FrameCache` directly in `data:image/png;base64,...` URLs, while continuing to PNG-encode NumPy arrays. This matches the existing `OllamaBackend.describe()` behavior.
- Audited the shared `robust_describe()` invocation path (standard, expansion, self-check, skeptic and repair): all use the same fixed `describe()` implementation; no other backend requires the same change.
- Added named `httpbackends` self-test: an ephemeral loopback HTTP server mimics `/api/chat` and `/chat/completions`, exercises each real backend class through `process_manifest()`, checks successful `vlm` rows, and validates compatible-backend data URI images decode to PNG bytes.
- All previous synthetic self-tests still pass. No real-model accuracy or GPU throughput claims are implied by the HTTP mock test.


## Round 6 — incremental persistence and HTTP recovery coverage (2026-10-28)

- Added `--flush-every N` (default 10, 0 disables intermediate checkpoints). Every N successfully written rows, persist the full manifest atomically with all unvisited rows unchanged byte-for-byte; fsync and atomically update progress. Preserve the first pre-run `.bak` through every checkpoint.
- Kept hash-resume, Gemini protection, registry append behavior, and manifest-level single-writer ownership. A resumed invocation treats its checkpointed manifest as its own pre-run backup source.
- Added `flushkill` test: actual subprocess SIGKILL after checkpoint, JSONL integrity, byte preservation, hash-skip resume, and exact processed counts.
- Added `httprecovery` test: both real HTTP backend classes against a request-aware local fake server for malformed JSON repair, alternate-candidate selection, and skeptic rewrite with persistent flags.
- Preserved PNG/JPEG data-URI MIME labeling in the compatible backend (the available prior file lacked this fix, so it is carried forward explicitly).
- No new caption-generation features, backends, or readiness thresholds.
