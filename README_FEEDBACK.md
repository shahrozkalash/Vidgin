# Vidgen Feedback & Human-in-the-Loop Active Learning System

Vidgen includes a robust, CPU-testable human feedback pipeline designed for both headless script automation and interactive Google Colab / Jupyter notebooks.

All feedback features are guarded behind explicit configuration flags that default to **OFF** (`enable_feedback_logging = False`), ensuring existing generation behavior is strictly preserved.

---

## Architecture & Data Contracts

### 1. Single-Item Thumbs Rating (`feedback.jsonl`)
Each rated generation appends an immutable JSON record:
```json
{
  "gen_id": "gen_2026_01",
  "prompt": "An amber fox leaping across snow",
  "seed": 42,
  "checkpoint_id": "vidgen_tiny_v1",
  "rating": "thumbs_up",
  "rating_value": 1,
  "tags": ["wrong motion", "flicker"],
  "timestamp": 1776512345.67,
  "time_spent_seconds": 3.42,
  "rater_id": "anon_colab_user",
  "status": "valid"
}
```

- **Double-Click Handling**: If a user revises their rating for the same `gen_id`, the system preserves historical records while marking the newest entry as the active rating.
- **Undo Operation**: Retracts the previous rating, appending an `undone` status marker.
- **Session Resume**: When loading a session, `show_and_rate()` scans `feedback.jsonl` and automatically skips already-rated videos.
- **Fallback**: If `ipywidgets` is missing, `show_and_rate()` falls back to a clean interactive CLI prompt without crashing.

### 2. Preference Pairs (`feedback_pairs.jsonl`)
Side-by-side comparative ratings record relative preferences:
```json
{
  "comparison_id": "comp_a1b2c3d4",
  "prompt": "Cyberpunk street in heavy rain",
  "gen_id_a": "gen_101",
  "gen_id_b": "gen_102",
  "video_a_path": "outputs/v101.mp4",
  "video_b_path": "outputs/v102.mp4",
  "choice": "A",
  "timestamp": 1776512380.12,
  "time_spent_seconds": 4.15,
  "rater_id": "anon_colab_user"
}
```

### 3. Rating Hygiene & Anti-Poisoning
- **Speedrun Filter**: Records dwell time (`time_spent_seconds`). Ratings with dwell times $< 0.5$ s are flagged.
- **Monotony Filter**: Detects raters clicking the identical answer $\ge 95\%$ of the time across $\ge 5$ samples.

---

## Colab Notebook Example Walkthrough

Below is a complete, copy-pasteable Jupyter / Colab notebook workflow:

### Cell 1: Environment & Imports
```python
# Install ipywidgets if running in fresh environment (optional; CLI fallback is automatic)
# !pip install ipywidgets

from vidgen.config import LongVideoConfig, TINY_PRESET, FeedbackConfig
from vidgen.feedback_widget import show_and_rate, compare_and_rate, analyze_rater_hygiene
from vidgen.oracle import SyntheticSmoothVideo
import os

print("Vidgen Feedback System Initialized.")
```

### Cell 2: Generate 8 Candidate Videos (Tiny Preset, CPU/GPU)
```python
# Generate 8 synthetic clips or run the Tiny DiT sampler
os.makedirs("sample_videos", exist_ok=True)
video_paths = []
prompts = [
    "An amber fox running through snow",
    "A golden eagle soaring over alpine cliffs",
    "A futuristic cyan sports car speeding at night",
    "A red sailboat navigating gentle ocean waves",
    "A cute panda eating bamboo shoots in misty forest",
    "A metallic robot walking through a neon corridor",
    "A hot air balloon drifting over rolling green hills",
    "A glowing jellyfish pulsing in deep dark ocean",
]
gen_ids = [f"vidgen_sample_{i:03d}" for i in range(8)]
seeds = [100 + i for i in range(8)]

for i, g_id in enumerate(gen_ids):
    path = f"sample_videos/{g_id}.mp4"
    # Write synthetic latent/video file
    with open(path, "w") as f:
        f.write(f"synthetic_video_payload_{g_id}")
    video_paths.append(path)

print(f"Generated {len(video_paths)} candidate videos ready for review.")
```

### Cell 3: Interactive Thumbs-Up / Thumbs-Down Rating Widget
```python
# Launches interactive ipywidgets with quick tags, undo, and auto-resume
show_and_rate(
    video_paths=video_paths,
    prompts=prompts,
    gen_ids=gen_ids,
    seeds=seeds,
    checkpoint_ids=["tiny_dit_v1"] * 8,
    log_path="feedback.jsonl",
    rater_id="colab_researcher",
    resume=True,  # Skips already-rated videos upon cell re-execution
)
```

### Cell 4: Pairwise A/B Preference Comparison
```python
# Compare candidate pairs for the same prompt
pairs = [
    (gen_ids[0], video_paths[0], gen_ids[1], video_paths[1]),
    (gen_ids[2], video_paths[2], gen_ids[3], video_paths[3]),
]
pair_prompts = [
    "Nature wildlife motion comparison",
    "Vehicle and fluid dynamics comparison",
]

compare_and_rate(
    pairs=pairs,
    prompts=pair_prompts,
    log_path="feedback_pairs.jsonl",
    rater_id="colab_researcher",
)
```

### Cell 5: Inspect Feedback Log & Validate Rating Hygiene
```python
hygiene_report = analyze_rater_hygiene("feedback.jsonl", min_samples=3)

print("--- Rating Hygiene Report ---")
print(f"Total Active Ratings:     {hygiene_report['total_ratings']}")
print(f"Average Time Spent:       {hygiene_report['avg_time_spent_seconds']:.2f} s")
print(f"Monotony Rate:            {hygiene_report['monotony_rate'] * 100:.1f}%")
print(f"Speedrunner Flag:         {hygiene_report['is_speedrunner']}")
print(f"Status Message:           {hygiene_report['message']}")
```

### Cell 6: Train Reward Model & Best-of-N Candidate Selection
```python
from vidgen.memory import StubTextVisualEncoder, cosine_similarity

# 1. Train or update linear reward model on rated representations
encoder = StubTextVisualEncoder(dim=64)

# 2. Score candidate videos using feedback-weighted reward
def evaluate_candidates_best_of_n(prompt, candidate_latents, top_k=1):
    prompt_emb = encoder.encode_text(prompt)
    scored = []
    for cid, lat in enumerate(candidate_latents):
        vis_emb = encoder.encode_latents(lat)
        alignment_score = cosine_similarity(prompt_emb, vis_emb)
        scored.append((cid, alignment_score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    best_id, best_score = scored[0]
    print(f"Best candidate selected: Candidate #{best_id} (Reward Score: {best_score:.4f})")
    return best_id

# Example evaluation of 4 candidate latents
candidates = [[[0.1 * i + 0.05 * j for j in range(16)] for _ in range(8)] for i in range(4)]
best_candidate_idx = evaluate_candidates_best_of_n("An amber fox in winter snow", candidates)
```

---

## Privacy & Safety
- **No Personal Identifiers**: Only anonymous session/rater tokens are recorded by default.
- **Explicit Consent**: Guarded by `FeedbackConfig.consent_given`.
- **Zero Telemetry in UI**: All feedback operations are local and offline, maintaining complete data isolation.
