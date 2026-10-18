"""
Vidgen Feedback Widget for Colab & Jupyter Notebooks.
Supports:
1. show_and_rate(video_paths, prompts, gen_ids):
   - Thumbs-up and thumbs-down buttons (ipywidgets).
   - Quick tags (wrong motion, wrong character, flicker, artifact, low quality).
   - Skip and Undo buttons.
   - Appends records to feedback JSONL (gen_id, prompt, seed, checkpoint_id, rating, tags, time_spent).
2. compare_and_rate(pairs, prompts):
   - Side-by-side preference comparison (A, B, or neither).
   - Records preference pairs in JSONL.
3. Resilience & Hygiene:
   - Double-click handling (retains latest rating).
   - Undo functionality (retracts previous rating).
   - Session reload / resume (skips already rated gen_ids).
   - Graceful fallback to CLI / text prompt when ipywidgets is missing.
   - Fake widget backend for headless CPU automated testing.
   - Rater hygiene analysis (flags monotony and speedrunners).
"""

import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from vidgen.config import FeedbackConfig


def read_feedback_records(log_path: str) -> List[Dict[str, Any]]:
    """Reads all valid (non-undone) records from feedback JSONL."""
    if not os.path.exists(log_path):
        return []
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def get_active_ratings(log_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns active ratings keyed by gen_id.
    Handles double clicks and updates: only the last non-undone rating for each gen_id is active.
    """
    records = read_feedback_records(log_path)
    active: Dict[str, Dict[str, Any]] = {}
    for r in records:
        gid = r.get("gen_id")
        if not gid:
            continue
        status = r.get("status", "valid")
        if status == "undone":
            if gid in active:
                del active[gid]
        elif status in ("valid", "updated"):
            active[gid] = r
    return active


def append_feedback_record(record: Dict[str, Any], log_path: str):
    """Appends a single JSON record to the feedback log."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def undo_last_feedback(log_path: str) -> Optional[Dict[str, Any]]:
    """
    Appends an undo marker or pops the last active rating from the feedback log.
    Returns the retracted record if successful.
    """
    records = read_feedback_records(log_path)
    if not records:
        return None

    # Find the last record that is not already undone
    target_idx = None
    for i in range(len(records) - 1, -1, -1):
        if records[i].get("status", "valid") != "undone":
            target_idx = i
            break

    if target_idx is None:
        return None

    last_record = records[target_idx]
    # Mark record as undone and append undo event
    undo_event = dict(last_record)
    undo_event["status"] = "undone"
    undo_event["undone_at"] = time.time()
    append_feedback_record(undo_event, log_path)
    return last_record


def analyze_rater_hygiene(
    log_path: str,
    min_samples: int = 5,
    speedrun_threshold_sec: float = 0.5,
    monotony_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Analyzes rating hygiene for raters:
    - Flags monotonous raters who always click the exact same answer (e.g. 100% thumbs-up or 100% thumbs-down).
    - Flags speedrunning raters with unrealistically low time spent (< 0.5s per video).
    """
    active_ratings = list(get_active_ratings(log_path).values())
    total_ratings = len(active_ratings)

    if total_ratings < min_samples:
        return {
            "total_ratings": total_ratings,
            "is_monotonous": False,
            "monotony_rate": 0.0,
            "avg_time_spent_seconds": 0.0,
            "is_speedrunner": False,
            "flagged": False,
            "message": f"Insufficient ratings ({total_ratings} < {min_samples}) for hygiene evaluation.",
        }

    # Time spent analysis
    times = [r.get("time_spent_seconds", 0.0) for r in active_ratings if "time_spent_seconds" in r]
    avg_time = sum(times) / len(times) if times else 0.0
    is_speedrunner = avg_time < speedrun_threshold_sec

    # Monotony analysis
    choices = [r.get("rating") for r in active_ratings if "rating" in r]
    counts: Dict[str, int] = {}
    for c in choices:
        counts[c] = counts.get(c, 0) + 1

    max_choice_count = max(counts.values()) if counts else 0
    monotony_rate = max_choice_count / len(choices) if choices else 0.0
    is_monotonous = monotony_rate >= monotony_threshold

    flagged = is_monotonous or is_speedrunner
    reasons = []
    if is_monotonous:
        reasons.append(f"Monotonous clicking ({monotony_rate * 100:.1f}% identical choices)")
    if is_speedrunner:
        reasons.append(f"Speedrunning rater (avg time {avg_time:.2f}s < {speedrun_threshold_sec}s)")

    return {
        "total_ratings": total_ratings,
        "is_monotonous": is_monotonous,
        "monotony_rate": monotony_rate,
        "avg_time_spent_seconds": avg_time,
        "is_speedrunner": is_speedrunner,
        "flagged": flagged,
        "message": "; ".join(reasons) if flagged else "Rating hygiene healthy.",
    }


class FakeWidgetBackend:
    """
    Mock widget backend for automated unit testing on CPU without Jupyter.
    Simulates interactive user clicks and tags.
    """
    def __init__(self, action_sequence: Optional[List[Dict[str, Any]]] = None):
        # Action sequence: list of actions to simulate, e.g.
        # [{"action": "thumbs_up", "tags": ["flicker"]}, {"action": "undo"}, ...]
        self.action_sequence = action_sequence or []
        self.action_idx = 0
        self.displayed_items: List[Dict[str, Any]] = []

    @property
    def has_more_actions(self) -> bool:
        return self.action_idx < len(self.action_sequence)

    def display_item(self, item_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.displayed_items.append(item_info)
        if self.action_idx < len(self.action_sequence):
            action = self.action_sequence[self.action_idx]
            self.action_idx += 1
            return action
        return {"action": "skip"}

    def display_pair(self, pair_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.displayed_items.append(pair_info)
        if self.action_idx < len(self.action_sequence):
            action = self.action_sequence[self.action_idx]
            self.action_idx += 1
            return action
        return {"action": "A"}


class CliBackend:
    """
    CLI/terminal interactive fallback when ipywidgets is not installed.
    """
    def display_item(self, item_info: Dict[str, Any]) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print(f"Video {item_info.get('index', 0) + 1}/{item_info.get('total', 1)}: {item_info.get('video_path')}")
        print(f"Prompt: {item_info.get('prompt')}")
        print("Options: [u] Thumbs Up  [d] Thumbs Down  [s] Skip  [z] Undo  [q] Quit")
        choice = input("Your choice: ").strip().lower()

        if choice in ("u", "up", "1", "+"):
            tag_input = input("Quick tags (comma-separated, optional): ").strip()
            tags = [t.strip() for t in tag_input.split(",") if t.strip()]
            return {"action": "thumbs_up", "tags": tags}
        elif choice in ("d", "down", "0", "-"):
            tag_input = input("Quick tags (comma-separated, optional): ").strip()
            tags = [t.strip() for t in tag_input.split(",") if t.strip()]
            return {"action": "thumbs_down", "tags": tags}
        elif choice in ("z", "undo"):
            return {"action": "undo"}
        elif choice in ("q", "quit"):
            return {"action": "quit"}
        else:
            return {"action": "skip"}

    def display_pair(self, pair_info: Dict[str, Any]) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("Preference Comparison:")
        print(f"Prompt: {pair_info.get('prompt')}")
        print(f"Video A: {pair_info.get('video_a_path')}")
        print(f"Video B: {pair_info.get('video_b_path')}")
        print("Which is better? [a] Video A  [b] Video B  [n] Neither  [z] Undo")
        choice = input("Your choice: ").strip().lower()

        if choice in ("a", "1"):
            return {"action": "A"}
        elif choice in ("b", "2"):
            return {"action": "B"}
        elif choice in ("z", "undo"):
            return {"action": "undo"}
        else:
            return {"action": "neither"}


def show_and_rate(
    video_paths: List[str],
    prompts: List[str],
    gen_ids: List[str],
    seeds: Optional[List[int]] = None,
    checkpoint_ids: Optional[List[str]] = None,
    log_path: str = "feedback.jsonl",
    rater_id: str = "default_rater",
    config: Optional[FeedbackConfig] = None,
    backend: Optional[Union[FakeWidgetBackend, CliBackend]] = None,
    resume: bool = True,
    auto_advance: bool = True,
) -> Dict[str, Any]:
    """
    Displays generated videos with thumbs-up and thumbs-down buttons,
    quick tags, skip, and undo.
    Appends feedback records to feedback.jsonl.
    Handles double clicks (keeps last rating) and session reload (skips already rated).
    """
    cfg = config or FeedbackConfig()
    num_items = len(video_paths)
    assert len(prompts) == num_items, "Mismatch between video_paths and prompts"
    assert len(gen_ids) == num_items, "Mismatch between video_paths and gen_ids"

    if seeds is None:
        seeds = [42] * num_items
    if checkpoint_ids is None:
        checkpoint_ids = ["vidgen_checkpoint"] * num_items

    # 1. Session reload / resume: inspect log and find already rated gen_ids
    active_ratings = get_active_ratings(log_path) if resume else {}
    rated_gen_ids = set(active_ratings.keys())

    # Choose backend
    if backend is None:
        try:
            import ipywidgets as widgets
            from IPython.display import display
            # In real Jupyter environment with ipywidgets, use ipywidgets backend
            backend = None  # Handled below
        except ImportError:
            # Fallback to CLI when ipywidgets is missing
            backend = CliBackend()

    # If running with fake backend or CLI
    if backend is not None:
        return _run_non_widget_loop(
            video_paths=video_paths,
            prompts=prompts,
            gen_ids=gen_ids,
            seeds=seeds,
            checkpoint_ids=checkpoint_ids,
            log_path=log_path,
            rater_id=rater_id,
            rated_gen_ids=rated_gen_ids,
            backend=backend,
        )

    # 2. Native ipywidgets implementation for Jupyter/Colab
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output

    out_container = widgets.Output()

    current_idx = 0
    # Fast-forward past already rated items if resume=True
    while current_idx < num_items and gen_ids[current_idx] in rated_gen_ids:
        current_idx += 1

    start_time = [time.time()]
    session_history: List[int] = []

    def render_current():
        nonlocal current_idx
        with out_container:
            clear_output(wait=True)
            if current_idx >= num_items:
                display(HTML("<h3>🎉 All videos in this session have been rated!</h3>"))
                hygiene = analyze_rater_hygiene(log_path)
                display(HTML(f"<p>Ratings recorded: <b>{hygiene['total_ratings']}</b>. Status: {hygiene['message']}</p>"))
                return

            v_path = video_paths[current_idx]
            p_text = prompts[current_idx]
            g_id = gen_ids[current_idx]

            display(HTML(f"<h4>Video {current_idx + 1} of {num_items}</h4>"))
            display(HTML(f"<p><b>Prompt:</b> {p_text}</p>"))
            display(HTML(f"<p style='color: #64748b;'>ID: {g_id}</p>"))

            # Display video or placeholder
            if os.path.exists(v_path):
                display(HTML(f"""
                <video width="320" height="240" controls autoplay loop style="border-radius: 8px;">
                  <source src="{v_path}" type="video/mp4">
                  Your browser does not support HTML5 video.
                </video>
                """))
            else:
                display(HTML(f"<div style='padding: 20px; background: #1e293b; color: #94a3b8; border-radius: 8px;'>[Video file: {v_path}]</div>"))

            # Quick tag checkboxes
            tag_cb_motion = widgets.Checkbox(value=False, description="Wrong motion")
            tag_cb_char = widgets.Checkbox(value=False, description="Wrong character")
            tag_cb_flicker = widgets.Checkbox(value=False, description="Flicker")
            tag_cb_artifact = widgets.Checkbox(value=False, description="Artifact")
            tag_cb_lowq = widgets.Checkbox(value=False, description="Low quality")
            tags_box = widgets.HBox([tag_cb_motion, tag_cb_char, tag_cb_flicker, tag_cb_artifact, tag_cb_lowq])

            # Buttons
            btn_up = widgets.Button(description="👍 Thumbs Up", button_style="success", icon="thumbs-up")
            btn_down = widgets.Button(description="👎 Thumbs Down", button_style="danger", icon="thumbs-down")
            btn_skip = widgets.Button(description="⏭️ Skip", button_style="info")
            btn_undo = widgets.Button(description="↩️ Undo", button_style="warning", disabled=(len(session_history) == 0))

            btn_box = widgets.HBox([btn_up, btn_down, btn_skip, btn_undo])

            def on_up_clicked(_):
                nonlocal current_idx
                elapsed = time.time() - start_time[0]
                selected_tags = []
                if tag_cb_motion.value: selected_tags.append("wrong motion")
                if tag_cb_char.value: selected_tags.append("wrong character")
                if tag_cb_flicker.value: selected_tags.append("flicker")
                if tag_cb_artifact.value: selected_tags.append("artifact")
                if tag_cb_lowq.value: selected_tags.append("low quality")

                rec = {
                    "gen_id": g_id,
                    "prompt": p_text,
                    "seed": seeds[current_idx],
                    "checkpoint_id": checkpoint_ids[current_idx],
                    "rating": "thumbs_up",
                    "rating_value": 1,
                    "tags": selected_tags,
                    "timestamp": time.time(),
                    "time_spent_seconds": round(elapsed, 2),
                    "rater_id": rater_id,
                    "status": "valid",
                }
                append_feedback_record(rec, log_path)
                session_history.append(current_idx)
                current_idx += 1
                start_time[0] = time.time()
                render_current()

            def on_down_clicked(_):
                nonlocal current_idx
                elapsed = time.time() - start_time[0]
                selected_tags = []
                if tag_cb_motion.value: selected_tags.append("wrong motion")
                if tag_cb_char.value: selected_tags.append("wrong character")
                if tag_cb_flicker.value: selected_tags.append("flicker")
                if tag_cb_artifact.value: selected_tags.append("artifact")
                if tag_cb_lowq.value: selected_tags.append("low quality")

                rec = {
                    "gen_id": g_id,
                    "prompt": p_text,
                    "seed": seeds[current_idx],
                    "checkpoint_id": checkpoint_ids[current_idx],
                    "rating": "thumbs_down",
                    "rating_value": 0,
                    "tags": selected_tags,
                    "timestamp": time.time(),
                    "time_spent_seconds": round(elapsed, 2),
                    "rater_id": rater_id,
                    "status": "valid",
                }
                append_feedback_record(rec, log_path)
                session_history.append(current_idx)
                current_idx += 1
                start_time[0] = time.time()
                render_current()

            def on_skip_clicked(_):
                nonlocal current_idx
                current_idx += 1
                start_time[0] = time.time()
                render_current()

            def on_undo_clicked(_):
                nonlocal current_idx
                if session_history:
                    prev_idx = session_history.pop()
                    undo_last_feedback(log_path)
                    current_idx = prev_idx
                    start_time[0] = time.time()
                    render_current()

            btn_up.on_click(on_up_clicked)
            btn_down.on_click(on_down_clicked)
            btn_skip.on_click(on_skip_clicked)
            btn_undo.on_click(on_undo_clicked)

            display(tags_box)
            display(btn_box)

    display(out_container)
    render_current()
    return {"status": "interactive_ipywidgets", "total": num_items, "log_path": log_path}


def _run_non_widget_loop(
    video_paths: List[str],
    prompts: List[str],
    gen_ids: List[str],
    seeds: List[int],
    checkpoint_ids: List[str],
    log_path: str,
    rater_id: str,
    rated_gen_ids: Set[str],
    backend: Union[FakeWidgetBackend, CliBackend],
) -> Dict[str, Any]:
    """Loop handler for CLI and Fake testing backends."""
    num_items = len(video_paths)
    current_idx = 0
    # Session resume
    while current_idx < num_items and gen_ids[current_idx] in rated_gen_ids:
        current_idx += 1

    session_history: List[int] = []
    rated_count = 0
    start_time = time.time()

    while True:
        if current_idx >= num_items:
            # Check if backend has more actions (such as undo) at session end
            has_more = getattr(backend, "has_more_actions", False)
            if session_history and has_more:
                final_info = {"index": num_items, "total": num_items, "gen_id": "", "status": "completed"}
                resp = backend.display_item(final_info) if hasattr(backend, "display_item") else None
                if resp and resp.get("action") == "undo":
                    prev_idx = session_history.pop()
                    undo_last_feedback(log_path)
                    current_idx = prev_idx
                    start_time = time.time()
                    continue
            break

        g_id = gen_ids[current_idx]
        p_text = prompts[current_idx]
        v_path = video_paths[current_idx]

        item_info = {
            "index": current_idx,
            "total": num_items,
            "gen_id": g_id,
            "prompt": p_text,
            "video_path": v_path,
        }

        resp = backend.display_item(item_info)
        action = resp.get("action", "skip")

        if action == "quit":
            break
        elif action == "undo":
            if session_history:
                prev_idx = session_history.pop()
                undo_last_feedback(log_path)
                current_idx = prev_idx
                start_time = time.time()
                continue
            else:
                current_idx += 1
                continue
        elif action in ("thumbs_up", "thumbs_down"):
            elapsed = time.time() - start_time
            tags = resp.get("tags", [])
            rating_val = 1 if action == "thumbs_up" else 0

            # Double click check: if previous was for the same gen_id, supersede it
            rec = {
                "gen_id": g_id,
                "prompt": p_text,
                "seed": seeds[current_idx],
                "checkpoint_id": checkpoint_ids[current_idx],
                "rating": action,
                "rating_value": rating_val,
                "tags": tags,
                "timestamp": time.time(),
                "time_spent_seconds": round(max(0.1, elapsed), 2),
                "rater_id": rater_id,
                "status": "valid",
            }
            append_feedback_record(rec, log_path)
            session_history.append(current_idx)
            rated_count += 1
            current_idx += 1
            start_time = time.time()
        else:
            # Skip
            current_idx += 1
            start_time = time.time()

    hygiene = analyze_rater_hygiene(log_path)
    return {
        "status": "completed",
        "rated_count": rated_count,
        "total_active_ratings": hygiene["total_ratings"],
        "hygiene": hygiene,
        "log_path": log_path,
    }


def compare_and_rate(
    pairs: List[Tuple[str, str, str, str]],  # (gen_id_a, video_a_path, gen_id_b, video_b_path)
    prompts: List[str],
    log_path: str = "feedback_pairs.jsonl",
    rater_id: str = "default_rater",
    backend: Optional[Union[FakeWidgetBackend, CliBackend]] = None,
) -> Dict[str, Any]:
    """
    Shows two videos for the same prompt side by side and records preference: A, B, or neither.
    Records preference pairs in JSONL format.
    """
    num_pairs = len(pairs)
    assert len(prompts) == num_pairs, "Mismatch between pairs and prompts count"

    # Fallback to CLI or fake backend if ipywidgets is missing
    if backend is None:
        try:
            import ipywidgets as widgets
            from IPython.display import display
            # In real Jupyter
            backend = None
        except ImportError:
            backend = CliBackend()

    if backend is not None:
        recorded = 0
        start_time = time.time()
        for idx in range(num_pairs):
            gid_a, v_a, gid_b, v_b = pairs[idx]
            p_text = prompts[idx]

            pair_info = {
                "index": idx,
                "total": num_pairs,
                "prompt": p_text,
                "gen_id_a": gid_a,
                "video_a_path": v_a,
                "gen_id_b": gid_b,
                "video_b_path": v_b,
            }

            resp = backend.display_pair(pair_info)
            choice = resp.get("action", "neither")
            elapsed = time.time() - start_time

            rec = {
                "comparison_id": f"comp_{uuid.uuid4().hex[:8]}",
                "prompt": p_text,
                "gen_id_a": gid_a,
                "gen_id_b": gid_b,
                "video_a_path": v_a,
                "video_b_path": v_b,
                "choice": choice,  # "A" | "B" | "neither" | "tie"
                "timestamp": time.time(),
                "time_spent_seconds": round(max(0.1, elapsed), 2),
                "rater_id": rater_id,
            }
            append_feedback_record(rec, log_path)
            recorded += 1
            start_time = time.time()

        return {"status": "completed", "recorded_pairs": recorded, "log_path": log_path}

    # Native ipywidgets implementation
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output

    out_container = widgets.Output()
    current_idx = 0
    start_time = [time.time()]

    def render_current_pair():
        nonlocal current_idx
        with out_container:
            clear_output(wait=True)
            if current_idx >= num_pairs:
                display(HTML("<h3>🎉 All pairwise comparisons completed!</h3>"))
                return

            gid_a, v_a, gid_b, v_b = pairs[current_idx]
            p_text = prompts[current_idx]

            display(HTML(f"<h4>Comparison {current_idx + 1} of {num_pairs}</h4>"))
            display(HTML(f"<p><b>Prompt:</b> {p_text}</p>"))

            # Two side-by-side video containers
            v1_html = f"<div style='flex: 1; padding: 10px; background: #0f172a; border-radius: 8px;'><p><b>Option A</b> ({gid_a})</p><video width='280' controls autoplay loop><source src='{v_a}' type='video/mp4'></video></div>"
            v2_html = f"<div style='flex: 1; padding: 10px; background: #0f172a; border-radius: 8px;'><p><b>Option B</b> ({gid_b})</p><video width='280' controls autoplay loop><source src='{v_b}' type='video/mp4'></video></div>"
            display(HTML(f"<div style='display: flex; gap: 15px;'>{v1_html}{v2_html}</div>"))

            btn_a = widgets.Button(description="👈 Option A is Better", button_style="success")
            btn_b = widgets.Button(description="Option B is Better 👉", button_style="primary")
            btn_neither = widgets.Button(description="🤷 Neither / Tie", button_style="warning")

            btn_box = widgets.HBox([btn_a, btn_b, btn_neither])

            def record_choice(chosen: str):
                nonlocal current_idx
                elapsed = time.time() - start_time[0]
                rec = {
                    "comparison_id": f"comp_{uuid.uuid4().hex[:8]}",
                    "prompt": p_text,
                    "gen_id_a": gid_a,
                    "gen_id_b": gid_b,
                    "video_a_path": v_a,
                    "video_b_path": v_b,
                    "choice": chosen,
                    "timestamp": time.time(),
                    "time_spent_seconds": round(max(0.1, elapsed), 2),
                    "rater_id": rater_id,
                }
                append_feedback_record(rec, log_path)
                current_idx += 1
                start_time[0] = time.time()
                render_current_pair()

            btn_a.on_click(lambda _: record_choice("A"))
            btn_b.on_click(lambda _: record_choice("B"))
            btn_neither.on_click(lambda _: record_choice("neither"))
            display(btn_box)

    display(out_container)
    render_current_pair()
    return {"status": "interactive_ipywidgets", "total_pairs": num_pairs, "log_path": log_path}
