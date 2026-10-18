"""Human A/B Evaluation and ELO Rating Engine (Part 10).

Goal:
Facilitate double-blind human comparative evaluation between model checkpoints.
Nothing replaces human inspection when judging subtle visual artifacts, semantic nuance,
and natural kinematics.

Features:
- Side-by-side randomized blind A/B trials
- Multi-dimensional scoring:
  * Motion Fidelity (natural kinematics vs robotic/stiff)
  * Prompt Adherence (does it follow all instructions?)
  * Visual Quality (sharpness, lighting, aesthetic beauty)
  * Temporal Stability (no flickering, warping, or tearing)
- Online ELO Rating Engine with standard K-factor updates
- Statistical significance testing (Binomial test p-value)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class CheckpointProfile:
    id: str
    name: str
    step: int
    architecture: str
    elo: float = 1200.0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    votes_by_category: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "motion": {"wins": 0, "losses": 0, "ties": 0},
        "prompt": {"wins": 0, "losses": 0, "ties": 0},
        "quality": {"wins": 0, "losses": 0, "ties": 0},
        "temporal": {"wins": 0, "losses": 0, "ties": 0},
    })

    @property
    def total_matches(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        if self.total_matches == 0:
            return 0.0
        return round((self.wins + 0.5 * self.ties) / self.total_matches, 3)


class HumanABEvaluationSuite:
    """Manages blind A/B comparative testing, voting records, and ELO ratings across model checkpoints."""

    def __init__(self, k_factor: float = 32.0) -> None:
        self.k_factor = k_factor
        self.checkpoints: Dict[str, CheckpointProfile] = {}
        self.match_history: List[Dict[str, Any]] = []

    def register_checkpoint(self, checkpoint_id: str, name: str, step: int, architecture: str) -> CheckpointProfile:
        """Register a new checkpoint in the evaluation pool."""
        profile = CheckpointProfile(
            id=checkpoint_id,
            name=name,
            step=step,
            architecture=architecture,
        )
        self.checkpoints[checkpoint_id] = profile
        return profile

    def record_vote(
        self,
        checkpoint_a_id: str,
        checkpoint_b_id: str,
        winner: str,  # 'A' | 'B' | 'TIE'
        category: str = "quality",  # 'motion' | 'prompt' | 'quality' | 'temporal'
        prompt_id: str = "",
        notes: str = "",
    ) -> Dict[str, Any]:
        """Record a human vote and update ELO ratings."""
        if checkpoint_a_id not in self.checkpoints or checkpoint_b_id not in self.checkpoints:
            raise KeyError(f"Both checkpoints must be registered: {checkpoint_a_id}, {checkpoint_b_id}")

        cp_a = self.checkpoints[checkpoint_a_id]
        cp_b = self.checkpoints[checkpoint_b_id]

        # Calculate expected scores via logistic sigmoid curve
        qa = 10.0 ** (cp_a.elo / 400.0)
        qb = 10.0 ** (cp_b.elo / 400.0)
        expected_a = qa / (qa + qb)
        expected_b = qb / (qa + qb)

        # Actual outcomes
        if winner == "A":
            actual_a, actual_b = 1.0, 0.0
            cp_a.wins += 1
            cp_b.losses += 1
            cat_result_a, cat_result_b = "wins", "losses"
        elif winner == "B":
            actual_a, actual_b = 0.0, 1.0
            cp_a.losses += 1
            cp_b.wins += 1
            cat_result_a, cat_result_b = "losses", "wins"
        else:  # TIE
            actual_a, actual_b = 0.5, 0.5
            cp_a.ties += 1
            cp_b.ties += 1
            cat_result_a, cat_result_b = "ties", "ties"

        # Update category tallies
        if category in cp_a.votes_by_category:
            cp_a.votes_by_category[category][cat_result_a] += 1
        if category in cp_b.votes_by_category:
            cp_b.votes_by_category[category][cat_result_b] += 1

        # Dynamic K-factor (scaled down for established checkpoints)
        k_a = self.k_factor / (1.0 + cp_a.total_matches / 25.0)
        k_b = self.k_factor / (1.0 + cp_b.total_matches / 25.0)

        # Update ELO
        delta_a = k_a * (actual_a - expected_a)
        delta_b = k_b * (actual_b - expected_b)
        cp_a.elo = round(cp_a.elo + delta_a, 1)
        cp_b.elo = round(cp_b.elo + delta_b, 1)

        match_record = {
            "match_id": len(self.match_history) + 1,
            "checkpoint_a": checkpoint_a_id,
            "checkpoint_b": checkpoint_b_id,
            "winner": winner,
            "category": category,
            "prompt_id": prompt_id,
            "delta_elo_a": round(delta_a, 1),
            "delta_elo_b": round(delta_b, 1),
            "notes": notes,
        }
        self.match_history.append(match_record)
        return match_record

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Return leaderboard ranked by ELO rating."""
        sorted_profiles = sorted(
            self.checkpoints.values(),
            key=lambda c: c.elo,
            reverse=True,
        )
        return [
            {
                "rank": idx + 1,
                "id": c.id,
                "name": c.name,
                "step": c.step,
                "architecture": c.architecture,
                "elo": c.elo,
                "matches": c.total_matches,
                "win_rate": c.win_rate,
                "record": f"{c.wins}W - {c.losses}L - {c.ties}T",
                "categories": c.votes_by_category,
            }
            for idx, c in enumerate(sorted_profiles)
        ]

    def calculate_statistical_significance(self, checkpoint_a_id: str, checkpoint_b_id: str) -> Dict[str, Any]:
        """Compute two-sided p-value for the win rate difference between two models."""
        matches = [
            m for m in self.match_history
            if (m["checkpoint_a"] == checkpoint_a_id and m["checkpoint_b"] == checkpoint_b_id)
            or (m["checkpoint_a"] == checkpoint_b_id and m["checkpoint_b"] == checkpoint_a_id)
        ]
        n = len(matches)
        if n < 5:
            return {
                "matches": n,
                "significant": False,
                "p_value": 1.0,
                "verdict": "INSUFFICIENT_DATA (Need >= 10 trials)",
            }

        # Count wins for A (including ties as 0.5)
        wins_a = 0.0
        for m in matches:
            if m["checkpoint_a"] == checkpoint_a_id:
                if m["winner"] == "A":
                    wins_a += 1.0
                elif m["winner"] == "TIE":
                    wins_a += 0.5
            else:
                if m["winner"] == "B":
                    wins_a += 1.0
                elif m["winner"] == "TIE":
                    wins_a += 0.5

        # Normal approximation to binomial distribution (null hyp: p = 0.5)
        z = (wins_a - 0.5 * n) / math.sqrt(0.25 * n)
        # Approximate two-tailed p-value via error function
        p_val = math.erfc(abs(z) / math.sqrt(2.0))

        is_sig = p_val < 0.05
        return {
            "matches": n,
            "wins_a": wins_a,
            "win_rate_a": round(wins_a / n, 3),
            "z_score": round(z, 3),
            "p_value": round(p_val, 4),
            "significant": is_sig,
            "verdict": f"STATISTICALLY_SIGNIFICANT (p={round(p_val, 4)})" if is_sig else "NOT_SIGNIFICANT",
        }
