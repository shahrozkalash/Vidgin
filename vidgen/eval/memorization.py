"""Video Memorization and Overfitting Check Module (Part 10).

Goal:
Embed generated videos and query nearest neighbors in the training set to detect
verbatim replication vs. creative generalization.

Method:
- Extract spatiotemporal video fingerprint embeddings (L2-normalized).
- Maintain an index of training set clip embeddings.
- Query nearest neighbors via Cosine Similarity and L2 distance.
- Flag potential memorization if top-1 similarity > threshold (default: 0.95).
- Output novelty score: 1.0 - max(similarity).
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def extract_video_embedding(video: torch.Tensor, embedding_dim: int = 128) -> np.ndarray:
    """Extract a compact, permutation-aware spatio-temporal fingerprint for a video clip.

    video: Tensor of shape [B, C, T, H, W] or [C, T, H, W]
    returns: Normalized numpy vector of shape [embedding_dim]
    """
    if video.ndim == 4:
        video = video.unsqueeze(0)
    B, C, T, H, W = video.shape

    with torch.no_grad():
        # Spatial pooling across 4x4 quadrants
        quadrants = F.adaptive_avg_pool3d(video, (min(T, 8), 4, 4))  # [B, C, T', 4, 4]
        # Temporal difference component (motion signature)
        if quadrants.shape[2] > 1:
            motion_sig = torch.abs(quadrants[:, :, 1:] - quadrants[:, :, :-1]).mean(dim=2)  # [B, C, 4, 4]
        else:
            motion_sig = quadrants[:, :, 0]

        # Combine spatial representation with motion signature
        combined = torch.cat([quadrants.mean(dim=2), motion_sig], dim=1)  # [B, 2*C, 4, 4]
        flat = combined.reshape(B, -1)

        # Adaptive projection to embedding_dim
        if flat.shape[-1] != embedding_dim:
            projected = F.adaptive_avg_pool1d(flat.unsqueeze(1), embedding_dim).squeeze(1)
        else:
            projected = flat

        # L2 normalization
        normed = F.normalize(projected, p=2, dim=-1)
        return normed[0].cpu().numpy()


class VideoMemorizationDetector:
    """Detects training set memorization by indexing training video representations

    and querying newly generated videos.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        warning_threshold: float = 0.90,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.warning_threshold = warning_threshold
        self.training_embeddings: List[np.ndarray] = []
        self.training_metadata: List[Dict[str, Any]] = []

    def index_training_clip(
        self,
        clip_id: str,
        video: Optional[torch.Tensor] = None,
        embedding: Optional[np.ndarray] = None,
        caption: str = "",
    ) -> None:
        """Add a training video clip to the memorization index."""
        if embedding is None:
            if video is None:
                raise ValueError("Must provide either video or embedding.")
            embedding = extract_video_embedding(video)

        self.training_embeddings.append(embedding)
        self.training_metadata.append({
            "clip_id": clip_id,
            "caption": caption,
            "index": len(self.training_embeddings) - 1,
        })

    def query(
        self,
        generated_video: Optional[torch.Tensor] = None,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Query nearest neighbors in the training set for a generated video."""
        if query_embedding is None:
            if generated_video is None:
                raise ValueError("Must provide either generated_video or query_embedding.")
            query_embedding = extract_video_embedding(generated_video)

        if not self.training_embeddings:
            return {
                "top_similarity": 0.0,
                "novelty_score": 1.0,
                "is_memorized": False,
                "status": "NO_TRAINING_DATA_INDEXED",
                "nearest_neighbors": [],
            }

        train_matrix = np.stack(self.training_embeddings, axis=0)  # [N, D]
        # Cosine similarity (since vectors are L2-normalized, dot product == cosine similarity)
        similarities = np.dot(train_matrix, query_embedding)  # [N]
        # Euclidean L2 distance = sqrt(2 - 2 * cos_sim)
        l2_distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarities))

        top_indices = np.argsort(similarities)[::-1][:top_k]

        nearest = []
        for rank, idx in enumerate(top_indices):
            meta = self.training_metadata[idx]
            sim = float(similarities[idx])
            dist = float(l2_distances[idx])
            nearest.append({
                "rank": rank + 1,
                "clip_id": meta["clip_id"],
                "caption": meta["caption"],
                "cosine_similarity": round(sim, 4),
                "l2_distance": round(dist, 4),
            })

        top_sim = float(similarities[top_indices[0]])
        novelty = round(float(max(0.0, 1.0 - top_sim)), 4)
        is_memorized = top_sim >= self.similarity_threshold
        is_warning = top_sim >= self.warning_threshold and not is_memorized

        if is_memorized:
            status = "CRITICAL_MEMORIZATION_DETECTED"
            recommendation = "Generated video replicates training clip. Increase data augmentation or regularize DiT backbone."
        elif is_warning:
            status = "HIGH_SIMILARITY_WARNING"
            recommendation = "Video exhibits notable resemblance to training data. Check caption diversity."
        else:
            status = "HEALTHY_GENERALIZATION"
            recommendation = "Novel video generation successfully interpolates latent manifolds."

        return {
            "top_similarity": round(top_sim, 4),
            "novelty_score": novelty,
            "is_memorized": is_memorized,
            "status": status,
            "recommendation": recommendation,
            "nearest_neighbors": nearest,
            "thresholds": {
                "similarity_threshold": self.similarity_threshold,
                "warning_threshold": self.warning_threshold,
            },
        }
