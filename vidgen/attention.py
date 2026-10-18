"""
Vidgen Decoupled Spatial/Temporal and Hybrid Attention Engine (Seedance Batch 1)
Supports:
1. attention_mode: "full" | "decoupled" | "hybrid" (full attention every N layers).
2. Spatial layer:
   - Attention inside each latent frame.
   - Text tokens joined into the same self-attention.
   - Separate Q/K/V weights for visual and text tokens.
3. Temporal layer:
   - Attention across frames at each spatial position (h, w).
   - Visual tokens only.
   - Configurable time window.
4. Block-causal chunked generation and KV cache with zero future leakage.
5. 3D Rotary Position Embeddings (RoPE) partitioned across (t, h, w).
6. Analytical FLOPs and peak VRAM memory estimator for 5s and 30s video lengths.
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Any, Union
from vidgen.config import AttentionConfig, PresetConfig, TINY_PRESET


def vector_dot(u: List[float], v: List[float]) -> float:
    """Dot product of two vectors."""
    return sum(a * b for a, b in zip(u, v))


def vector_add(u: List[float], v: List[float]) -> List[float]:
    """Element-wise addition of two vectors."""
    return [a + b for a, b in zip(u, v)]


def vector_sub(u: List[float], v: List[float]) -> List[float]:
    """Element-wise subtraction of two vectors."""
    return [a - b for a, b in zip(u, v)]


def vector_scale(v: List[float], scalar: float) -> List[float]:
    """Scale a vector by a scalar."""
    return [x * scalar for x in v]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication C = A @ B."""
    rows_A = len(A)
    cols_A = len(A[0]) if rows_A > 0 else 0
    rows_B = len(B)
    cols_B = len(B[0]) if rows_B > 0 else 0
    assert cols_A == rows_B, f"Shape mismatch for matmul: ({rows_A}, {cols_A}) vs ({rows_B}, {cols_B})"

    # Transpose B for memory and cache friendly dot products
    B_T = [[B[r][c] for r in range(rows_B)] for c in range(cols_B)]
    C = []
    for r in range(rows_A):
        row_A = A[r]
        row_C = [sum(row_A[k] * B_T[c][k] for k in range(cols_A)) for c in range(cols_B)]
        C.append(row_C)
    return C


def softmax(logits: List[float], mask: Optional[List[bool]] = None) -> List[float]:
    """
    Numerically stable softmax with optional boolean mask (True = attend, False = ignore).
    """
    filtered_logits = []
    for i, val in enumerate(logits):
        if mask is not None and not mask[i]:
            filtered_logits.append(-1e9)
        else:
            filtered_logits.append(val)

    max_val = max(filtered_logits) if filtered_logits else 0.0
    exp_vals = [math.exp(v - max_val) if v > -1e8 else 0.0 for v in filtered_logits]
    sum_exp = sum(exp_vals)
    if sum_exp < 1e-12:
        # Uniform fallback if all masked
        count = len(logits)
        return [1.0 / count] * count if count > 0 else []
    return [v / sum_exp for v in exp_vals]


def get_rope_partition(head_dim: int) -> Tuple[int, int, int]:
    """
    Partition head_dim into (dt, dh, dw) with all even components.
    - head_dim = 128 -> (44, 42, 42)
    - head_dim = 64  -> (22, 22, 20)
    - custom head_dim -> even split
    """
    if head_dim == 128:
        return (44, 42, 42)
    elif head_dim == 64:
        return (22, 22, 20)
    else:
        dt = (head_dim // 3) // 2 * 2
        dh = (head_dim // 3) // 2 * 2
        dw = head_dim - dt - dh
        if dw % 2 != 0:
            dw -= 1
            dt += 1
        return (dt, dh, dw)


def compute_rope_freqs(dim: int, base: float = 10000.0) -> List[float]:
    """Compute RoPE inverse frequencies."""
    freqs = []
    for i in range(0, dim, 2):
        freqs.append(1.0 / (base ** (i / dim)))
    return freqs


def apply_rope_1d(
    vector: List[float],
    pos: int,
    freqs: Optional[List[float]] = None
) -> List[float]:
    """Apply 1D RoPE Givens rotation along pairs of coordinates."""
    dim = len(vector)
    assert dim % 2 == 0, f"Dimension {dim} must be even for RoPE"
    if freqs is None:
        freqs = compute_rope_freqs(dim)

    out = list(vector)
    for i in range(0, dim, 2):
        theta = pos * freqs[i // 2]
        c = math.cos(theta)
        s = math.sin(theta)
        v0 = vector[i]
        v1 = vector[i + 1]
        out[i] = v0 * c - v1 * s
        out[i + 1] = v0 * s + v1 * c
    return out


def apply_rope_3d(
    head_vector: List[float],
    t: int,
    h: int,
    w: int,
    partition: Optional[Tuple[int, int, int]] = None,
    freqs_t: Optional[List[float]] = None,
    freqs_h: Optional[List[float]] = None,
    freqs_w: Optional[List[float]] = None,
) -> List[float]:
    """
    Apply 3D RoPE to a single attention head vector.
    Coordinates: t (temporal), h (height), w (width).
    Vector is partitioned into dt, dh, dw dimensions.
    """
    head_dim = len(head_vector)
    dt, dh, dw = partition or get_rope_partition(head_dim)
    assert dt + dh + dw == head_dim, f"Partition ({dt}, {dh}, {dw}) != {head_dim}"

    v_t = head_vector[:dt]
    v_h = head_vector[dt : dt + dh]
    v_w = head_vector[dt + dh :]

    rot_t = apply_rope_1d(v_t, t, freqs_t)
    rot_h = apply_rope_1d(v_h, h, freqs_h)
    rot_w = apply_rope_1d(v_w, w, freqs_w)

    return rot_t + rot_h + rot_w


class Linear:
    """Standard Linear layer: y = x @ W + b."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, seed: int = 42):
        self.in_features = in_features
        self.out_features = out_features
        scale = 1.0 / math.sqrt(max(1, in_features))
        self.weight: List[List[float]] = []
        for i in range(in_features):
            row = [math.sin((i + 1) * 0.17 + (j + 1) * 0.31 + seed * 0.05) * scale for j in range(out_features)]
            self.weight.append(row)
        if bias:
            self.bias: Optional[List[float]] = [0.0] * out_features
        else:
            self.bias = None

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        # x is [N, in_features]
        out = matmul(x, self.weight)
        if self.bias is not None:
            out = [[row[j] + self.bias[j] for j in range(self.out_features)] for row in out]
        return out


class SpatialAttention:
    """
    Decoupled Spatial Attention Layer:
    - Runs attention inside each latent frame t in [0, T-1].
    - Visual tokens at frame t attend to each other and to the text tokens.
    - Separate Q/K/V linear weights for visual and text tokens.
    - Visual queries attend to joint visual + text keys.
    - Visual output depends strictly on frame t and text tokens.
    """
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        head_dim: int = 64,
        seed: int = 101,
        separate_text_qkv: bool = True,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.partition = get_rope_partition(head_dim)
        self.separate_text_qkv = separate_text_qkv

        dt, dh, dw = self.partition
        self.freqs_t = compute_rope_freqs(dt)
        self.freqs_h = compute_rope_freqs(dh)
        self.freqs_w = compute_rope_freqs(dw)

        # Visual Q/K/V and Out projections
        self.q_proj_v = Linear(dim, dim, bias=True, seed=seed)
        self.k_proj_v = Linear(dim, dim, bias=True, seed=seed + 1)
        self.v_proj_v = Linear(dim, dim, bias=True, seed=seed + 2)
        self.out_proj_v = Linear(dim, dim, bias=True, seed=seed + 3)

        # Separate Text Q/K/V and Out projections
        self.q_proj_t = Linear(dim, dim, bias=True, seed=seed + 10)
        self.k_proj_t = Linear(dim, dim, bias=True, seed=seed + 11)
        self.v_proj_t = Linear(dim, dim, bias=True, seed=seed + 12)
        self.out_proj_t = Linear(dim, dim, bias=True, seed=seed + 13)

    def _split_heads(self, x: List[List[float]]) -> List[List[List[float]]]:
        """
        Split [N, dim] -> [num_heads, N, head_dim].
        """
        N = len(x)
        heads = []
        for h in range(self.num_heads):
            head_tokens = []
            for i in range(N):
                start = h * self.head_dim
                head_tokens.append(x[i][start : start + self.head_dim])
            heads.append(head_tokens)
        return heads

    def _merge_heads(self, heads: List[List[List[float]]]) -> List[List[float]]:
        """
        Merge [num_heads, N, head_dim] -> [N, dim].
        """
        N = len(heads[0])
        merged = []
        for i in range(N):
            row = []
            for h in range(self.num_heads):
                row.extend(heads[h][i])
            merged.append(row)
        return merged

    def forward_frame(
        self,
        frame_visual: List[List[float]],
        spatial_coords: List[Tuple[int, int]],
        t_pos: int,
        text_tokens: Optional[List[List[float]]] = None,
    ) -> Tuple[List[List[float]], Optional[List[List[float]]]]:
        """
        Compute spatial self-attention inside a single temporal frame t.
        frame_visual: list of S visual tokens at frame t, each of length dim.
        spatial_coords: list of S coordinates (h, w).
        text_tokens: list of L text tokens of length dim.
        """
        S = len(frame_visual)
        L = len(text_tokens) if text_tokens is not None else 0

        # Visual projections
        Q_v = self.q_proj_v.forward(frame_visual)
        K_v = self.k_proj_v.forward(frame_visual)
        V_v = self.v_proj_v.forward(frame_visual)

        # Split into heads
        Q_v_heads = self._split_heads(Q_v)
        K_v_heads = self._split_heads(K_v)
        V_v_heads = self._split_heads(V_v)

        # Apply 3D RoPE to visual queries and keys with coordinate (t, h, w)
        for h_idx in range(self.num_heads):
            for s in range(S):
                h, w = spatial_coords[s]
                Q_v_heads[h_idx][s] = apply_rope_3d(
                    Q_v_heads[h_idx][s], t=t_pos, h=h, w=w,
                    partition=self.partition,
                    freqs_t=self.freqs_t, freqs_h=self.freqs_h, freqs_w=self.freqs_w
                )
                K_v_heads[h_idx][s] = apply_rope_3d(
                    K_v_heads[h_idx][s], t=t_pos, h=h, w=w,
                    partition=self.partition,
                    freqs_t=self.freqs_t, freqs_h=self.freqs_h, freqs_w=self.freqs_w
                )

        # Text projections if provided
        Q_t_heads, K_t_heads, V_t_heads = None, None, None
        if L > 0 and text_tokens is not None:
            Q_t = self.q_proj_t.forward(text_tokens)
            K_t = self.k_proj_t.forward(text_tokens)
            V_t = self.v_proj_t.forward(text_tokens)
            Q_t_heads = self._split_heads(Q_t)
            K_t_heads = self._split_heads(K_t)
            V_t_heads = self._split_heads(V_t)

        out_v_heads = []
        out_t_heads = [] if L > 0 else None

        for h_idx in range(self.num_heads):
            # Joint keys and values: [visual_tokens; text_tokens]
            if L > 0 and K_t_heads is not None and V_t_heads is not None:
                joint_K = K_v_heads[h_idx] + K_t_heads[h_idx]
                joint_V = V_v_heads[h_idx] + V_t_heads[h_idx]
            else:
                joint_K = K_v_heads[h_idx]
                joint_V = V_v_heads[h_idx]

            K_len = len(joint_K)

            # Compute visual query attention over joint keys
            head_out_v = []
            for s in range(S):
                q = Q_v_heads[h_idx][s]
                scores = [vector_dot(q, joint_K[k]) * self.scale for k in range(K_len)]
                weights = softmax(scores)
                # Weighted sum of joint_V
                context = [0.0] * self.head_dim
                for k in range(K_len):
                    w_k = weights[k]
                    if w_k > 0:
                        v_k = joint_V[k]
                        for d in range(self.head_dim):
                            context[d] += w_k * v_k[d]
                head_out_v.append(context)
            out_v_heads.append(head_out_v)

            # Compute text output if text queries are present
            if L > 0 and Q_t_heads is not None and out_t_heads is not None:
                head_out_t = []
                for l in range(L):
                    q = Q_t_heads[h_idx][l]
                    scores = [vector_dot(q, joint_K[k]) * self.scale for k in range(K_len)]
                    weights = softmax(scores)
                    context = [0.0] * self.head_dim
                    for k in range(K_len):
                        w_k = weights[k]
                        if w_k > 0:
                            v_k = joint_V[k]
                            for d in range(self.head_dim):
                                context[d] += w_k * v_k[d]
                    head_out_t.append(context)
                out_t_heads.append(head_out_t)

        # Merge heads & project
        merged_v = self._merge_heads(out_v_heads)
        projected_v = self.out_proj_v.forward(merged_v)

        projected_t = None
        if L > 0 and out_t_heads is not None:
            merged_t = self._merge_heads(out_t_heads)
            projected_t = self.out_proj_t.forward(merged_t)

        return projected_v, projected_t

    def forward(
        self,
        frames_visual: List[List[List[float]]],
        spatial_coords: List[Tuple[int, int]],
        text_tokens: Optional[List[List[float]]] = None,
    ) -> Tuple[List[List[List[float]]], Optional[List[List[float]]]]:
        """
        Forward over all T frames.
        frames_visual: [T, S, dim]
        spatial_coords: [S] list of (h, w) coordinates.
        text_tokens: [L, dim]
        """
        T = len(frames_visual)
        out_frames = []
        last_out_text = None
        for t in range(T):
            v_t, t_out = self.forward_frame(
                frame_visual=frames_visual[t],
                spatial_coords=spatial_coords,
                t_pos=t,
                text_tokens=text_tokens,
            )
            out_frames.append(v_t)
            if t_out is not None:
                last_out_text = t_out
        return out_frames, last_out_text


class TemporalAttention:
    """
    Decoupled Temporal Attention Layer:
    - Runs attention across frames for each spatial position (h, w).
    - Visual tokens only (text tokens are not included in temporal attention).
    - Configurable time window.
    - Supports block-causal chunked generation and KV cache for streaming long video.
    """
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        head_dim: int = 64,
        seed: int = 202,
        temporal_window_size: Optional[int] = None,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.temporal_window_size = temporal_window_size

        self.partition = get_rope_partition(head_dim)
        dt, _, _ = self.partition
        self.freqs_t = compute_rope_freqs(dt)

        self.q_proj = Linear(dim, dim, bias=True, seed=seed)
        self.k_proj = Linear(dim, dim, bias=True, seed=seed + 1)
        self.v_proj = Linear(dim, dim, bias=True, seed=seed + 2)
        self.out_proj = Linear(dim, dim, bias=True, seed=seed + 3)

    def _split_heads(self, x: List[List[float]]) -> List[List[List[float]]]:
        N = len(x)
        heads = []
        for h in range(self.num_heads):
            head_tokens = []
            for i in range(N):
                start = h * self.head_dim
                head_tokens.append(x[i][start : start + self.head_dim])
            heads.append(head_tokens)
        return heads

    def _merge_heads(self, heads: List[List[List[float]]]) -> List[List[float]]:
        N = len(heads[0])
        merged = []
        for i in range(N):
            row = []
            for h in range(self.num_heads):
                row.extend(heads[h][i])
            merged.append(row)
        return merged

    def forward_column(
        self,
        column_tokens: List[List[float]],
        time_indices: List[int],
        cached_keys: Optional[List[List[List[float]]]] = None,
        cached_values: Optional[List[List[List[float]]]] = None,
        cached_time_indices: Optional[List[int]] = None,
        chunk_causal_mask: Optional[List[List[bool]]] = None,
    ) -> Tuple[List[List[float]], List[List[List[float]]], List[List[List[float]]]]:
        """
        Temporal attention along a single spatial coordinate (h, w) across time.
        column_tokens: list of T tokens for this spatial position.
        time_indices: list of temporal positions for column_tokens.
        cached_keys, cached_values: [num_heads, cached_len, head_dim]
        Returns:
            (out_column, new_k_heads, new_v_heads)
        """
        T = len(column_tokens)
        Q = self.q_proj.forward(column_tokens)
        K = self.k_proj.forward(column_tokens)
        V = self.v_proj.forward(column_tokens)

        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        dt, _, _ = self.partition
        # Apply 1D temporal RoPE to queries and keys
        for h_idx in range(self.num_heads):
            for t_idx in range(T):
                t_val = time_indices[t_idx]
                # Rotate first dt elements with t_val
                q_head = Q_heads[h_idx][t_idx]
                k_head = K_heads[h_idx][t_idx]
                rot_q_t = apply_rope_1d(q_head[:dt], t_val, self.freqs_t)
                rot_k_t = apply_rope_1d(k_head[:dt], t_val, self.freqs_t)
                Q_heads[h_idx][t_idx] = rot_q_t + q_head[dt:]
                K_heads[h_idx][t_idx] = rot_k_t + k_head[dt:]

        # Merge with cached keys/values if present
        out_heads = []
        new_k_heads = []
        new_v_heads = []

        has_cache = cached_keys is not None and cached_values is not None and len(cached_keys[0]) > 0
        cache_len = len(cached_keys[0]) if has_cache else 0
        total_time_indices = (cached_time_indices or []) + time_indices

        for h_idx in range(self.num_heads):
            curr_K = K_heads[h_idx]
            curr_V = V_heads[h_idx]

            if has_cache and cached_keys is not None and cached_values is not None:
                full_K = cached_keys[h_idx] + curr_K
                full_V = cached_values[h_idx] + curr_V
            else:
                full_K = curr_K
                full_V = curr_V

            new_k_heads.append(full_K)
            new_v_heads.append(full_V)

            total_K_len = len(full_K)
            head_out = []

            for q_idx in range(T):
                q = Q_heads[h_idx][q_idx]
                q_time = time_indices[q_idx]

                scores = [vector_dot(q, full_K[k_idx]) * self.scale for k_idx in range(total_K_len)]

                # Apply temporal window or chunk causal mask
                mask = [True] * total_K_len
                for k_idx in range(total_K_len):
                    k_time = total_time_indices[k_idx]
                    # Temporal window restriction
                    if self.temporal_window_size is not None:
                        if abs(q_time - k_time) > self.temporal_window_size:
                            mask[k_idx] = False

                    # Block-causal restriction: never attend to future time frames!
                    if k_time > q_time:
                        mask[k_idx] = False

                    # Custom chunk causal mask if provided
                    if chunk_causal_mask is not None:
                        if not chunk_causal_mask[q_idx][k_idx]:
                            mask[k_idx] = False

                weights = softmax(scores, mask=mask)
                context = [0.0] * self.head_dim
                for k_idx in range(total_K_len):
                    w_k = weights[k_idx]
                    if w_k > 0:
                        v_k = full_V[k_idx]
                        for d in range(self.head_dim):
                            context[d] += w_k * v_k[d]
                head_out.append(context)
            out_heads.append(head_out)

        merged = self._merge_heads(out_heads)
        projected = self.out_proj.forward(merged)
        return projected, new_k_heads, new_v_heads

    def forward(
        self,
        frames_visual: List[List[List[float]]],
        spatial_coords: List[Tuple[int, int]],
        time_indices: Optional[List[int]] = None,
        kv_cache: Optional[Dict[str, Any]] = None,
        chunk_causal_mask: Optional[List[List[bool]]] = None,
    ) -> Tuple[List[List[List[float]]], Optional[Dict[str, Any]]]:
        """
        Forward over all spatial positions (h, w).
        frames_visual: [T, S, dim]
        spatial_coords: [S] list of (h, w) coordinates.
        time_indices: [T] temporal coordinate list (defaults to 0..T-1).
        kv_cache: Optional dictionary holding per-spatial-column KV states.
        """
        T = len(frames_visual)
        S = len(spatial_coords)
        if time_indices is None:
            time_indices = list(range(T))

        # Re-pack [T, S, dim] -> S columns of length T: [S, T, dim]
        columns = [[frames_visual[t][s] for t in range(T)] for s in range(S)]

        new_kv_cache: Optional[Dict[str, Any]] = {} if kv_cache is not None else None
        out_columns = []

        for s in range(S):
            col_key = f"col_{s}"
            c_k = kv_cache.get(f"{col_key}_k") if kv_cache else None
            c_v = kv_cache.get(f"{col_key}_v") if kv_cache else None
            c_t = kv_cache.get("time_indices") if kv_cache else None

            out_col, new_k, new_v = self.forward_column(
                column_tokens=columns[s],
                time_indices=time_indices,
                cached_keys=c_k,
                cached_values=c_v,
                cached_time_indices=c_t,
                chunk_causal_mask=chunk_causal_mask,
            )
            out_columns.append(out_col)

            if new_kv_cache is not None:
                new_kv_cache[f"{col_key}_k"] = new_k
                new_kv_cache[f"{col_key}_v"] = new_v

        if new_kv_cache is not None:
            new_kv_cache["time_indices"] = (kv_cache.get("time_indices", []) if kv_cache else []) + time_indices

        # Re-pack [S, T, dim] -> [T, S, dim]
        out_frames = [[out_columns[s][t] for s in range(S)] for t in range(T)]
        return out_frames, new_kv_cache


class FullSpacetimeAttention:
    """
    Standard Full 3D Spacetime Attention:
    All visual tokens across all frames [T, S] attend to each other and text tokens.
    Uses 3D RoPE and supports block-causal attention across chunks.
    """
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        head_dim: int = 64,
        seed: int = 303,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.partition = get_rope_partition(head_dim)

        dt, dh, dw = self.partition
        self.freqs_t = compute_rope_freqs(dt)
        self.freqs_h = compute_rope_freqs(dh)
        self.freqs_w = compute_rope_freqs(dw)

        self.q_proj = Linear(dim, dim, bias=True, seed=seed)
        self.k_proj = Linear(dim, dim, bias=True, seed=seed + 1)
        self.v_proj = Linear(dim, dim, bias=True, seed=seed + 2)
        self.out_proj = Linear(dim, dim, bias=True, seed=seed + 3)

    def _split_heads(self, x: List[List[float]]) -> List[List[List[float]]]:
        N = len(x)
        heads = []
        for h in range(self.num_heads):
            head_tokens = []
            for i in range(N):
                start = h * self.head_dim
                head_tokens.append(x[i][start : start + self.head_dim])
            heads.append(head_tokens)
        return heads

    def _merge_heads(self, heads: List[List[List[float]]]) -> List[List[float]]:
        N = len(heads[0])
        merged = []
        for i in range(N):
            row = []
            for h in range(self.num_heads):
                row.extend(heads[h][i])
            merged.append(row)
        return merged

    def forward(
        self,
        frames_visual: List[List[List[float]]],
        spatial_coords: List[Tuple[int, int]],
        text_tokens: Optional[List[List[float]]] = None,
        time_indices: Optional[List[int]] = None,
        chunk_indices: Optional[List[int]] = None,  # chunk ID per frame
        window_chunks: int = 1,
        has_anchor: bool = True,
    ) -> Tuple[List[List[List[float]]], Optional[List[List[float]]]]:
        """
        Full 3D spacetime attention across all tokens.
        """
        T = len(frames_visual)
        S = len(spatial_coords)
        L = len(text_tokens) if text_tokens is not None else 0
        if time_indices is None:
            time_indices = list(range(T))

        # Flatten visual tokens: [T * S, dim]
        flat_visual = []
        token_positions = []  # (t, h, w, chunk_id)
        for t in range(T):
            t_val = time_indices[t]
            c_val = chunk_indices[t] if chunk_indices else 0
            for s in range(S):
                flat_visual.append(frames_visual[t][s])
                h, w = spatial_coords[s]
                token_positions.append((t_val, h, w, c_val))

        total_tokens = flat_visual + (text_tokens or [])
        N_tot = len(total_tokens)
        N_vis = len(flat_visual)

        Q = self.q_proj.forward(total_tokens)
        K = self.k_proj.forward(total_tokens)
        V = self.v_proj.forward(total_tokens)

        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        # Apply 3D RoPE to visual queries and keys
        for h_idx in range(self.num_heads):
            for i in range(N_vis):
                t_val, h, w, _ = token_positions[i]
                Q_heads[h_idx][i] = apply_rope_3d(
                    Q_heads[h_idx][i], t=t_val, h=h, w=w,
                    partition=self.partition,
                    freqs_t=self.freqs_t, freqs_h=self.freqs_h, freqs_w=self.freqs_w
                )
                K_heads[h_idx][i] = apply_rope_3d(
                    K_heads[h_idx][i], t=t_val, h=h, w=w,
                    partition=self.partition,
                    freqs_t=self.freqs_t, freqs_h=self.freqs_h, freqs_w=self.freqs_w
                )

        out_heads = []
        for h_idx in range(self.num_heads):
            head_out = []
            for q_idx in range(N_tot):
                q = Q_heads[h_idx][q_idx]
                scores = [vector_dot(q, K_heads[h_idx][k_idx]) * self.scale for k_idx in range(N_tot)]

                # Block-causal mask: queries in chunk c can attend to chunk 0 and chunks in [c - window, c]
                mask = [True] * N_tot
                if q_idx < N_vis:
                    q_chunk = token_positions[q_idx][3]
                    for k_idx in range(N_vis):
                        k_chunk = token_positions[k_idx][3]
                        if k_chunk > q_chunk:
                            mask[k_idx] = False  # Zero future leakage!
                        elif has_anchor and k_chunk == 0:
                            mask[k_idx] = True
                        elif q_chunk - k_chunk > window_chunks:
                            mask[k_idx] = False

                weights = softmax(scores, mask=mask)
                context = [0.0] * self.head_dim
                for k_idx in range(N_tot):
                    w_k = weights[k_idx]
                    if w_k > 0:
                        v_k = V_heads[h_idx][k_idx]
                        for d in range(self.head_dim):
                            context[d] += w_k * v_k[d]
                head_out.append(context)
            out_heads.append(head_out)

        merged = self._merge_heads(out_heads)
        projected = self.out_proj.forward(merged)

        # Unpack visual and text
        out_vis = projected[:N_vis]
        out_text = projected[N_vis:] if L > 0 else None

        # Re-pack visual into [T, S, dim]
        out_frames = []
        for t in range(T):
            start = t * S
            out_frames.append(out_vis[start : start + S])

        return out_frames, out_text


class DiTAttentionBlock:
    """
    DiT Transformer Block supporting:
    - "full": Standard 3D Spacetime Attention.
    - "decoupled": Decoupled Spatial Attention followed by Temporal Attention.
    - "hybrid": Decoupled layers with a full-attention layer every N layers.
    """
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        head_dim: int = 64,
        layer_idx: int = 0,
        config: Optional[AttentionConfig] = None,
        seed: int = 404,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.config = config or AttentionConfig()

        # Determine effective attention type for this layer
        mode = self.config.attention_mode.lower()
        if mode == "hybrid":
            if (layer_idx + 1) % self.config.hybrid_full_attn_interval == 0:
                self.effective_type = "full"
            else:
                self.effective_type = "decoupled"
        elif mode == "decoupled" or self.config.enable_decoupled_attention:
            self.effective_type = "decoupled"
        else:
            self.effective_type = "full"

        # Instantiate attention submodules
        if self.effective_type == "full":
            self.full_attn = FullSpacetimeAttention(
                dim=dim, num_heads=num_heads, head_dim=head_dim, seed=seed + layer_idx * 10
            )
            self.spatial_attn = None
            self.temporal_attn = None
        else:
            self.full_attn = None
            self.spatial_attn = SpatialAttention(
                dim=dim, num_heads=num_heads, head_dim=head_dim, seed=seed + layer_idx * 10,
                separate_text_qkv=self.config.separate_text_qkv
            )
            self.temporal_attn = TemporalAttention(
                dim=dim, num_heads=num_heads, head_dim=head_dim, seed=seed + layer_idx * 10 + 5,
                temporal_window_size=self.config.temporal_window_size
            )

        # FFN MLP layer (with residual connection)
        self.mlp_fc1 = Linear(dim, dim * 2, bias=True, seed=seed + layer_idx * 10 + 20)
        self.mlp_fc2 = Linear(dim * 2, dim, bias=True, seed=seed + layer_idx * 10 + 21)

    def _mlp(self, tokens: List[List[float]]) -> List[List[float]]:
        """GELU-style MLP."""
        hidden = self.mlp_fc1.forward(tokens)
        activated = [[x * 0.5 * (1.0 + math.erf(x / 1.41421356)) for x in row] for row in hidden]
        return self.mlp_fc2.forward(activated)

    def forward(
        self,
        frames_visual: List[List[List[float]]],
        spatial_coords: List[Tuple[int, int]],
        text_tokens: Optional[List[List[float]]] = None,
        time_indices: Optional[List[int]] = None,
        chunk_indices: Optional[List[int]] = None,
        kv_cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[List[List[float]]], Optional[List[List[float]]], Optional[Dict[str, Any]]]:
        """
        Forward pass with residual connections.
        """
        T = len(frames_visual)
        S = len(spatial_coords)

        new_kv_cache = None

        if self.effective_type == "full":
            assert self.full_attn is not None
            out_vis, out_text = self.full_attn.forward(
                frames_visual=frames_visual,
                spatial_coords=spatial_coords,
                text_tokens=text_tokens,
                time_indices=time_indices,
                chunk_indices=chunk_indices,
            )
            # Residual 1
            res1_frames = []
            for t in range(T):
                res1_frames.append([vector_add(frames_visual[t][s], out_vis[t][s]) for s in range(S)])
            res1_text = None
            if text_tokens and out_text:
                res1_text = [vector_add(text_tokens[i], out_text[i]) for i in range(len(text_tokens))]

            # FFN & Residual 2
            final_frames = []
            for t in range(T):
                mlp_out = self._mlp(res1_frames[t])
                final_frames.append([vector_add(res1_frames[t][s], mlp_out[s]) for s in range(S)])

            final_text = res1_text
            if res1_text:
                mlp_t = self._mlp(res1_text)
                final_text = [vector_add(res1_text[i], mlp_t[i]) for i in range(len(res1_text))]

            return final_frames, final_text, None

        else:
            # Decoupled Mode: Spatial -> Temporal -> MLP
            assert self.spatial_attn is not None and self.temporal_attn is not None

            # 1. Spatial Attention (Visual + Text)
            sp_vis, sp_text = self.spatial_attn.forward(
                frames_visual=frames_visual,
                spatial_coords=spatial_coords,
                text_tokens=text_tokens,
            )
            # Residual 1 (Spatial)
            res_sp_frames = []
            for t in range(T):
                res_sp_frames.append([vector_add(frames_visual[t][s], sp_vis[t][s]) for s in range(S)])
            res_sp_text = None
            if text_tokens and sp_text:
                res_sp_text = [vector_add(text_tokens[i], sp_text[i]) for i in range(len(text_tokens))]

            # 2. Temporal Attention (Visual only)
            temp_vis, new_kv_cache = self.temporal_attn.forward(
                frames_visual=res_sp_frames,
                spatial_coords=spatial_coords,
                time_indices=time_indices,
                kv_cache=kv_cache,
            )
            # Residual 2 (Temporal)
            res_temp_frames = []
            for t in range(T):
                res_temp_frames.append([vector_add(res_sp_frames[t][s], temp_vis[t][s]) for s in range(S)])

            # 3. FFN & Residual 3
            final_frames = []
            for t in range(T):
                mlp_out = self._mlp(res_temp_frames[t])
                final_frames.append([vector_add(res_temp_frames[t][s], mlp_out[s]) for s in range(S)])

            final_text = res_sp_text
            if res_sp_text:
                mlp_t = self._mlp(res_sp_text)
                final_text = [vector_add(res_sp_text[i], mlp_t[i]) for i in range(len(res_sp_text))]

            return final_frames, final_text, new_kv_cache


class DecoupledDiTModel:
    """
    DiT Backbone model for long video latent diffusion.
    Combines D layers of DiTAttentionBlock configured as full, decoupled, or hybrid.
    Supports complete sequence and chunk-by-chunk KV-cache execution.
    """
    def __init__(
        self,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        head_dim: int = 64,
        config: Optional[AttentionConfig] = None,
        seed: int = 42,
    ):
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.config = config or AttentionConfig()

        self.blocks: List[DiTAttentionBlock] = []
        for l in range(depth):
            block = DiTAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                layer_idx=l,
                config=self.config,
                seed=seed + l * 50,
            )
            self.blocks.append(block)

    def forward(
        self,
        frames_visual: List[List[List[float]]],
        spatial_coords: List[Tuple[int, int]],
        text_tokens: Optional[List[List[float]]] = None,
        time_indices: Optional[List[int]] = None,
        chunk_indices: Optional[List[int]] = None,
        kv_caches: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[List[List[float]]], Optional[List[List[float]]], Optional[List[Dict[str, Any]]]]:
        """
        Forward pass through all depth blocks.
        """
        curr_vis = frames_visual
        curr_text = text_tokens
        new_caches = [] if kv_caches is not None else None

        for l, block in enumerate(self.blocks):
            layer_cache = kv_caches[l] if kv_caches is not None else None
            curr_vis, curr_text, next_cache = block.forward(
                frames_visual=curr_vis,
                spatial_coords=spatial_coords,
                text_tokens=curr_text,
                time_indices=time_indices,
                chunk_indices=chunk_indices,
                kv_cache=layer_cache,
            )
            if new_caches is not None and next_cache is not None:
                new_caches.append(next_cache)

        return curr_vis, curr_text, new_caches


def estimate_attention_flops_and_memory(
    duration_seconds: float,
    fps: int = 16,
    height: int = 480,
    width: int = 832,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    vae_stride: Tuple[int, int, int] = (4, 8, 8),
    dim: int = 1536,
    depth: int = 30,
    num_heads: int = 12,
    text_tokens: int = 77,
    attention_mode: str = "decoupled",
    hybrid_interval: int = 4,
    temporal_window_size: Optional[int] = None,
    chunk_latents: int = 21,
) -> Dict[str, Any]:
    """
    Computes analytical FLOPs and peak VRAM memory for Full, Decoupled, and Hybrid attention.
    Compares 5s (81 frames, 21 latents) and 30s (481 frames, 121 latents).
    """
    # 1. Spacetime token calculation
    raw_frames = int(round(duration_seconds * fps)) + 1
    k = max(0, round((raw_frames - 1) / vae_stride[0]))
    total_raw_frames = 4 * k + 1
    latent_t = (total_raw_frames - 1) // vae_stride[0] + 1

    latent_h = height // vae_stride[1]
    latent_w = width // vae_stride[2]

    grid_h = latent_h // patch_size[1]
    grid_w = latent_w // patch_size[2]
    spatial_tokens_per_frame = grid_h * grid_w
    total_visual_tokens = latent_t * spatial_tokens_per_frame
    total_seq_len = total_visual_tokens + text_tokens

    head_dim = dim // num_heads

    # A. Full Attention (O(N^2))
    # QK^T: 2 * N_tot^2 * d, Softmax * V: 2 * N_tot^2 * d -> 4 * N_tot^2 * d per layer
    full_attn_flops_per_layer = 4 * (total_seq_len ** 2) * dim
    full_attn_flops_total = full_attn_flops_per_layer * depth

    # Peak Attention Matrix Memory in MB (fp16: 2 bytes per element):
    # num_heads * N_tot^2 * 2 bytes
    full_attn_matrix_mb = (num_heads * (total_seq_len ** 2) * 2) / (1024 * 1024)

    # B. Decoupled Attention
    # Spatial: T frames, each attention over (S + L) tokens
    # FLOPs per frame: 4 * (S + L)^2 * d -> Total spatial: T * 4 * (S + L)^2 * d
    spatial_seq_len = spatial_tokens_per_frame + text_tokens
    spatial_flops_per_layer = latent_t * 4 * (spatial_seq_len ** 2) * dim

    # Temporal: S spatial positions, each attention over T frames (visual only)
    # If window W is set: S * 4 * T * min(T, W) * d
    t_effective_attend = min(latent_t, temporal_window_size) if temporal_window_size else latent_t
    temporal_flops_per_layer = spatial_tokens_per_frame * 4 * latent_t * t_effective_attend * dim

    decoupled_flops_per_layer = spatial_flops_per_layer + temporal_flops_per_layer
    decoupled_flops_total = decoupled_flops_per_layer * depth

    # Decoupled Peak Activation Memory:
    # Spatial matrix: num_heads * (S + L)^2 * 2 bytes
    spatial_matrix_mb = (num_heads * (spatial_seq_len ** 2) * 2) / (1024 * 1024)
    # Temporal matrix: num_heads * T^2 * 2 bytes (or chunk_latents^2 if chunked)
    temporal_matrix_mb = (num_heads * (chunk_latents ** 2) * 2) / (1024 * 1024)
    decoupled_peak_matrix_mb = max(spatial_matrix_mb, temporal_matrix_mb)

    # C. Hybrid Attention
    full_layers = depth // hybrid_interval
    decoupled_layers = depth - full_layers
    hybrid_flops_total = (decoupled_layers * decoupled_flops_per_layer) + (full_layers * full_attn_flops_per_layer)
    hybrid_peak_matrix_mb = full_attn_matrix_mb  # Peak is during full attention layers

    return {
        "duration_seconds": duration_seconds,
        "fps": fps,
        "total_raw_frames": total_raw_frames,
        "latent_frames": latent_t,
        "spatial_tokens_per_frame": spatial_tokens_per_frame,
        "total_visual_tokens": total_visual_tokens,
        "total_seq_len": total_seq_len,
        "full_attention": {
            "flops_gflops": full_attn_flops_total / 1e9,
            "flops_tflops": full_attn_flops_total / 1e12,
            "peak_attn_matrix_mb": full_attn_matrix_mb,
            "peak_attn_matrix_gb": full_attn_matrix_mb / 1024.0,
        },
        "decoupled_attention": {
            "flops_gflops": decoupled_flops_total / 1e9,
            "flops_tflops": decoupled_flops_total / 1e12,
            "peak_attn_matrix_mb": decoupled_peak_matrix_mb,
            "peak_attn_matrix_gb": decoupled_peak_matrix_mb / 1024.0,
            "flops_reduction_ratio": full_attn_flops_total / max(1.0, decoupled_flops_total),
            "memory_reduction_ratio": full_attn_matrix_mb / max(0.001, decoupled_peak_matrix_mb),
        },
        "hybrid_attention": {
            "hybrid_interval": hybrid_interval,
            "full_layers": full_layers,
            "decoupled_layers": decoupled_layers,
            "flops_gflops": hybrid_flops_total / 1e9,
            "flops_tflops": hybrid_flops_total / 1e12,
            "peak_attn_matrix_mb": hybrid_peak_matrix_mb,
            "flops_reduction_ratio": full_attn_flops_total / max(1.0, hybrid_flops_total),
        },
    }
