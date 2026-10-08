"""Gated Cross-Attention (GCA) fusion.

Unlike the concat-preserving / weighted-sum gated_concat family (fusion/gated_concat.py),
this treats each stream's flattened [P3|P4|P5] feature vector as a short token sequence
in a shared embedding space and lets the visual stream query the tactile stream via
cross-attention, rather than gating whole streams or channels directly:

    1. Tokenize: split each stream's `dim`-length flattened vector into `num_tokens`
       equal chunks and linearly project each chunk into `embed_dim` -- the "1x1 conv /
       linear projection to a common embedding space" step. Each projected token is then
       LayerNorm'd (see below).
    2. Cross-attend: visual tokens are the Query; tactile tokens are the Key and Value
       (`nn.MultiheadAttention`), so each visual token pulls in whatever tactile context
       is relevant to it. The attended output is LayerNorm'd too.
    3. Gate: a sigmoid linear layer over [T_rgb || T_attn] produces a per-token,
       per-channel gate g in [0, 1], which blends the attended tactile context back
       into the raw (projected) visual tokens: T_fused = g * T_attn + (1 - g) * T_rgb.
       This lets the model fall back to pure visual tokens where attending to tactile
       isn't useful, instead of always trusting the cross-attention output.
    4. Pool + project: global-average-pool T_fused over the token sequence to a single
       `embed_dim` vector, then a linear layer projects it back to `dim` -- the same
       width the existing MLP heads already expect -- so this fusion is a drop-in,
       weighted-sum-style alternative (one `dim`-length vector out, like `attention`/
       `gated`) rather than a concat-preserving one.

    LayerNorm placement (vis_norm/tac_norm/attn_norm below) fixes a scale confound found by
    inspecting a trained checkpoint: splitting the flattened [P3|P4|P5] vector into
    `num_tokens` equal chunks means each token's raw magnitude is dominated by which pyramid
    scale it came from (P5-derived tokens ran ~0.53 mean magnitude vs ~0.04 for P3/P4), not by
    sample content. Fed unnormalized into the gate, this made the gate track "which scale is
    this" (corr(gate, vis_tokens magnitude) = 0.97 across tokens, near-zero between-sample
    variance) rather than "is tactile useful here". Normalizing vis_tokens/tac_tokens before
    attention and attn_out before gating removes that structural confound from both the
    attention scores and the gate's input.
"""

import torch
import torch.nn as nn

from . import register_fusion
from .base import BaseFusion


@register_fusion("gated_cross_attention")
class GatedCrossAttentionFusion(BaseFusion):
    """Cross-attend tactile tokens into visual tokens, gate, pool, project back to `dim`.

    Requires exactly 2 streams in (visual, tactile) order -- the default
    `streams=("mono_cam", "tactile")` -- since the design is asymmetric: visual is
    always the query stream, tactile always supplies keys/values.
    """

    def __init__(self, dim: int, num_streams: int, embed_dim: int = 256, num_heads: int = 4,
                 num_tokens: int = 100, dropout: float = 0.1):
        super().__init__(dim, num_streams)
        if num_streams != 2:
            raise ValueError(
                "GatedCrossAttentionFusion requires exactly 2 streams (visual query, "
                f"tactile key/value), got num_streams={num_streams}."
            )
        if dim % num_tokens != 0:
            raise ValueError(
                f"dim={dim} is not divisible by num_tokens={num_tokens}; pass a "
                "num_tokens that evenly divides the flattened per-stream feature size."
            )

        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.token_dim = dim // num_tokens

        # Common embedding space (the "1x1 conv / linear projection" step).
        self.visual_proj = nn.Linear(self.token_dim, embed_dim)
        self.tactile_proj = nn.Linear(self.token_dim, embed_dim)

        # Normalize each token to unit scale before attention/gating, so per-token magnitude
        # (which is dominated by which pyramid scale a token's chunk came from, not by sample
        # content) can't drive the attention scores or the gate -- see class/module docstring.
        self.vis_norm = nn.LayerNorm(embed_dim)
        self.tac_norm = nn.LayerNorm(embed_dim)
        self.attn_norm = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout,
                                                  batch_first=True)

        # Per-token, per-channel gate over [T_rgb || T_attn] -> (B, N, embed_dim).
        self.gate = nn.Linear(2 * embed_dim, embed_dim)

        # Compatibility projection: v_gca (B, embed_dim) -> (B, dim), matching the existing
        # MLP head's input width so the head architecture stays identical for this benchmark.
        self.output_proj = nn.Linear(embed_dim, dim)

        self.last_gate = None

    def _tokenize(self, feature: torch.Tensor) -> torch.Tensor:
        return feature.reshape(feature.shape[0], self.num_tokens, self.token_dim)

    def forward(self, features_list):
        self._validate(features_list)
        visual, tactile = features_list

        vis_tokens = self.vis_norm(self.visual_proj(self._tokenize(visual)))    # (B, N, embed_dim) -- Q
        tac_tokens = self.tac_norm(self.tactile_proj(self._tokenize(tactile)))  # (B, N, embed_dim) -- K, V

        attn_out, _ = self.cross_attn(vis_tokens, tac_tokens, tac_tokens)  # (B, N, embed_dim)
        attn_out = self.attn_norm(attn_out)  # T_attn: (B, N, embed_dim)

        gate_logits = self.gate(torch.cat([vis_tokens, attn_out], dim=-1))
        g = torch.sigmoid(gate_logits)  # (B, N, embed_dim)
        self.last_gate = g.detach()

        fused_tokens = g * attn_out + (1.0 - g) * vis_tokens  # T_fused: (B, N, embed_dim)

        v_gca = fused_tokens.mean(dim=1)  # Global average pooling over the token sequence.
        return self.output_proj(v_gca)    # (B, dim)

@register_fusion("se_gated_cross_attention")
class SEGatedCrossAttentionFusion(BaseFusion):
    def __init__(self, dim, num_streams, embed_dim=256, num_heads=4, num_tokens=100, dropout=0.1, se_reduction=4):
        super().__init__(dim, num_streams)
        if num_streams != 2:
            raise ValueError(
                "SEGatedCrossAttentionFusion requires exactly 2 streams (visual query, "
                f"tactile key/value), got num_streams={num_streams}."
            )
        if dim % num_tokens != 0:
            raise ValueError(
                f"dim={dim} is not divisible by num_tokens={num_tokens}; pass a "
                "num_tokens that evenly divides the flattened per-stream feature size."
            )

        self.embed_dim = embed_dim
        self.num_tokens = num_tokens    
        self.token_dim = dim // num_tokens

        # Projections
        self.visual_proj = nn.Linear(self.token_dim, embed_dim)
        self.tactile_proj = nn.Linear(self.token_dim, embed_dim)

        # -- TEST: Scale confound mitigation --
        self.pos_encoder = ScaleAwarePositionalEncoding(embed_dim=embed_dim, num_tokens=num_tokens)


        self.vis_norm = nn.LayerNorm(embed_dim)
        self.tac_norm = nn.LayerNorm(embed_dim)
        self.attn_norm = nn.LayerNorm(embed_dim)
        # Separate norms for the post-positional-encoding step -- sharing vis_norm/tac_norm
        # between "normalize the raw projected tokens" and "normalize tokens + pos_emb" would
        # force one learnable affine to compromise between two different jobs.
        self.vis_pos_norm = nn.LayerNorm(embed_dim)
        self.tac_pos_norm = nn.LayerNorm(embed_dim)

        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_gate = nn.Linear(2 * embed_dim, embed_dim)     # (B, N, 2*embed_dim) -> (B, N, embed_dim)

        # Squeeze-and-Excitation (SE) block for channel-wise gating
        se_in_dim = 3 * 2 * embed_dim  # Concatenated [T_rgb || T_attn || T_fused] for SE
        hidden_dim = max(16, se_in_dim // se_reduction)

        self.se_block = nn.Sequential(
            nn.LayerNorm(se_in_dim),
            nn.Linear(se_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid()
        )

        # Final projection to match the original MLP head's input width
        self.output_proj = nn.Linear(embed_dim, dim)

        self.last_gate = None
        self.last_se_weights = None

    def _tokenize(self, feature: torch.Tensor) -> torch.Tensor:
        return feature.reshape(feature.shape[0], self.num_tokens, self.token_dim)

    def forward(self, features_list):
        self._validate(features_list)
        visual, tactile = features_list

        vis_tokens = self.vis_norm(self.visual_proj(self._tokenize(visual)))    # (B, N, embed_dim) -- Q
        tac_tokens = self.tac_norm(self.tactile_proj(self._tokenize(tactile)))  # (B, N, embed_dim) -- K, V


        # generate positional and scale-aware embeddings for the tokens
        pos_emb = self.pos_encoder(batch_size=vis_tokens.shape[0], device=vis_tokens.device)  # (B, N, embed_dim)
        vis_tokens = self.vis_pos_norm(vis_tokens + pos_emb)
        tac_tokens = self.tac_pos_norm(tac_tokens + pos_emb)



        attn_out, attn_weights = self.cross_attn(vis_tokens, tac_tokens, tac_tokens)  # (B, N, embed_dim)
        attn_out = self.attn_norm(attn_out)  # T_attn: (B, N, embed_dim)

        # Squeeze-and-Excitation (SE) block for channel-wise gating
        descr_vis = torch.cat([vis_tokens.mean(dim=1), vis_tokens.abs().mean(dim=1)], dim=-1)  # (B, 2*embed_dim)
        descr_tac = torch.cat([tac_tokens.mean(dim=1), tac_tokens.abs().mean(dim=1)], dim=-1)  # (B, 2*embed_dim)
        descr_attn = torch.cat([attn_out.mean(dim=1), attn_out.abs().mean(dim=1)], dim=-1)  # (B, 2*embed_dim)

        se_descriptor = torch.cat([descr_vis, descr_tac, descr_attn], dim=-1)  # (B, 3*2*embed_dim)

        # SE block to compute channel-wise weights (B, 1, embed_dim)
        se_channel_weights = self.se_block(se_descriptor).unsqueeze(1)  # (B, 1, embed_dim)
        self.last_se_weights = se_channel_weights.detach()

        # Token-wise spatial gating logic
        self.spatial_gate_logits = self.spatial_gate(torch.cat([vis_tokens, attn_out], dim=-1))  # (B, N, embed_dim)
        g_spacial = torch.sigmoid(self.spatial_gate_logits)  # (B, N, embed_dim)

        g_hybrid = g_spacial * se_channel_weights  # (B, N, embed_dim)
        self.last_gate = g_hybrid.detach()

        # Residual connection with gated fusion
        fused_tokens = g_hybrid * attn_out + (1.0 - g_hybrid) * vis_tokens  # T_fused: (B, N, embed_dim)

        # global average pooling and output projection
        v_gca = fused_tokens.mean(dim=1)  # Global average pooling over the token sequence.
        return self.output_proj(v_gca)    # (B, dim)



# ----- TEST ------ 
# Spatial and scale embedding to possibly mitigate the attention collapse feature scale dominance 
import math

class ScaleAwarePositionalEncoding(nn.Module):
    """Generates 2D spatial + 1D scale positional embeddings for multi-scale pyramid tokens.

    Assumes features are chunked/flattened from P3, P4, P5 scales into N tokens.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 100,
        scales_spatial_shapes: list = None,
    ):
        """Args:

        embed_dim: Hidden embedding dimension of tokens.
        num_tokens: Total sequence length (e.g. 100).
        scales_spatial_shapes: List of tuples [(H3, W3), (H4, W4), (H5, W5)] or grid
        allocation per scale. Default auto-constructs for P3(20x20), P4(10x10),
        P5(5x5) equivalent balance.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # 1. Scale Level Embedding (3 levels: P3, P4, P5)
        self.scale_embed = nn.Embedding(3, embed_dim // 2)

        # 2. Linear projection to merge 2D Fourier spatial PE + 1D Scale PE into embed_dim
        spatial_dim = embed_dim // 2
        self.spatial_proj = nn.Linear(spatial_dim, embed_dim // 2)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Precompute static spatial (x, y) coordinates and scale indices per token [N]
        coords_x, coords_y, scale_ids = self._build_token_grid_mapping(
            num_tokens, scales_spatial_shapes
        )

        # Register non-trainable coordinate buffers
        self.register_buffer("coords_x", coords_x)  # (N,)
        self.register_buffer("coords_y", coords_y)  # (N,)
        self.register_buffer("scale_ids", scale_ids)  # (N,), dtype=long

    def _build_token_grid_mapping(self, num_tokens, shapes):
        """Maps each of the N tokens to a normalized (x,y) in [0,1] and a scale index [0,1,2]."""
        # Default distribution assuming 56 total channels split across P3, P4, P5 chunks
        # Adapt these splits/shapes if your token chunking ratio across P3,P4,P5 differs
        if shapes is None:
            # Example token allocation across P3, P4, P5: e.g. 50 tokens P3, 30 tokens P4, 20 tokens P5
            tokens_per_scale = [50, 30, 20]
            grid_sizes = [(7, 7), (5, 6), (4, 5)]  # Approx grid structures
        else:
            grid_sizes = shapes
            tokens_per_scale = [h * w for h, w in shapes]

        all_x, all_y, all_scales = [], [], []

        token_count = 0
        for scale_idx, (h, w) in enumerate(grid_sizes):
            for i in range(h):
                for j in range(w):
                    if token_count >= num_tokens:
                        break
                    # Normalized coordinates [0.0, 1.0]
                    norm_x = (j + 0.5) / w
                    norm_y = (i + 0.5) / h
                    all_x.append(norm_x)
                    all_y.append(norm_y)
                    all_scales.append(scale_idx)
                    token_count += 1

        # Fill remaining tokens if any rounding misalignment occurs
        while len(all_x) < num_tokens:
            all_x.append(0.5)
            all_y.append(0.5)
            all_scales.append(2)

        return (
            torch.tensor(all_x, dtype=torch.float32),
            torch.tensor(all_y, dtype=torch.float32),
            torch.tensor(all_scales, dtype=torch.long),
        )

    def _fourier_2d_enc(self, x: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
        """2D Sinusoidal / Fourier positional encoding for normalized coordinates (B, N)."""
        num_bands = dim // 4
        bands = 2.0 ** torch.linspace(
            0, math.log2(100), num_bands, device=x.device
        )  # (num_bands,)

        # x, y: (B, N) -> (B, N, num_bands)
        x_proj = x.unsqueeze(-1) * bands * math.pi
        y_proj = y.unsqueeze(-1) * bands * math.pi

        pe = torch.cat(
            [torch.sin(x_proj), torch.cos(x_proj), torch.sin(y_proj), torch.cos(y_proj)],
            dim=-1,
        )
        return pe  # (B, N, dim)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Expand coordinates to batch size: (B, N)
        cx = self.coords_x.unsqueeze(0).expand(batch_size, -1)
        cy = self.coords_y.unsqueeze(0).expand(batch_size, -1)
        scales = self.scale_ids.unsqueeze(0).expand(batch_size, -1)

        # 1. 2D Spatial Encoding: (B, N, embed_dim // 2)
        spatial_pe = self._fourier_2d_enc(cx, cy, dim=self.embed_dim // 2)
        spatial_emb = self.spatial_proj(spatial_pe)

        # 2. 1D Scale Encoding: (B, N, embed_dim // 2)
        scale_emb = self.scale_embed(scales)

        # 3. Concatenate and project: (B, N, embed_dim)
        pos_emb = torch.cat([spatial_emb, scale_emb], dim=-1)
        return self.out_proj(pos_emb)