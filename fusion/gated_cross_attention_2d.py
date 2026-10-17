"""SE-gated cross-attention over real 2D feature-map tokens ("2D" variant of
`se_gated_cross_attention`, fusion/gated_cross_attention.py).

The 1D version cuts each stream's flattened [P3|P4|P5] vector into equal chunks, so a "token" is an
arbitrary slice of one channel's map and its positional encoding is a guess. This version undoes the
flatten first and makes every grid cell of every pyramid scale one token:

    1. Unflatten:  [P3 | P4 | P5] -> (B, C3, H3, W3), (B, C4, H4, W4), (B, C5, H5, W5). The grids and
                   channel counts come from `scale_dims` / `scale_channels` (PoseEstimationNet passes
                   them), so the module follows `pre_head_channel_divisor` automatically
                   (32/64/128 channels at divisor 1, 8/16/32 at divisor 4; grids stay 20/10/5).
    2. Tokenize:   one token per grid cell -> (B, H*W, C); a separate Linear(C -> embed_dim) per scale
                   and stream, each followed by a LayerNorm (P5-derived values are ~13x larger than
                   P3/P4, so un-normalized scale would dominate attention and the gate). All scales
                   are concatenated (400 + 100 + 25 = 525 tokens at 20/10/5 grids).
    3. Position:   fixed 2D Fourier features of the true normalized (x, y) of each cell -> learned
                   Linear, plus a learned P3/P4/P5 embedding. RGB and tactile each have their OWN
                   table: the camera and the DIGIT gel are not registered, so they must not be
                   pushed towards "same coordinates = same place". `tactile_pos="none"` drops the
                   tactile table (content-only keys/values) for an ablation.
    4. Cross-attend: RGB tokens are the query, tactile tokens key/value (as in the 1D version).
    5. Gate:       g = sigmoid(token_logits + se_logits). token_logits come from [rgb || attended]
                   per token and channel; se_logits come from a per-scale squeeze-and-excitation
                   descriptor (mean and mean|x| of rgb / tactile / attended tokens) and give one
                   value per (scale, channel), shared by all tokens of that scale. Adding in logit
                   space (instead of multiplying two sigmoids) keeps gradients healthy, and the bias
                   starts g at `gate_init_prob` (default 0.1), so the module begins close to the
                   RGB-only path. fused = rgb + g * (attended - rgb).
    6. Output (`output_mode`):
         "map":  split the fused tokens back into the three scales, Linear(embed_dim -> C_scale)
                 per scale, reshape to (B, C, H, W) and flatten. Same layout/width (`dim`) as the
                 input, nothing is pooled away, so the heads see the spatial layout.
         "pool": mean over all tokens -> Linear(embed_dim -> dim), exactly like the 1D module's
                 256-d bottleneck. Exists so the tokenization and the bottleneck can be ablated
                 separately.

Registered names (thin subclasses that only change defaults, because build_fusion forwards no other
kwargs from the config):
    se_gated_cross_attention_2d             output_mode="map",  tactile_pos="learned"
    se_gated_cross_attention_2d_pool        output_mode="pool", tactile_pos="learned"
    se_gated_cross_attention_2d_notacpos_onlyRGBposenc       output_mode="map",  tactile_pos="none"
    gated_cross_attention_2d                use_se=False (token gate only, no SE branch -- SE ablation)

Returns a single (B, dim) tensor, a weighted-sum-style output (not concat-preserving), so the model
needs no change. Diagnostics: `last_gate` (B, N, embed_dim), `last_se_weights` (B, n_scales, embed_dim).
"""

import contextlib
import math

import torch
import torch.nn as nn

from . import register_fusion
from .base import BaseFusion

_SCALE_NAMES = ("p3", "p4", "p5")


class _TokenPositionEncoding(nn.Module):
    """Fixed 2D Fourier features of each token's true normalized (x, y) -> learned Linear, plus a
    learned scale-level embedding. One instance per modality."""

    def __init__(self, grids, embed_dim, num_bands=6):
        super().__init__()
        xs, ys, levels = [], [], []
        for level, (h, w) in enumerate(grids):
            for i in range(h):
                for j in range(w):
                    ys.append((i + 0.5) / h)
                    xs.append((j + 0.5) / w)
                    levels.append(level)
        coords = torch.stack([torch.tensor(xs), torch.tensor(ys)], dim=-1)           # (N, 2)
        freqs = math.pi * (2.0 ** torch.arange(num_bands, dtype=torch.float32))      # (bands,)
        angles = coords.unsqueeze(-1) * freqs                                        # (N, 2, bands)
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(1)         # (N, 4*bands)
        self.register_buffer("fourier", fourier, persistent=False)
        self.proj = nn.Linear(fourier.shape[1], embed_dim)
        self.scale_embed = nn.Embedding(len(grids), embed_dim)
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.long), persistent=False)

    def forward(self):
        return self.proj(self.fourier) + self.scale_embed(self.levels)               # (N, embed_dim)


class _StreamTokenizer(nn.Module):
    """Per-scale Linear(C -> embed_dim) + LayerNorm on (B, H*W, C) tokens, concatenated over scales."""

    def __init__(self, scale_channels, embed_dim):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(c, embed_dim) for c in scale_channels])
        self.norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in scale_channels])

    def forward(self, maps):
        return torch.cat([n(p(m)) for m, p, n in zip(maps, self.proj, self.norm)], dim=1)


class SEGatedCrossAttention2DFusion(BaseFusion):
    # Registered subclasses only change these defaults (build_fusion inspects __init__'s
    # signature to decide which kwargs to forward, so they must not redefine __init__).
    default_output_mode = "map"
    default_tactile_pos = "learned"
    default_use_se = True

    def __init__(self, dim, num_streams, scale_dims=None, scale_channels=None, embed_dim=256,
                 num_heads=4, dropout=0.1, se_reduction=4, gate_init_prob=0.1,
                 output_mode=None, tactile_pos=None, use_se=None):
        super().__init__(dim, num_streams)
        output_mode = self.default_output_mode if output_mode is None else output_mode
        tactile_pos = self.default_tactile_pos if tactile_pos is None else tactile_pos
        use_se = self.default_use_se if use_se is None else use_se
        if num_streams != 2:
            raise ValueError(
                "SEGatedCrossAttention2DFusion requires exactly 2 streams (visual query, "
                f"tactile key/value), got num_streams={num_streams}."
            )
        if scale_dims is None or scale_channels is None:
            raise ValueError(
                "SEGatedCrossAttention2DFusion needs scale_dims and scale_channels "
                "(PoseEstimationNet.stream_scale_dims / stream_scale_channels) to unflatten "
                "the per-stream vector into feature maps."
            )
        if output_mode not in ("map", "pool"):
            raise ValueError(f"output_mode must be 'map' or 'pool', got {output_mode!r}")
        if tactile_pos not in ("learned", "none"):
            raise ValueError(f"tactile_pos must be 'learned' or 'none', got {tactile_pos!r}")
        if not 0.0 < gate_init_prob < 1.0:
            raise ValueError(f"gate_init_prob must be strictly between 0 and 1, got {gate_init_prob}")

        self.scale_dims = [int(scale_dims[k]) for k in _SCALE_NAMES]
        self.scale_channels = [int(scale_channels[k]) for k in _SCALE_NAMES]
        if sum(self.scale_dims) != dim:
            raise ValueError(f"sum(scale_dims)={sum(self.scale_dims)} != dim={dim}")
        self.grids = []
        for name, d, c in zip(_SCALE_NAMES, self.scale_dims, self.scale_channels):
            cells = d // c
            side = math.isqrt(cells)
            if d != c * cells or side * side != cells:
                raise ValueError(
                    f"{name}: flattened size {d} with {c} channels is not a square grid "
                    f"({cells} cells per channel)."
                )
            self.grids.append((side, side))
        self.tokens_per_scale = [h * w for h, w in self.grids]
        self.num_scales = len(self.grids)
        self.embed_dim = embed_dim
        self.output_mode = output_mode
        self.tactile_pos = tactile_pos
        self.use_se = use_se

        # Tokenization (own projections per stream; the two sensors share no weights).
        self.visual_tokenizer = _StreamTokenizer(self.scale_channels, embed_dim)
        self.tactile_tokenizer = _StreamTokenizer(self.scale_channels, embed_dim)

        # Position information: a separate table per modality (sensors are not registered).
        self.visual_pos = _TokenPositionEncoding(self.grids, embed_dim)
        self.tactile_pos_enc = _TokenPositionEncoding(self.grids, embed_dim) if tactile_pos == "learned" else None
        # Separate norms for "tokens only" vs "tokens + position" (same reasoning as the 1D module).
        self.vis_pos_norm = nn.LayerNorm(embed_dim)
        self.tac_pos_norm = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout,
                                                batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Gate = sigmoid(token logits + SE logits).
        self.spatial_gate = nn.Linear(2 * embed_dim, embed_dim)
        nn.init.constant_(self.spatial_gate.bias, math.log(gate_init_prob / (1.0 - gate_init_prob)))

        if use_se:
            se_in_dim = 3 * 2 * embed_dim                                 # [rgb | tactile | attended] x [mean, mean|x|]
            se_hidden = max(16, se_in_dim // se_reduction)
            self.se_scale_embed = nn.Embedding(self.num_scales, se_in_dim)    # tells the shared MLP which scale it sees
            self.se_block = nn.Sequential(
                nn.LayerNorm(se_in_dim),
                nn.Linear(se_in_dim, se_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(se_hidden, embed_dim),
            )
            # Start with a neutral SE contribution so the initial gate is exactly gate_init_prob.
            nn.init.zeros_(self.se_block[-1].weight)
            nn.init.zeros_(self.se_block[-1].bias)

        self.register_buffer(
            "token_levels",
            torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(self.tokens_per_scale)]),
            persistent=False,
        )

        if output_mode == "map":
            self.out_proj = nn.ModuleList([nn.Linear(embed_dim, c) for c in self.scale_channels])
        else:
            self.out_proj = nn.Linear(embed_dim, dim)

        self.last_gate = None
        self.last_se_weights = None

    # -- helpers ----------------------------------------------------------------------------
    def _unflatten(self, feature):
        """(B, dim) -> list of (B, H*W, C) per scale. Exact inverse of the channel-major flatten."""
        maps = []
        offset = 0
        for d, c, (h, w) in zip(self.scale_dims, self.scale_channels, self.grids):
            block = feature[:, offset:offset + d].reshape(feature.shape[0], c, h * w)
            maps.append(block.transpose(1, 2))
            offset += d
        return maps

    def _flatten(self, tokens_per_scale):
        """list of (B, H*W, C) -> (B, dim). Exact inverse of `_unflatten`."""
        return torch.cat([t.transpose(1, 2).reshape(t.shape[0], -1) for t in tokens_per_scale], dim=1)

    def _scale_stats(self, tokens):
        """(B, N, E) -> (B, n_scales, 2E): per-scale mean and mean|x| over that scale's tokens."""
        stats = []
        for t in torch.split(tokens, self.tokens_per_scale, dim=1):
            stats.append(torch.cat([t.mean(dim=1), t.abs().mean(dim=1)], dim=-1))
        return torch.stack(stats, dim=1)

    def _attention_context(self):
        # Flash / mem-efficient SDPA backward is non-deterministic; at N=525 the math backend is
        # cheap, so keep training.deterministic honest when it is enabled.
        if torch.are_deterministic_algorithms_enabled():
            from torch.nn.attention import SDPBackend, sdpa_kernel
            return sdpa_kernel(SDPBackend.MATH)
        return contextlib.nullcontext()

    # -- forward ----------------------------------------------------------------------------
    def forward(self, features_list):
        self._validate(features_list)
        visual, tactile = features_list

        vis = self.visual_tokenizer(self._unflatten(visual))      # (B, N, E)  -- query
        tac = self.tactile_tokenizer(self._unflatten(tactile))    # (B, N, E)  -- key / value

        vis = self.vis_pos_norm(vis + self.visual_pos().unsqueeze(0))
        if self.tactile_pos_enc is not None:
            tac = self.tac_pos_norm(tac + self.tactile_pos_enc().unsqueeze(0))
        else:
            tac = self.tac_pos_norm(tac)

        with self._attention_context():
            attn, _ = self.cross_attn(vis, tac, tac, need_weights=False)
        attn = self.attn_norm(attn)                                # (B, N, E)

        token_logits = self.spatial_gate(torch.cat([vis, attn], dim=-1))   # (B, N, E)
        if self.use_se:
            # SE logits: one vector per (scale, channel), broadcast to that scale's tokens.
            descriptor = torch.cat(
                [self._scale_stats(vis), self._scale_stats(tac), self._scale_stats(attn)], dim=-1
            )                                                      # (B, S, 6E)
            descriptor = descriptor + self.se_scale_embed.weight.unsqueeze(0)
            se_logits = self.se_block(descriptor)                  # (B, S, E)
            self.last_se_weights = torch.sigmoid(se_logits).detach()
            token_logits = token_logits + se_logits[:, self.token_levels, :]
        gate = torch.sigmoid(token_logits)
        self.last_gate = gate.detach()

        fused = torch.lerp(vis, attn, gate)                        # vis + gate * (attn - vis)

        if self.output_mode == "pool":
            return self.out_proj(fused.mean(dim=1))                # (B, dim)
        per_scale = torch.split(fused, self.tokens_per_scale, dim=1)
        return self._flatten([proj(t) for proj, t in zip(self.out_proj, per_scale)])   # (B, dim)


@register_fusion("se_gated_cross_attention_2d")
class SEGatedCrossAttention2DMapFusion(SEGatedCrossAttention2DFusion):
    """output_mode="map", tactile_pos="learned" (spatially preserving output)."""


@register_fusion("se_gated_cross_attention_2d_pool")
class SEGatedCrossAttention2DPoolFusion(SEGatedCrossAttention2DFusion):
    """output_mode="pool": 2D tokens, but the old mean-pool + Linear(embed_dim -> dim) bottleneck."""
    default_output_mode = "pool"


@register_fusion("se_gated_cross_attention_2d_notacpos_onlyRGBposenc")
class SEGatedCrossAttention2DNoTactilePosFusion(SEGatedCrossAttention2DFusion):
    """tactile_pos="none": content-only tactile keys/values (positional-encoding ablation)."""
    default_tactile_pos = "none"


@register_fusion("gated_cross_attention_2d")
class GatedCrossAttention2DFusion(SEGatedCrossAttention2DFusion):
    """use_se=False: same 2D tokens, positions, cross-attention and output as
    `se_gated_cross_attention_2d`, but the gate is the token-wise gate only (no SE branch)."""
    default_use_se = False
