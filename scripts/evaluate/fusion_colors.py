"""Shared colour scheme for fusion methods / stream baselines in the evaluation plots.

Greys = baselines, blues = modality-level gates, greens = feature-/spatial-wise gates,
amber/reds = token/attention-based. Dual-fusion variants (``dual_*``) get the darker/deeper hue of
their family and are drawn with a star marker (``MARKERS``).
"""

COLORS = {
    # 1. baselines (Concat both streams = heavy reference; RGB-only = bronze; DIGIT-only = steel blue-gray)
    "concat": "#1A1A1A",
    "rgb_only": "#8C6D46",
    "digit_only": "#4A6572",
    # 2. modality-level gates
    "gated": "#6BAED6",
    "gated_concat": "#2171B5",
    "residual_gated_concat": "#08306B",
    "dual_gated_concat": "#5A189A",
    # 3. feature- & spatial-wise gates
    "se_channel_gated_concat": "#2CA25F",
    "dual_se_channel_gated_concat": "#004B23",
    "channelwise_gated_concat": "#006D2C",
    # 4. token / attention-based
    "se_gated_cross_attention": "#E6550D",
    "gated_cross_attention_2d": "#A50F15",
    "dual_gated_cross_attention": "#6B0F1A",
}

MARKERS = {k: "*" for k in COLORS if k.startswith("dual_")}


def color_for(key: str, default: str = "#999999") -> str:
    return COLORS.get(key, default)


def marker_for(key: str, default: str = "o") -> str:
    return MARKERS.get(key, default)
