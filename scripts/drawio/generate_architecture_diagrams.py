#!/usr/bin/env python3
"""
Generate draw.io block diagrams of the 11 fusion candidates.

Candidates 1-7 are the original final_ablation_spec.yml grid; 8-9 (SE-channel gated concat)
and 10-11 (gated / SE-gated cross-attention) were added afterwards as new fusion/gated_concat.py
and fusion/gated_cross_attention.py strategies, appended rather than renumbering 1-7 (see
scripts/evaluate/final_ablation_candidate_analysis.py's "Candidates 8/9" comment for that
convention):

    1 concat | 2 gated | 3 gated_concat | 4 residual_gated_concat |
    5 channelwise_gated_concat_local | 6 channelwise_residual_gated_concat_local |
    7 gated_concat + dual_fusion | 8 se_channel_gated_concat | 9 se_channel_gated_concat_renorm |
    10 gated_cross_attention | 11 se_gated_cross_attention

Each diagram starts at the per-stream feature vectors (flattened P3|P4|P5 = 22,400-d per
stream) and shows fusion -> rotation / translation heads -> outputs. Dimensions were read off
a constructed PoseEstimationNet (yolo26n.pt, 640x640) and the head widths come from
configs/final_ablation_spec.yml (candidates 1-7) / configs/generation_specs/se_channel_gated_concat_vs_gated_concat_spec.yml
(candidates 8-9) / configs/generation_specs/gated_cross_attention_seeds36_40_spec.yml
and se_gated_cross_attention_seeds36_40_spec.yml (candidates 10-11); parameter counts are
recomputed and asserted against the measured totals so the head/fusion sizes below cannot
silently drift from the code.

Output (default docs/architecture_diagrams/):
    01_concat.drawio ... 11_se_gated_cross_attention.drawio   one page each
    all_candidates.drawio                                     all 11 pages in one file

Run from the repository root:
    python scripts/drawio/generate_architecture_diagrams.py
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FEAT = 22400                                     # per-stream flattened P3|P4|P5 vector
BACKBONES_AND_NECK_PARAMS = 4_525_248 + 430_752  # both YOLO26n backbones + shared 1x1/stride-2 neck
GATE_MLP_PARAMS = 44800 * 32 + 32 + 32 * 2 + 2   # Linear 44800->32, Linear 32->2
LOCAL_GATE_PARAMS = (2 * 16 + 16) + (16 * 2 + 2)  # Conv1d 2->16, Conv1d 16->2

# se_channel_gated_concat[_renorm] (fusion/gated_concat.py::_BaseSEChannelGatedConcat): one gate
# per (scale, channel) and stream -- NUM_CHANNELS = 224 (P3:32 + P4:64 + P5:128) -- computed from
# a squeeze descriptor (mean + mean|x| per channel, both streams: DESC_DIM = 2 streams * 2 stats *
# 224 channels = 896) through LayerNorm(896) -> Linear(896->64) -> ReLU -> Dropout -> Linear(64->448).
SE_NUM_CHANNELS = 224
SE_DESC_DIM = 2 * 2 * SE_NUM_CHANNELS   # 896
SE_HIDDEN_DIM = 64
SE_GATE_OUT = 2 * SE_NUM_CHANNELS       # 448
SE_CHANNEL_GATE_PARAMS = (2 * SE_DESC_DIM) + (SE_DESC_DIM * SE_HIDDEN_DIM + SE_HIDDEN_DIM) + \
    (SE_HIDDEN_DIM * SE_GATE_OUT + SE_GATE_OUT)  # LayerNorm affine + 2 Linears = 88,320

# gated_cross_attention / se_gated_cross_attention (fusion/gated_cross_attention.py): each
# stream's flattened FEAT-dim vector is split into NUM_TOKENS tokens of TOKEN_DIM each and
# projected into a shared EMBED_DIM space; visual tokens attend (as Q) to tactile tokens
# (K, V) via nn.MultiheadAttention, then a gate blends the attended output back into the
# raw visual tokens before pooling and projecting back to FEAT (weighted-sum-style output,
# like `gated` -- not concat-preserving, so fused_dim = FEAT, not 2*FEAT).
EMBED_DIM = 256
NUM_HEADS = 4
NUM_TOKENS = 100
TOKEN_DIM = FEAT // NUM_TOKENS  # 224


def linear_params(in_dim, out_dim):
    return in_dim * out_dim + out_dim


def layernorm_params(dim):
    return 2 * dim


def mha_params(embed_dim):
    # nn.MultiheadAttention(embed_dim, ...): in_proj (packed Q/K/V) + out_proj.
    return (3 * embed_dim * embed_dim + 3 * embed_dim) + (embed_dim * embed_dim + embed_dim)


# Shared tokenize+project+attend backbone common to both cross-attention variants.
_GCA_SHARED_PARAMS = (
    2 * linear_params(TOKEN_DIM, EMBED_DIM)      # visual_proj, tactile_proj
    + 3 * layernorm_params(EMBED_DIM)            # vis_norm, tac_norm, attn_norm
    + mha_params(EMBED_DIM)                      # cross_attn
)
GCA_GATE_PARAMS = linear_params(2 * EMBED_DIM, EMBED_DIM)       # gate: [vis||attn] -> g
GCA_OUTPUT_PROJ_PARAMS = linear_params(EMBED_DIM, FEAT)          # output_proj: embed_dim -> FEAT
GCA_PARAMS = _GCA_SHARED_PARAMS + GCA_GATE_PARAMS + GCA_OUTPUT_PROJ_PARAMS  # 6,268,032

# se_gated_cross_attention additionally has a ScaleAwarePositionalEncoding, two extra
# post-pos-encoding LayerNorms, a Squeeze-and-Excitation block over [vis||tac||attn]
# descriptors, and renames `gate` to `spatial_gate` (same shape/params).
POS_ENC_PARAMS = (
    3 * (EMBED_DIM // 2)                                    # scale_embed: Embedding(3, embed_dim//2)
    + linear_params(EMBED_DIM // 2, EMBED_DIM // 2)          # spatial_proj
    + linear_params(EMBED_DIM, EMBED_DIM) + layernorm_params(EMBED_DIM)  # out_proj Sequential
)  # 83,200
SE_REDUCTION = 4
SE_GCA_DESC_DIM = 3 * 2 * EMBED_DIM  # [vis||tac||attn], each (mean, mean|x|) -> 1,536
SE_GCA_HIDDEN_DIM = max(16, SE_GCA_DESC_DIM // SE_REDUCTION)  # 384
SE_BLOCK_PARAMS = (
    layernorm_params(SE_GCA_DESC_DIM)
    + linear_params(SE_GCA_DESC_DIM, SE_GCA_HIDDEN_DIM)
    + linear_params(SE_GCA_HIDDEN_DIM, EMBED_DIM)
)  # 691,840
SE_GCA_PARAMS = (
    _GCA_SHARED_PARAMS + POS_ENC_PARAMS + 2 * layernorm_params(EMBED_DIM)  # vis_pos_norm, tac_pos_norm
    + GCA_GATE_PARAMS + SE_BLOCK_PARAMS + GCA_OUTPUT_PROJ_PARAMS
)  # 7,044,096


def mlp_params(in_dim, hidden, out_dim):
    total, prev = 0, in_dim
    for h in hidden + [out_dim]:
        total += prev * h + h
        prev = h
    return total


def fmt(n):
    return f"{n:,}"


# name shown in the title, fusion kind, dual flag, head widths (from configs/final_ablation_spec.yml),
# per-head input dim, fusion params, measured total params
CANDIDATES = [
    dict(file="01_concat", title="1. Concat", kind="concat", dual=False,
         rot=[256, 1024], trans=[1024, 1024, 16], fused=2 * FEAT, fusion_params=0, total=63_634_582),
    dict(file="02_gated", title="2. Gated Fusion (Softmax)", kind="gated", dual=False,
         rot=[256, 1024], trans=[1024, 1024, 16], fused=FEAT, fusion_params=GATE_MLP_PARAMS, total=36_396_280),
    dict(file="03_gated_concat", title="3. Gated Concat Fusion", kind="gated_concat", dual=False,
         rot=[256, 1024], trans=[1024, 1024, 16], fused=2 * FEAT, fusion_params=GATE_MLP_PARAMS, total=65_068_280),
    dict(file="04_residual_gated_concat", title="4. Residual Gated Concat Fusion", kind="residual_gated_concat",
         dual=False, rot=[256, 1024], trans=[1024, 1024, 16], fused=2 * FEAT, fusion_params=GATE_MLP_PARAMS,
         total=65_068_280),
    dict(file="05_channelwise_gated_concat_local", title="5. Channelwise Gated Concat (Local)", kind="cw_local",
         dual=False, rot=[2048, 1024, 512], trans=[1024, 256, 64], fused=2 * FEAT, fusion_params=LOCAL_GATE_PARAMS,
         total=145_488_760),
    dict(file="06_channelwise_residual_gated_concat_local", title="6. Channelwise Residual Gated Concat (Local)",
         kind="cw_res_local", dual=False, rot=[2048, 1024, 512], trans=[1024, 256, 64], fused=2 * FEAT,
         fusion_params=LOCAL_GATE_PARAMS, total=145_488_760),
    dict(file="07_dual_gated_concat", title="7. Dual Fusion (Gated Concat)", kind="dual", dual=True,
         rot=[256, 1024], trans=[1024, 1024, 16], fused=2 * FEAT, fusion_params=2 * GATE_MLP_PARAMS,
         total=66_501_978),
    dict(file="08_se_channel_gated_concat", title="8. SE-Channel Gated Concat", kind="se_channel", dual=False,
         rot=[256, 1024], trans=[1024, 1024, 16], fused=2 * FEAT, fusion_params=SE_CHANNEL_GATE_PARAMS,
         total=63_722_902),
    dict(file="09_se_channel_gated_concat_renorm", title="9. SE-Channel Gated Concat (Renorm)",
         kind="se_channel_renorm", dual=False, rot=[256, 1024], trans=[1024, 1024, 16], fused=2 * FEAT,
         fusion_params=SE_CHANNEL_GATE_PARAMS, total=63_722_902),
    dict(file="10_gated_cross_attention", title="10. Gated Cross-Attention", kind="gca", dual=False,
         rot=[256, 1024], trans=[1024, 1024, 16], fused=FEAT, fusion_params=GCA_PARAMS, total=41_230_614),
    dict(file="11_se_gated_cross_attention", title="11. SE-Gated Cross-Attention", kind="se_gca", dual=False,
         rot=[256, 1024], trans=[1024, 1024, 16], fused=FEAT, fusion_params=SE_GCA_PARAMS, total=42_006_678),
]

# ---------------------------------------------------------------------------
# Styling (matches the reference diagrams: light fills, rounded boxes, plain arrows)
# ---------------------------------------------------------------------------
STYLE = {
    "rgb":    ("#e3f2fd", "#000000", 1.5),
    "tac":    ("#f3e5f5", "#000000", 1.5),
    "stack":  ("#fff8e1", "#ff9800", 1.5),
    "gate":   ("#ffe0b2", "#ff9800", 2),
    "renorm": ("#fce4ec", "#c2185b", 1.5),
    "attn":   ("#ede7f6", "#5e35b1", 2),
    "pos":    ("#e0f2f1", "#00695c", 1.5),
    "gated":  ("#e8f5e9", "#1aa53b", 1.5),
    "fused":  ("#e8f5e9", "#1aa53b", 2),
    "rot":    ("#e3f2fd", "#0066cc", 2),
    "trans":  ("#f3e5f5", "#0066cc", 2),
    "out":    ("#e8f5e9", "#1aa53b", 2),
}
BOX = "rounded=1;whiteSpace=wrap;html=1;arcSize=16;fontSize=14;fillColor={f};strokeColor={s};strokeWidth={w};"
EDGE = ("rounded=0;html=1;endArrow=open;endSize=10;strokeWidth=1.5;strokeColor=#000000;"
        "labelBackgroundColor=#ffffff;fontSize=11;{c}")

PAGE_W = 900
CL, CM, CR = 240, 450, 660        # left (RGB / rotation), centre, right (tactile / translation) columns
Y_TITLE, Y_FEAT, H_FEAT = 14, 74, 80


class Page:
    def __init__(self, name):
        self.name, self.cells, self.geo, self._n = name, [], {}, 1

    def _id(self, p):
        self._n += 1
        return f"{p}{self._n}"

    def box(self, text, cx, y, w, h, kind):
        f, s, sw = STYLE[kind]
        cid = self._id("n")
        self.cells.append(dict(id=cid, value=text, style=BOX.format(f=f, s=s, w=sw), x=cx - w / 2, y=y, w=w, h=h,
                               edge=False))
        self.geo[cid] = (cx - w / 2, y, w, h)
        return cid

    def title(self, text):
        cid = self._id("n")
        self.cells.append(dict(id=cid, value=f"<b>{text}</b>", x=40, y=Y_TITLE, w=820, h=36, edge=False,
                               style="text;html=1;align=left;verticalAlign=middle;fontSize=22;"))
        self.geo[cid] = (40, Y_TITLE, 820, 36)

    def anchor(self, nid, side, frac=0.5):
        x, y, w, h = self.geo[nid]
        return {"t": (x + frac * w, y), "b": (x + frac * w, y + h),
                "l": (x, y + frac * h), "r": (x + w, y + frac * h)}[side]

    def edge(self, a, b, sa="b", ta="t", fa=0.5, fb=0.5, via=None, midy=None, label=None):
        s, e = self.anchor(a, sa, fa), self.anchor(b, ta, fb)
        if via is None:
            via = []
            if sa == "b" and ta == "t" and abs(s[0] - e[0]) > 0.5:
                my = midy if midy is not None else (s[1] + e[1]) / 2
                via = [(s[0], my), (e[0], my)]
            elif sa == "b" and ta in ("l", "r") and abs(s[0] - e[0]) > 0.5:
                via = [(s[0], e[1])]
        ex, ey = {"t": (fa, 0), "b": (fa, 1), "l": (0, fa), "r": (1, fa)}[sa]
        nx, ny = {"t": (fb, 0), "b": (fb, 1), "l": (0, fb), "r": (1, fb)}[ta]
        c = f"exitX={ex};exitY={ey};exitPerimeter=0;entryX={nx};entryY={ny};entryPerimeter=0;"
        cid = self._id("e")
        self.cells.append(dict(id=cid, value=label or "", style=EDGE.format(c=c), source=a, target=b,
                               via=[(float(x), float(y)) for x, y in via], edge=True))

    def to_xml(self, height):
        d = ET.Element("diagram", id=f"page-{self.name}", name=self.name)
        m = ET.SubElement(d, "mxGraphModel", dx="1000", dy="800", grid="1", gridSize="10", guides="1", tooltips="1",
                          connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth=str(PAGE_W),
                          pageHeight=str(int(height)), math="0", shadow="0")
        root = ET.SubElement(m, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")
        for c in self.cells:
            if c["edge"]:
                cell = ET.SubElement(root, "mxCell", id=c["id"], value=c["value"], style=c["style"], edge="1",
                                     parent="1", source=c["source"], target=c["target"])
                g = ET.SubElement(cell, "mxGeometry", relative="1")
                g.set("as", "geometry")
                if c["via"]:
                    arr = ET.SubElement(g, "Array")
                    arr.set("as", "points")
                    for px, py in c["via"]:
                        ET.SubElement(arr, "mxPoint", x=f"{px:g}", y=f"{py:g}")
            else:
                cell = ET.SubElement(root, "mxCell", id=c["id"], value=c["value"], style=c["style"], vertex="1",
                                     parent="1")
                g = ET.SubElement(cell, "mxGeometry", x=f"{c['x']:g}", y=f"{c['y']:g}", width=f"{c['w']:g}",
                                  height=f"{c['h']:g}")
                g.set("as", "geometry")
        return d


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def features(p):
    rgb = p.box(f"RGB Features<br>{fmt(FEAT)}-dim", CL, Y_FEAT, 340, H_FEAT, "rgb")
    tac = p.box(f"Tactile Features<br>{fmt(FEAT)}-dim", CR, Y_FEAT, 340, H_FEAT, "tac")
    return rgb, tac


def side_feed(p, rgb, tac, rgb_target, tac_target):
    """Route the raw features around the gate on the outer edges into a later box's side."""
    p.edge(rgb, rgb_target, fa=0.05, ta="l")
    p.edge(tac, tac_target, fa=0.95, ta="r")


def heads(p, cand, src_rot, src_trans, y):
    d = fmt(cand["fused"])
    rot = " → ".join([d] + [str(h) for h in cand["rot"]] + ["4"])
    trans = " → ".join([d] + [str(h) for h in cand["trans"]] + ["2"])
    hr = p.box(f"Rotation Head (MLP)<br>{rot}<br><font style='font-size:11px'>GELU · Dropout 0.1</font>", CL, y, 400,
               96, "rot")
    ht = p.box(f"Translation Head (MLP)<br>{trans}<br><font style='font-size:11px'>GELU · Dropout 0.1</font>", CR, y,
               400, 96, "trans")
    p.edge(src_rot, hr, midy=y - 30)
    p.edge(src_trans, ht, midy=y - 30)
    yo = y + 96 + 56
    q = p.box("Quaternion <i><b>q</b></i><br><font style='font-size:11px'>L2-normalised, ℝ⁴</font>", CL, yo, 300, 64,
              "out")
    t = p.box("Translation Vector [<i>y</i>, <i>z</i>]<br><font style='font-size:11px'>x fixed to 0</font>", CR, yo,
              300, 64, "out")
    p.edge(hr, q)
    p.edge(ht, t)
    return yo + 64 + 30


# ---------------------------------------------------------------------------
# Fusion variants -> (source for rotation head, source for translation head, y where heads start)
# ---------------------------------------------------------------------------
def fusion_concat(p, rgb, tac):
    f = p.box(f"Concatenate Streams<br>[RGB | Tactile]<br>{fmt(2 * FEAT)}-dim", CM, 210, 420, 90, "fused")
    p.edge(rgb, f)
    p.edge(tac, f)
    return f, f, 370


def fusion_gated(p, rgb, tac):
    g = p.box("Gate (Linear → ReLU → Dropout → Linear → Softmax)<br>[RGB | Tactile] 44,800 → 32 → 2 weights (w₁, w₂)"
              "<br><font style='font-size:11px'>softmax(logits / T), T = 50</font>", CM, 205, 600, 100, "gate")
    p.edge(rgb, g)
    p.edge(tac, g)
    f = p.box(f"Fused Features<br>w₁·RGB + w₂·Tactile<br>{fmt(FEAT)}-dim", CM, 385, 420, 90, "fused")
    p.edge(g, f)
    side_feed(p, rgb, tac, f, f)
    return f, f, 545


def fusion_scalar_concat(p, rgb, tac, residual):
    scale_r, scale_t = ("(1 + g_rgb)", "(1 + g_tac)") if residual else ("g_rgb", "g_tac")
    g = p.box("Gate MLP (one gate per stream)<br>[RGB | Tactile] 44,800 → 32 → 2<br>Linear → ReLU → Dropout → Linear "
              "→ Sigmoid<br>[g_rgb, g_tac] ∈ (0,1)²", CM, 205, 600, 120, "gate")
    p.edge(rgb, g)
    p.edge(tac, g)
    f = p.box(f"[{scale_r}·RGB | {scale_t}·Tactile]<br>{fmt(2 * FEAT)}-dim", CM, 405, 460, 90, "fused")
    p.edge(g, f)
    side_feed(p, rgb, tac, f, f)
    return f, f, 565


def fusion_channelwise_local(p, rgb, tac, residual):
    s = p.box(f"Stack RGB + Tactile<br>(B, 2, {fmt(FEAT)})<br>with batch size B", CM, 205, 420, 90, "stack")
    p.edge(rgb, s)
    p.edge(tac, s)
    g = p.box("Gate (applied at every position i, shared weights)<br>[RGB_i, Tactile_i] → Conv1d 1×1 (2→16) → ReLU<br>"
              "→ Conv1d 1×1 (16→2) → Sigmoid<br>[g_rgb_i, g_tac_i] ∈ (0,1)²", CM, 355, 620, 120, "gate")
    p.edge(s, g)
    scale = "(1 + g_{})" if residual else "g_{}"
    gr = p.box(f"Gated RGB<br>{scale.format('rgb')} ⊙ RGB<br>(B, {fmt(FEAT)})<br>"
               "<font style='font-size:11px'>scales each position</font>", CL, 555, 240, 110, "gated")
    gt = p.box(f"Gated Tactile<br>{scale.format('tac')} ⊙ Tactile<br>(B, {fmt(FEAT)})<br>"
               "<font style='font-size:11px'>scales each position</font>", CR, 555, 240, 110, "gated")
    p.edge(g, gr)
    p.edge(g, gt)
    p.edge(rgb, gr, fa=0.05, ta="l")
    p.edge(tac, gt, fa=0.95, ta="r")
    c = p.box(f"Concatenate Streams<br>[{scale.format('rgb')}⊙RGB | {scale.format('tac')}⊙Tactile]<br>"
              f"(B, {fmt(2 * FEAT)}-dim)", CM, 725, 420, 90, "fused")
    p.edge(gr, c)
    p.edge(gt, c)
    return c, c, 885


def fusion_dual(p, rgb, tac):
    txt = ("Fusion {n}<br>Gate MLP: 44,800 → 32 → 2 (Sigmoid)<br>[g_rgb·RGB | g_tac·Tactile]<br>"
           f"Output: {fmt(2 * FEAT)}-dim")
    fr = p.box(txt.format(n="Rotation"), CL, 235, 400, 130, "rot")
    ft = p.box(txt.format(n="Translation"), CR, 235, 400, 130, "trans")
    # both streams feed both (independent) fusion modules
    p.edge(rgb, fr, fa=0.32, fb=0.35)
    p.edge(tac, ft, fa=0.68, fb=0.65)
    p.edge(rgb, ft, fa=0.765, fb=0.85, via=[(330, 196), (800, 196)])
    p.edge(tac, fr, fa=0.2, fb=0.65, via=[(558, 214), (300, 214)])
    return fr, ft, 425


def fusion_se_channel(p, rgb, tac, renorm):
    sq = p.box(f"Squeeze: per-channel pooling<br>{SE_NUM_CHANNELS} channels/stream (P3:32, P4:64, P5:128)<br>"
               "mean + mean|x| over each channel's grid cells<br>"
               f"→ {fmt(SE_DESC_DIM)}-dim descriptor (both streams)", CM, 205, 640, 110, "stack")
    p.edge(rgb, sq)
    p.edge(tac, sq)
    ex = p.box(f"Excite (Gate MLP)<br>LayerNorm({SE_DESC_DIM}) → Linear {SE_DESC_DIM}→{SE_HIDDEN_DIM} → ReLU → "
               f"Dropout 0.3<br>→ Linear {SE_HIDDEN_DIM}→{SE_GATE_OUT} → Sigmoid<br>"
               f"→ {SE_NUM_CHANNELS} gates/stream, [g_rgb, g_tac] ∈ (0,1)^(2×{SE_NUM_CHANNELS})", CM, 335, 640, 130,
               "gate")
    p.edge(sq, ex)
    y = 485
    if renorm:
        rn = p.box("Renormalise (per sample)<br>rescale gates so their cell-weighted mean over both streams = 1.0<br>"
                   "→ gates redistribute trust between channels/streams; cannot change overall feature scale", CM, y,
                   640, 90, "renorm")
        p.edge(ex, rn)
        ex = rn
        y += 90 + 40
    exp = p.box(f"Expand to per-element<br>broadcast each channel's gate to its grid cells<br>"
               f"(B, 2, {fmt(FEAT)})", CM, y, 460, 80, "gated")
    p.edge(ex, exp)
    y += 80 + 40
    side_feed(p, rgb, tac, exp, exp)
    f = p.box(f"[g_rgb⊙RGB | g_tac⊙Tactile]<br>{fmt(2 * FEAT)}-dim", CM, y, 420, 80, "fused")
    p.edge(exp, f)
    return f, f, y + 80 + 40


def fusion_gated_cross_attention(p, rgb, tac):
    vis = p.box(f"Visual: Tokenize + Project (Q)<br>reshape {fmt(FEAT)} → {NUM_TOKENS}×{TOKEN_DIM}<br>"
               f"Linear {TOKEN_DIM}→{EMBED_DIM} → LayerNorm<br>vis_tokens (B,{NUM_TOKENS},{EMBED_DIM})", CL, 205, 380,
               120, "rgb")
    tacb = p.box(f"Tactile: Tokenize + Project (K,V)<br>reshape {fmt(FEAT)} → {NUM_TOKENS}×{TOKEN_DIM}<br>"
                f"Linear {TOKEN_DIM}→{EMBED_DIM} → LayerNorm<br>tac_tokens (B,{NUM_TOKENS},{EMBED_DIM})", CR, 205,
                380, 120, "tac")
    p.edge(rgb, vis)
    p.edge(tac, tacb)

    attn = p.box(f"Cross-Attention (MultiheadAttention)<br>Q = vis_tokens, K = V = tac_tokens<br>"
                f"embed_dim {EMBED_DIM}, heads {NUM_HEADS}, dropout 0.1 → LayerNorm<br>"
                f"attn_out T_attn (B,{NUM_TOKENS},{EMBED_DIM})", CM, 365, 620, 110, "attn")
    p.edge(vis, attn, fa=0.9, fb=0.1, midy=345)
    p.edge(tacb, attn, fa=0.1, fb=0.9, midy=345)

    gate = p.box(f"Gate<br>Linear {2*EMBED_DIM}→{EMBED_DIM} (Sigmoid) on [vis_tokens ‖ attn_out]<br>"
                f"→ g ∈ (0,1)^({NUM_TOKENS}×{EMBED_DIM}), per-token per-channel", CM, 515, 600, 90, "gate")
    p.edge(attn, gate)
    # vis_tokens is needed again here (gate reads [vis_tokens || attn_out]); route it down the
    # left margin, outside every box's footprint, then in through the gate's left side.
    p.edge(vis, gate, sa="l", fa=0.85, ta="l", fb=0.5, via=[(20, 560)])

    fuse = p.box(f"Fuse (per token, per channel)<br>T_fused = g ⊙ attn_out + (1 − g) ⊙ vis_tokens<br>"
               f"(B,{NUM_TOKENS},{EMBED_DIM})", CM, 645, 500, 90, "gated")
    p.edge(gate, fuse)
    p.edge(attn, fuse, sa="r", fa=0.7, ta="r", fb=0.5, via=[(880, 690)])
    p.edge(vis, fuse, sa="l", fa=0.95, ta="l", fb=0.5, via=[(20, 690)])

    pool = p.box(f"Global Average Pool (over {NUM_TOKENS} tokens)<br>v_gca (B, {EMBED_DIM})", CM, 775, 420, 70,
                "fused")
    p.edge(fuse, pool)
    out = p.box(f"Output Projection<br>Linear {EMBED_DIM}→{fmt(FEAT)}<br>fused (B, {fmt(FEAT)})", CM, 885, 420, 70,
               "fused")
    p.edge(pool, out)
    return out, out, 995


def fusion_se_gated_cross_attention(p, rgb, tac):
    # Row 1 (y=205): tokenize + project each stream. The positional encoder is independent of
    # RGB/Tactile (built only from fixed per-token coordinates), so it gets its own row below
    # rather than sharing row 1 -- sharing it would overlap the vis/tac boxes' footprint.
    vis = p.box(f"Visual: Tokenize + Project<br>reshape {fmt(FEAT)} → {NUM_TOKENS}×{TOKEN_DIM}<br>"
               f"Linear {TOKEN_DIM}→{EMBED_DIM} → LayerNorm", CL, 205, 360, 90, "rgb")
    tacb = p.box(f"Tactile: Tokenize + Project<br>reshape {fmt(FEAT)} → {NUM_TOKENS}×{TOKEN_DIM}<br>"
                f"Linear {TOKEN_DIM}→{EMBED_DIM} → LayerNorm", CR, 205, 360, 90, "tac")
    p.edge(rgb, vis)
    p.edge(tac, tacb)

    # Row 2 (y=325): positional encoding, independent of both streams.
    pos = p.box(f"Scale-Aware Positional Encoding<br>(independent of RGB/Tactile -- built from fixed "
               f"per-token coordinates)<br>2D Fourier (x,y) per token + scale Embedding(3) for P3/P4/P5"
               f"<br>→ pos_emb (B,{NUM_TOKENS},{EMBED_DIM})", CM, 325, 460, 90, "pos")

    # Row 3 (y=445): add pos_emb to each stream's tokens, then LayerNorm.
    vis2 = p.box(f"+ pos_emb → LayerNorm<br>vis_tokens (B,{NUM_TOKENS},{EMBED_DIM}) [Q]", CL, 445, 340, 70, "rgb")
    tac2 = p.box(f"+ pos_emb → LayerNorm<br>tac_tokens (B,{NUM_TOKENS},{EMBED_DIM}) [K,V]", CR, 445, 340, 70, "tac")
    p.edge(vis, vis2)
    p.edge(tacb, tac2)
    p.edge(pos, vis2, fa=0.2, fb=0.8)
    p.edge(pos, tac2, fa=0.8, fb=0.2)

    # Row 4 (y=555): cross-attention.
    attn = p.box(f"Cross-Attention (MultiheadAttention)<br>Q = vis_tokens, K = V = tac_tokens<br>"
                f"embed_dim {EMBED_DIM}, heads {NUM_HEADS}, dropout 0.1 → LayerNorm<br>"
                f"attn_out T_attn (B,{NUM_TOKENS},{EMBED_DIM})", CM, 555, 620, 110, "attn")
    p.edge(vis2, attn, fa=0.9, fb=0.1, midy=535)
    p.edge(tac2, attn, fa=0.1, fb=0.9, midy=535)

    # Row 5 (y=705): the two independent gate paths -- spatial (CL) and SE squeeze (CR) -- run
    # in parallel, each only reading attn_out / vis_tokens / tac_tokens, not each other.
    sgate = p.box(f"Spatial Gate (per token)<br>Linear {2*EMBED_DIM}→{EMBED_DIM} (Sigmoid)<br>"
                 f"on [vis_tokens ‖ attn_out]<br>→ g_spatial (B,{NUM_TOKENS},{EMBED_DIM})", CL, 705, 340, 110, "gate")
    p.edge(attn, sgate, fa=0.15, fb=0.9)
    p.edge(vis2, sgate, sa="l", fa=0.9, ta="l", fb=0.5, via=[(20, 760)])

    se_sq = p.box(f"Squeeze (SE descriptor)<br>mean + mean|x| over the {NUM_TOKENS} tokens, for vis_tokens, "
                 f"tac_tokens, attn_out<br>→ concat → {fmt(SE_GCA_DESC_DIM)}-dim descriptor", CR, 705, 340, 110,
                 "stack")
    p.edge(attn, se_sq, fa=0.85, fb=0.1)
    # vis_tokens is needed again here; route right along the row-3/4 gap (y=535, above attn's
    # top at y=555) then down the right margin (x=880, clear of se_sq/se_ex which end at x=830).
    p.edge(vis2, se_sq, fa=0.9, ta="r", fb=0.3, via=[(880, 535), (880, 738)])
    p.edge(tac2, se_sq, fa=0.9, fb=0.9)

    # Row 6 (y=855): SE excite, below its own squeeze (CR column only -- spatial gate has no
    # counterpart here, so nothing else occupies this row's CL/CM space).
    se_ex = p.box(f"Excite (SE block)<br>LayerNorm({SE_GCA_DESC_DIM}) → Linear {SE_GCA_DESC_DIM}→"
                 f"{SE_GCA_HIDDEN_DIM} → ReLU → Dropout 0.1<br>→ Linear {SE_GCA_HIDDEN_DIM}→{EMBED_DIM} → Sigmoid<br>"
                 f"→ se_channel_weights (B,1,{EMBED_DIM}), per-channel, shared over all tokens", CR, 855, 340, 130,
                 "gate")
    p.edge(se_sq, se_ex)

    # Row 7 (y=1025): combine the two gates (below both row-5/6 branches, so no overlap).
    hybrid = p.box(f"g_hybrid = g_spatial ⊙ se_channel_weights<br>(se weights broadcast over the {NUM_TOKENS} "
                  "tokens)", CM, 1025, 460, 70, "gated")
    p.edge(sgate, hybrid, fa=0.9, fb=0.1, midy=1005)
    p.edge(se_ex, hybrid, fa=0.1, fb=0.9, midy=1005)

    # Row 8 (y=1135): fuse -- needs g_hybrid (straight above), attn_out and vis_tokens (routed
    # down the outer margins, clear of every box between rows 4 and 8).
    fuse = p.box(f"Fuse (per token, per channel)<br>T_fused = g_hybrid ⊙ attn_out + (1 − g_hybrid) ⊙ vis_tokens<br>"
               f"(B,{NUM_TOKENS},{EMBED_DIM})", CM, 1135, 500, 90, "gated")
    p.edge(hybrid, fuse)
    # attn_out / vis_tokens routed down the outer margins (x=880 / x=20), clear of every box
    # between rows 4 and 8 (se_sq/se_ex max out at x=830 on the right, sgate starts at x=70 on
    # the left) -- see the row-5/6/7 comments above for the exact box extents.
    p.edge(attn, fuse, sa="r", fa=0.8, ta="r", fb=0.5, via=[(880, 643), (880, 1180)])
    p.edge(vis2, fuse, sa="l", fa=0.9, ta="l", fb=0.5, via=[(20, 508), (20, 1180)])

    pool = p.box(f"Global Average Pool (over {NUM_TOKENS} tokens)<br>v_gca (B, {EMBED_DIM})", CM, 1265, 420, 70,
                "fused")
    p.edge(fuse, pool)
    out = p.box(f"Output Projection<br>Linear {EMBED_DIM}→{fmt(FEAT)}<br>fused (B, {fmt(FEAT)})", CM, 1375, 420, 70,
               "fused")
    p.edge(pool, out)
    return out, out, 1485


def build_page(cand):
    kind = cand["kind"]
    # sanity: recompute the parameter total from the layer sizes above
    total = (BACKBONES_AND_NECK_PARAMS + cand["fusion_params"] + mlp_params(cand["fused"], cand["rot"], 4)
             + mlp_params(cand["fused"], cand["trans"], 2))
    assert total == cand["total"], (cand["title"], total, cand["total"])

    p = Page(cand["file"])
    p.title(cand["title"])
    rgb, tac = features(p)
    if kind == "concat":
        sr, st, y = fusion_concat(p, rgb, tac)
    elif kind == "gated":
        sr, st, y = fusion_gated(p, rgb, tac)
    elif kind in ("gated_concat", "residual_gated_concat"):
        sr, st, y = fusion_scalar_concat(p, rgb, tac, residual=kind.startswith("residual"))
    elif kind in ("cw_local", "cw_res_local"):
        sr, st, y = fusion_channelwise_local(p, rgb, tac, residual=kind == "cw_res_local")
    elif kind in ("se_channel", "se_channel_renorm"):
        sr, st, y = fusion_se_channel(p, rgb, tac, renorm=kind == "se_channel_renorm")
    elif kind == "gca":
        sr, st, y = fusion_gated_cross_attention(p, rgb, tac)
    elif kind == "se_gca":
        sr, st, y = fusion_se_gated_cross_attention(p, rgb, tac)
    else:
        sr, st, y = fusion_dual(p, rgb, tac)
    height = heads(p, cand, sr, st, y)
    return p, height


def _indent(elem, level=0):
    """Pretty-print helper (ET.indent only exists on Python >= 3.9)."""
    pad = "\n" + level * " "
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + " "
        for child in elem:
            _indent(child, level + 1)
            if not (child.tail or "").strip():
                child.tail = pad + " "
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


def write_mxfile(pages, path):
    root = ET.Element("mxfile", host="app.diagrams.net", agent="generate_architecture_diagrams.py", version="24.0.0")
    for d in pages:
        root.append(d)
    _indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "docs" / "architecture_diagrams"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_pages = []
    for cand in CANDIDATES:
        page, height = build_page(cand)
        write_mxfile([page.to_xml(height)], out / f"{cand['file']}.drawio")
        all_pages.append(page.to_xml(height))
        print(f"wrote {out / (cand['file'] + '.drawio')}  ({cand['total']:,} params)")
    write_mxfile(all_pages, out / "all_candidates.drawio")
    print(f"wrote {out / 'all_candidates.drawio'}")


if __name__ == "__main__":
    sys.exit(main())
