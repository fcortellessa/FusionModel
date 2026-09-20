#!/usr/bin/env python3
"""
Generate draw.io block diagrams of the 7 final-ablation fusion candidates.

The candidates are the same 7 used by scripts/evaluate/final_ablation_candidate_analysis.py
and configs/final_ablation_spec.yml:

    1 concat | 2 gated | 3 gated_concat | 4 residual_gated_concat |
    5 channelwise_gated_concat_local | 6 channelwise_residual_gated_concat_local |
    7 gated_concat + dual_fusion

Each diagram starts at the per-stream feature vectors (flattened P3|P4|P5 = 22,400-d per
stream) and shows fusion -> rotation / translation heads -> outputs. Dimensions were read off
a constructed PoseEstimationNet (yolo26n.pt, 640x640) and the head widths come from
configs/final_ablation_spec.yml; parameter counts are recomputed and asserted against the
measured totals so the head/fusion sizes below cannot silently drift from the code.

Output (default docs/architecture_diagrams/):
    01_concat.drawio ... 07_dual_gated_concat.drawio   one page each
    all_candidates.drawio                              the 7 pages in one file

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
]

# ---------------------------------------------------------------------------
# Styling (matches the reference diagrams: light fills, rounded boxes, plain arrows)
# ---------------------------------------------------------------------------
STYLE = {
    "rgb":    ("#e3f2fd", "#000000", 1.5),
    "tac":    ("#f3e5f5", "#000000", 1.5),
    "stack":  ("#fff8e1", "#ff9800", 1.5),
    "gate":   ("#ffe0b2", "#ff9800", 2),
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
