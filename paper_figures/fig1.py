"""Fig. 1 - Pipeline diagram, v9.

Changes vs v8
-------------
0. First classifier is now XGBoost (was: stacking RF + XGB).  The neural-net
   glyph is replaced by a single boosted-tree glyph -- one model, not an
   ensemble of two.  Everything else is byte-identical to v8.

Carried over from v7 -> v8
--------------------------
1. Cylinder captions are now placed by a helper that reserves a guaranteed
   gap below the bottom ellipse (no more "12 training drugs" touching the drum).
2. Every arrow lands on the *edge* of its target with a tangent that points
   INTO the target (no more grazing).  Feeder arrows into the CV block and
   into the calibration block are horizontal, so the head is unambiguous.
3. Vertical rhythm reworked: the two drums sit exactly in the gutters of the
   model stack, so the horizontal feeds never collide with a model box.
4. pdf.fonttype = 42 -> TrueType embedded + subsetted (IEEE PDF eXpress).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType, not Type-3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyArrowPatch, Ellipse, Rectangle,
                                Polygon, Circle)

NAVY, MID, LITE, PALE = "#1B4F72", "#2E86C1", "#A9CCE3", "#EDF4FA"
INK = "#000000"
LW = 1.0
CYL_RY = 0.042          # semi-minor axis of the cylinder end-caps

FIG_W, FIG_H = 7.16, 3.02
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H); ax.axis("off")


# ---------------------------------------------------------------- primitives
def cylinder(cx, cy, w, h):
    rx = w / 2
    y = cy + h / 2
    for fc, fr in ((NAVY, 0.34), (MID, 0.33), (LITE, 0.33)):
        bh = h * fr
        ax.add_patch(Rectangle((cx - rx, y - bh), w, bh, facecolor=fc,
                               edgecolor=INK, linewidth=LW, zorder=4))
        y -= bh
    ax.add_patch(Ellipse((cx, cy - h/2), w, CYL_RY*2, facecolor=LITE,
                         edgecolor=INK, linewidth=LW, zorder=3))
    ax.add_patch(Ellipse((cx, cy + h/2), w, CYL_RY*2, facecolor="#D6EAF8",
                         edgecolor=INK, linewidth=LW, zorder=6))


def sbox(x, y, w, h, fc, z=4, ls="-", lw=LW, ec=INK):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, linestyle=ls, zorder=z))


def txt(x, y, s, size, weight="normal", ha="center", va="center",
        ls=1.3, z=8, style="normal"):
    return ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=INK,
                   fontweight=weight, linespacing=ls, zorder=z, style=style)


def caption(cx, cy, h, s, size=6.3, gap=0.07):
    """Caption under a cylinder, with a guaranteed `gap` (inches) of air."""
    y = cy - h/2 - CYL_RY - gap - 0.5 * (size / 72.0) * 1.16
    txt(cx, y, s, size, "bold")
    return y


def arrow(x0, y0, x1, y1, rad=0.0, dashed=False, lw=1.15, z=7):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
        linewidth=lw, color=INK, zorder=z, shrinkA=0, shrinkB=0,
        joinstyle="miter", capstyle="butt",
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))


# -------------------------------------------------------------------- icons
def tree_icon(cx, cy, s):
    n = [(0, 1), (-.7, .1), (.7, .1), (-1.05, -.85), (-.35, -.85),
         (.35, -.85), (1.05, -.85)]
    for a, b in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)):
        ax.plot([cx+n[a][0]*s, cx+n[b][0]*s], [cy+n[a][1]*s, cy+n[b][1]*s],
                color=INK, lw=0.7, zorder=8, solid_capstyle="round")
    for (px, py), fc in zip(n, [NAVY, MID, MID, LITE, LITE, LITE, LITE]):
        ax.add_patch(Circle((cx+px*s, cy+py*s), 0.17*s, facecolor=fc,
                            edgecolor=INK, linewidth=0.6, zorder=9))


def forest_icon(cx, cy, s):
    tree_icon(cx - 0.40*s, cy, s*0.60)
    tree_icon(cx + 0.40*s, cy, s*0.60)


def boost_icon(cx, cy, s):
    """Single tree -> one boosted-tree model, not an ensemble of models.

    Drawn at ~1.4x the scale of one forest tree so that at print size the
    contrast with the Random-forest box reads as one-vs-many, not as a
    smaller copy of the same glyph.  The tree spans -0.85s..+1.0s in y, so
    nudge it down to sit optically centred inside the box.
    """
    tree_icon(cx, cy - 0.075*s, s)


def sigmoid_icon(cx, cy, s, dot=False):
    ax.add_patch(Rectangle((cx-s, cy-s*0.82), s*2, s*1.64, facecolor="white",
                           edgecolor=INK, linewidth=0.7, zorder=8))
    t = np.linspace(-0.84, 0.84, 60)
    y = 1/(1+np.exp(-5.5*t))
    ax.plot(cx+t*s, cy+(y-0.5)*1.30*s, color=NAVY, lw=1.4, zorder=9,
            solid_capstyle="round")
    if dot:
        ax.add_patch(Circle((cx+0.54*s, cy+0.43*s), 0.15*s, facecolor=MID,
                            edgecolor=INK, linewidth=0.6, zorder=10))


def funnel_icon(cx, cy, s):
    ax.add_patch(Polygon([(cx-s, cy+s*.78), (cx+s, cy+s*.78),
                          (cx+s*.17, cy-s*.18), (cx+s*.17, cy-s*.88),
                          (cx-s*.17, cy-s*.88), (cx-s*.17, cy-s*.18)],
                         closed=True, facecolor=MID, edgecolor=INK,
                         linewidth=LW, zorder=8))


def clipboard_icon(cx, cy, s):
    ax.add_patch(Rectangle((cx-s*.64, cy-s*.80), s*1.28, s*1.60,
                           facecolor=LITE, edgecolor=INK, linewidth=LW, zorder=8))
    ax.add_patch(Rectangle((cx-s*.24, cy+s*.68), s*.48, s*.22, facecolor=NAVY,
                           edgecolor=INK, linewidth=0.7, zorder=9))
    ax.add_patch(Circle((cx, cy-s*.06), s*.40, facecolor="white",
                        edgecolor=INK, linewidth=0.8, zorder=9))
    ax.plot([cx-s*.19, cx-s*.04, cx+s*.21], [cy-s*.06, cy-s*.22, cy+s*.19],
            color=NAVY, lw=1.3, zorder=10, solid_capstyle="round",
            solid_joinstyle="round")


# ============================================================ STAGE 1 : data
X1 = 0.64
txt(X1, 2.90, "Input dataset", 7.6, "bold")

BM_X, BM_W, BM_Y, BM_H = 0.03, 1.22, 1.66, 1.10
sbox(BM_X, BM_Y, BM_W, BM_H, "white")
txt(X1, 2.66, "11 in silico biomarkers", 6.4, "bold")
bl = ["qNet", r"APD$_{50}$, APD$_{90}$", r"CaD$_{50}$, CaD$_{90}$",
      r"$V_{m\,\mathrm{peak}}$, $V_{m\,\mathrm{rest}}$",
      r"$dV_m/dt_{\mathrm{max}}$, $dV_m/dt_{\mathrm{repol}}$",
      r"Ca$_{\mathrm{peak}}$, Ca$_{\mathrm{rest}}$"]
for i, b in enumerate(bl):
    txt(BM_X + 0.10, 2.46 - i*0.128, b, 5.9, ha="left")

CIPA_CY, CIPA_W, CIPA_H = 0.98, 0.74, 0.52
cylinder(X1, CIPA_CY, CIPA_W, CIPA_H)
y = caption(X1, CIPA_CY, CIPA_H, "28 CiPA drugs", 6.4)
txt(X1, y - 0.15, "2000 cells / drug", 5.8)
txt(X1, y - 0.29, "3 platforms", 5.8)

# box bottom -> top end-cap of the drum
arrow(X1, BM_Y - 0.02, X1, CIPA_CY + CIPA_H/2 + CYL_RY + 0.012)

# ======================================================== STAGE 2 : the split
X2, DRUM_W, DRUM_H = 1.94, 0.70, 0.48
TR_CY, TE_CY = 1.705, 0.765           # aligned with the gutters of stage 3
DRUM_L, DRUM_R = X2 - DRUM_W/2, X2 + DRUM_W/2

cylinder(X2, TR_CY, DRUM_W, DRUM_H)
caption(X2, TR_CY, DRUM_H, "12 training drugs", 6.3)

cylinder(X2, TE_CY, DRUM_W, DRUM_H)
caption(X2, TE_CY, DRUM_H, "16 testing drugs", 6.3)

CIPA_R = X1 + CIPA_W/2
# up-branch: leaves the right flank, arrives ~20 deg into the left flank
arrow(CIPA_R + 0.01, 1.12, DRUM_L - 0.005, TR_CY, rad=-0.26)
# down-branch: essentially horizontal
arrow(CIPA_R + 0.01, 0.88, DRUM_L - 0.005, TE_CY, rad=0.08)

# ================================================= STAGE 3 : model comparison
CX, CW = 2.60, 1.60                     # model boxes
CL, CR = CX - 0.06, CX + CW + 0.06      # dashed container
CB, CT = 0.24, 2.22
sbox(CL, CB, CR - CL, CT - CB, PALE, z=2, ls="--", lw=1.0, ec=MID)
txt((CL + CR)/2, 2.46, "Grouped cross-validation", 6.8, "bold")
txt((CL + CR)/2, 2.32, "StratifiedGroupKFold, k = 4, by drug", 5.9)

BH = 0.38
for yy, lab, kind in [(1.75, "XGBoost", "xgb"),
                      (1.28, "Random forest", "for"),
                      (0.81, "Logistic regression", "sig"),
                      (0.34, "Log. reg. (qNet only)", "sigq")]:
    sbox(CX, yy, CW, BH, "white", z=4)
    ix, iy = CX + 0.24, yy + BH/2
    if kind == "xgb":   boost_icon(ix, iy, 0.136)
    elif kind == "for": forest_icon(ix, iy, 0.170)
    elif kind == "sig": sigmoid_icon(ix, iy, 0.110)
    else:               sigmoid_icon(ix, iy, 0.110, dot=True)
    txt(CX + 0.46, iy, lab, 6.4, "bold", ha="left")

# both feeds are horizontal and land in a gutter -> unambiguous heads
arrow(DRUM_R + 0.01, TR_CY, CL - 0.012, TR_CY)
arrow(DRUM_R + 0.01, TE_CY, CL - 0.012, TE_CY)

# ================================================ STAGE 4 : conformal wrapper
S4, CF_W = 5.08, 1.16
CF_L, CF_R = S4 - CF_W/2, S4 + CF_W/2
CF_B, CF_T = 0.50, 2.00
sbox(CF_L, CF_B, CF_W, CF_T - CF_B, "white")
funnel_icon(S4, 1.78, 0.135)
txt(S4, 1.49, "Cross-conformal", 6.8, "bold")
txt(S4, 1.35, "calibration", 6.8, "bold")
txt(S4, 1.15, "leave-one-drug-out", 5.9)
txt(S4, 1.03, "12 refits, 11 drugs each", 5.9)
sbox(S4 - 0.52, 0.77, 1.04, 0.18, LITE, z=5)
txt(S4, 0.86, r"$\hat{q}$ = 11th of 12 scores", 5.7, "bold", z=6)  # <-- rank: update after pipeline re-run
txt(S4, 0.63, "(12th if calibration-only)", 5.2, style="italic")

arrow(CR + 0.02, (CB + CT)/2, CF_L - 0.012, (CB + CT)/2)

# calibration split: high arc, comes down into the top edge at ~45 deg
arrow(X2, TR_CY + DRUM_H/2 + CYL_RY + 0.012, S4, CF_T + 0.015,
      rad=-0.50, dashed=True, lw=1.0)
txt(3.50, 2.92, "calibration data", 5.9, style="italic")

# ============================================== STAGE 5 & 6 : output + action
S5, OUT_W = 6.47, 1.18
OUT_L, OUT_R = S5 - OUT_W/2, S5 + OUT_W/2

PS_B, PS_T = 1.44, 2.02
sbox(OUT_L, PS_B, OUT_W, PS_T - PS_B, "white")
txt(S5, 1.90, "Prediction set", 6.8, "bold")
for i, (lb, fc) in enumerate([("high", MID), ("int.", MID), ("low", "white")]):
    bx = S5 - 0.49 + i*0.34
    sbox(bx, 1.52, 0.30, 0.20, fc, z=5)
    txt(bx + 0.15, 1.62, lb, 5.7, "bold", z=6)

arrow(CF_R + 0.01, 1.62, OUT_L - 0.012, 1.70, rad=-0.10)

CA_B, CA_T = 0.50, 1.14
sbox(OUT_L, CA_B, OUT_W, CA_T - CA_B, "white")
clipboard_icon(OUT_L + 0.25, 0.86, 0.175)
txt(S5 + 0.19, 0.94, "Clinical", 6.8, "bold")
txt(S5 + 0.19, 0.80, "action", 6.8, "bold")
txt(S5, 0.60, "most severe class in set", 5.7)

arrow(S5, PS_B - 0.02, S5, CA_T + 0.012)

# ------------------------------------------------------------------ QA check
import os
out_dir = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(out_dir, "fig1")
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=300)

if __name__ == "__main__":
    fig.canvas.draw()
    inv = ax.transData.inverted()
    print(f"{'text':<40}{'x0':>7}{'x1':>7}{'y0':>7}{'y1':>7}")
    for t in ax.texts:
        bb = t.get_window_extent(fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = inv.transform(bb.get_points())
        flag = "  <<< OUT OF CANVAS" if (x0 < 0 or x1 > FIG_W
                                         or y0 < 0 or y1 > FIG_H) else ""
        print(f"{t.get_text()[:38]:<40}{x0:7.3f}{x1:7.3f}"
              f"{y0:7.3f}{y1:7.3f}{flag}")
    print("saved", out)
