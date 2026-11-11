"""
Fig. 1 - Block diagram of the conformal stacking pipeline.

Layout is defined in INCHES (axes units == figure inches). Every box is an explicit
rectangle and every arrow is drawn edge-to-edge between rectangles, so nothing is
positioned by guessing where rendered text will land.

Output: true vector PDF at exactly 7.16 in (IEEE two-column width) + PNG preview.
Greyscale only. No author, institution, or product names (double-blind safe).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

FIG_W, FIG_H = 7.16, 3.70
FS = 8.0                 # body font size (pt) - never below this
FS_LABEL = 8.0           # group label font size

ROW1_Y = 2.50            # vertical centre, row 1
ROW2_Y = 0.72            # vertical centre, row 2
GTOP, GBOT = 3.44, 1.98  # Level 1 / Level 2 group frames
LINK_Y = 1.60            # horizontal run of the row1 -> row2 connector

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
ax.set_aspect("equal")


def layout(widths, x_start, x_end):
    total = sum(widths)
    gap = (x_end - x_start - total) / (len(widths) - 1)
    spans, x = [], x_start
    for w in widths:
        spans.append((x, x + w))
        x += w + gap
    return spans


def box(x0, x1, yc, h, text, dashed=False, lw=1.1, z=3):
    p = FancyBboxPatch(
        (x0, yc - h / 2), x1 - x0, h,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor="white", edgecolor="black",
        linewidth=lw, linestyle="--" if dashed else "-", zorder=z)
    ax.add_patch(p)
    ax.text((x0 + x1) / 2, yc, text, ha="center", va="center",
            fontsize=FS, zorder=z + 1, linespacing=1.35)
    return {"x0": x0, "x1": x1, "yc": yc, "top": yc + h / 2, "bot": yc - h / 2}


def arrow(p0, p1, z=2):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.1,
                                shrinkA=0, shrinkB=0, mutation_scale=9), zorder=z)


def connect_h(a, b):
    arrow((a["x1"], a["yc"]), (b["x0"], b["yc"]))


# ------------------------------------------------------------------- ROW 1
s_drug, s_cipa, s_bio, s_l1, s_l2 = layout(
    [0.86, 1.12, 1.26, 1.46, 1.24], 0.16, 7.00)

b_drug = box(*s_drug, ROW1_Y, 0.44, "Drug\n(IC$_{50}$, Hill)")
b_cipa = box(*s_cipa, ROW1_Y, 0.44, "CiPA cell-model\nsimulation")
b_bio  = box(*s_bio,  ROW1_Y, 0.60, "11 biomarkers\n(pre-normalized,\nno scaling)")

for gx0, gx1 in (s_l1, s_l2):
    ax.add_patch(Rectangle((gx0, GBOT), gx1 - gx0, GTOP - GBOT,
                           facecolor="none", edgecolor="black",
                           linestyle=":", linewidth=1.2, zorder=1))

ax.text(sum(s_l1) / 2, GTOP - 0.07,
        "LEVEL 1  base learners\npredict_proba via\nStratifiedGroupKFold (k=4, by drug)",
        ha="center", va="top", fontsize=FS_LABEL, fontweight="bold",
        linespacing=1.25, zorder=2)
ax.text(sum(s_l2) / 2, GTOP - 0.07, "LEVEL 2\nstacking",
        ha="center", va="top", fontsize=FS_LABEL, fontweight="bold",
        linespacing=1.25, zorder=2)

b_rf   = box(s_l1[0] + 0.11, s_l1[1] - 0.11, 2.75, 0.34, "Random Forest")
b_xgb  = box(s_l1[0] + 0.11, s_l1[1] - 0.11, 2.25, 0.34, "XGBoost")
b_meta = box(s_l2[0] + 0.11, s_l2[1] - 0.11, ROW1_Y, 0.50,
             "Random Forest\nmeta-learner")

connect_h(b_drug, b_cipa)
connect_h(b_cipa, b_bio)
arrow((b_bio["x1"], ROW1_Y), (b_rf["x0"], b_rf["yc"]))
arrow((b_bio["x1"], ROW1_Y), (b_xgb["x0"], b_xgb["yc"]))
arrow((b_rf["x1"], b_rf["yc"]),  (b_meta["x0"], b_meta["yc"] + 0.10))
arrow((b_xgb["x1"], b_xgb["yc"]), (b_meta["x0"], b_meta["yc"] - 0.10))

# ------------------------------------------------------------------- ROW 2
s_cp, s_dm, s_cf, s_ps, s_act = layout(
    [1.12, 1.30, 1.44, 1.06, 1.20], 0.16, 6.96)

b_cp = box(*s_cp, ROW2_Y, 0.60, "Class probabilities\n(high / int. / low)")
b_dm = box(*s_dm, ROW2_Y, 0.60, "Drug-level mean\nprobabilities\n(over 2000 cells)")

pan_x0, pan_x1 = s_cf[0] - 0.13, s_act[1] + 0.13
ax.add_patch(Rectangle((pan_x0, 0.14), pan_x1 - pan_x0, 1.24,
                       facecolor="#ededed", edgecolor="black",
                       linestyle="--", linewidth=1.2, zorder=0))
ax.text(pan_x0 + 0.08, 1.30, "Safety assurance layer", ha="left", va="top",
        fontsize=FS_LABEL, fontstyle="italic", fontweight="bold", zorder=1)

b_cf  = box(*s_cf, ROW2_Y, 0.60,
            "Cross-conformal layer\n(" + r"$\hat{q}$" + " from LODO\ncalibration)",
            dashed=True, lw=1.5)
b_ps  = box(*s_ps, ROW2_Y, 0.60, "PREDICTION\nSET")
b_act = box(*s_act, ROW2_Y, 0.60, "Clinical action =\nmost severe\nclass in set")

connect_h(b_cp, b_dm)
connect_h(b_dm, b_cf)
connect_h(b_cf, b_ps)
connect_h(b_ps, b_act)

# --------------------------------------------- ROW 1 -> ROW 2 CONNECTOR
cx, tx = sum(s_l2) / 2, sum(s_cp) / 2
ax.plot([cx, cx], [GBOT, LINK_Y], color="black", lw=1.1, zorder=2)
ax.plot([cx, tx], [LINK_Y, LINK_Y], color="black", lw=1.1, zorder=2)
arrow((tx, LINK_Y), (tx, b_cp["top"]))

out = "/home/claude/fig1_block_diagram"
fig.savefig(out + ".pdf", format="pdf")
fig.savefig(out + ".png", format="png", dpi=200)
print("pdf KB:", round(os.path.getsize(out + ".pdf") / 1024, 1))