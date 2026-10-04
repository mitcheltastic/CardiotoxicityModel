"""Generate End-to-End Model Architecture & Deployment Pipeline Diagram (Fig. Architecture).

This script generates a comprehensive, publication-quality diagram illustrating:
1. Data Input & Preprocessing (11 Biomarkers, 28 Drugs, Sentinel Filtering)
2. Grouped Cross-Validation (StratifiedGroupKFold, k=4 by Drug)
3. Stacking Ensemble Architecture (RF + XGB -> Logistic Regression Meta-Learner)
4. Cross-Conformal Calibration (LODO, q_hat = 0.6275)
5. Gradio & Hugging Face Deployment Package (hf_space/ with Conformal Sets & SHAP)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Ellipse, Polygon, Circle, FancyBboxPatch

NAVY, MID, LITE, PALE = "#1B4F72", "#2E86C1", "#A9CCE3", "#EDF4FA"
ACCENT, WARN_COLOR = "#D4AC0D", "#C0392B"
INK = "#000000"
LW = 1.0

FIG_W, FIG_H = 12.0, 6.5
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H); ax.axis("off")

# ---------------------------------------------------------------- Primitives
def box(x, y, w, h, fc=PALE, ec=MID, lw=LW, ls="-", z=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, linestyle=ls, zorder=z))

def fbox(x, y, w, h, fc=PALE, ec=MID, lw=LW, ls="-", z=2, boxstyle="round,pad=0.03"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=boxstyle, facecolor=fc,
                                edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z))

def txt(x, y, s, size=8.0, weight="normal", ha="center", va="center", color=INK, style="normal", z=8):
    return ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color,
                   fontweight=weight, zorder=z, style=style)

def arrow(x0, y0, x1, y1, rad=0.0, dashed=False, lw=1.1, color=INK, z=7):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
        linewidth=lw, color=color, zorder=z, shrinkA=0, shrinkB=0,
        joinstyle="miter", capstyle="butt",
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))

def cylinder(cx, cy, w, h):
    rx = w / 2
    y = cy + h / 2
    for fc, fr in ((NAVY, 0.34), (MID, 0.33), (LITE, 0.33)):
        bh = h * fr
        ax.add_patch(Rectangle((cx - rx, y - bh), w, bh, facecolor=fc,
                               edgecolor=INK, linewidth=0.8, zorder=4))
        y -= bh
    ax.add_patch(Ellipse((cx, cy - h/2), w, 0.08, facecolor=LITE,
                         edgecolor=INK, linewidth=0.8, zorder=3))
    ax.add_patch(Ellipse((cx, cy + h/2), w, 0.08, facecolor="#D6EAF8",
                         edgecolor=INK, linewidth=0.8, zorder=6))

# ---------------------------------------------------------------- Header
txt(FIG_W/2, 6.20, "CARDIO TOXICITY ML PIPELINE & DEPLOYMENT ARCHITECTURE", 13.0, "bold", color=NAVY)
txt(FIG_W/2, 5.95, "End-to-End Workflow: Data Ingestion -> Preprocessing -> Stacking Ensemble -> Conformal Calibration -> Gradio UI", 8.5, style="italic")

# ========================================================= STAGE 1: Data & Preprocessing
S1_X, S1_Y, S1_W, S1_H = 0.40, 0.40, 2.70, 5.20
box(S1_X, S1_Y, S1_W, S1_H, fc="white", ec=NAVY, lw=1.2)
txt(S1_X + S1_W/2, S1_Y + S1_H - 0.25, "STAGE 1: Data & Preprocessing", 9.5, "bold", color=NAVY)

# Sub-box 1A: Benchmark Dataset
box(S1_X + 0.15, 3.75, 2.40, 1.45, fc=PALE, ec=MID)
txt(S1_X + 1.35, 5.00, "28 CiPA Benchmark Drugs", 8.5, "bold")
cylinder(S1_X + 0.65, 4.30, 0.70, 0.45)
txt(S1_X + 0.65, 3.92, "2,000 cells/drug", 6.8)
txt(S1_X + 1.80, 4.60, "12 Train Drugs\n(24,000 cells)", 7.2, "bold")
txt(S1_X + 1.80, 4.05, "16 Test Drugs\n(32,000 cells)", 7.2, "bold")

# Sub-box 1B: Sentinel Cleaning
box(S1_X + 0.15, 2.30, 2.40, 1.30, fc="#FADBD8", ec=WARN_COLOR)
txt(S1_X + 1.35, 3.40, "Data Audit & Cleaning", 8.5, "bold", color=WARN_COLOR)
txt(S1_X + 1.35, 3.10, "Sentinel check ('#VALUE!')", 7.2)
txt(S1_X + 1.35, 2.80, "Dropped 26 corrupted rows\n(1.25% mexiletine, 0.05% ibutilide)", 7.0)
txt(S1_X + 1.35, 2.45, "Audit Log: DROPPED_ROWS_REV.csv", 6.8, style="italic")

# Sub-box 1C: Features
box(S1_X + 0.15, 0.55, 2.40, 1.60, fc=PALE, ec=MID)
txt(S1_X + 1.35, 1.95, "11 Canonical Biomarkers", 8.5, "bold")
bms = ["qNet", "dvdtmax", "vmax", "vrest", "APD50", "APD90", "max_dv", "camax", "carest", "CaTD50", "CaTD90"]
for i, bm in enumerate(bms[:6]):
    txt(S1_X + 0.75, 1.70 - i*0.18, bm, 6.8)
for i, bm in enumerate(bms[6:]):
    txt(S1_X + 1.95, 1.70 - i*0.18, bm, 6.8)
txt(S1_X + 1.35, 0.70, "*Pre-normalized by provider (no StandardScaler)", 6.5, style="italic", color=NAVY)

# ========================================================= STAGE 2: Grouped CV
S2_X, S2_Y, S2_W, S2_H = 3.35, 0.40, 2.20, 5.20
box(S2_X, S2_Y, S2_W, S2_H, fc="white", ec=NAVY, lw=1.2)
txt(S2_X + S2_W/2, S2_Y + S2_H - 0.25, "STAGE 2: Grouped CV", 9.5, "bold", color=NAVY)

txt(S2_X + S2_W/2, 5.15, "StratifiedGroupKFold\n(k = 4, by drug_name)", 8.2, "bold")

# Show 4 Folds
for fold in range(4):
    fy = 3.95 - fold*0.88
    box(S2_X + 0.15, fy, 1.90, 0.72, fc=PALE, ec=MID)
    txt(S2_X + 1.10, fy + 0.52, f"Fold {fold+1}: 9 Train / 3 Val", 7.5, "bold")
    # Draw mini block
    for b in range(4):
        bx = S2_X + 0.35 + b*0.40
        color = MID if b == fold else NAVY
        ax.add_patch(Rectangle((bx, fy + 0.10), 0.32, 0.22, facecolor=color, edgecolor=INK, lw=0.6))
        txt(bx + 0.16, fy + 0.21, "V" if b == fold else "T", 6.5, color="white", weight="bold")

txt(S2_X + S2_W/2, 0.70, "Guarantees zero drug leakage\nfor out-of-fold meta-features", 7.2, style="italic", color=NAVY)

# Connect Stage 1 to Stage 2
arrow(S1_X + S1_W, 3.00, S2_X, 3.00)

# ========================================================= STAGE 3: Stacking Ensemble
S3_X, S3_Y, S3_W, S3_H = 5.80, 0.40, 3.10, 5.20
box(S3_X, S3_Y, S3_W, S3_H, fc="white", ec=NAVY, lw=1.2)
txt(S3_X + S3_W/2, S3_Y + S3_H - 0.25, "STAGE 3: Stacking Ensemble", 9.5, "bold", color=NAVY)

# Level 1 Box
box(S3_X + 0.15, 2.70, 2.80, 2.45, fc="#EBF5FB", ec=MID, ls="--")
txt(S3_X + 1.55, 4.95, "Level 1: Base Learners", 8.5, "bold", color=NAVY)

# Base Learner 1: RF
box(S3_X + 0.30, 3.85, 2.50, 0.90, fc="white", ec=MID)
txt(S3_X + 1.55, 4.50, "Random Forest Classifier", 8.0, "bold")
txt(S3_X + 1.55, 4.25, "n_estimators=400, criterion='entropy'", 7.0)
txt(S3_X + 1.55, 4.02, "Outputs: [p_high, p_int, p_low]_RF", 6.8, style="italic", color=NAVY)

# Base Learner 2: XGB
box(S3_X + 0.30, 2.85, 2.50, 0.90, fc="white", ec=MID)
txt(S3_X + 1.55, 3.50, "XGBoost Classifier", 8.0, "bold")
txt(S3_X + 1.55, 3.25, "n_estimators=300, lr=0.1, max_depth=10", 7.0)
txt(S3_X + 1.55, 3.02, "Outputs: [p_high, p_int, p_low]_XGB", 6.8, style="italic", color=NAVY)

# Level 2 Box: Meta Learner
box(S3_X + 0.15, 0.55, 2.80, 1.85, fc="#EAFAF1", ec="#27AE60")
txt(S3_X + 1.55, 2.20, "Level 2: Meta-Learner (Blender)", 8.5, "bold", color="#1E8449")
box(S3_X + 0.30, 1.05, 2.50, 0.90, fc="white", ec="#27AE60")
txt(S3_X + 1.55, 1.70, "Logistic Regression Meta-Learner", 8.0, "bold")
txt(S3_X + 1.55, 1.45, "C=1.0, max_iter=1000", 7.0)
txt(S3_X + 1.55, 1.20, "Produces smooth, calibrated probabilities!", 7.0, "bold", color="#1E8449")
txt(S3_X + 1.55, 0.72, "*Fixes 100%/0% step-function decision tree artifacts", 6.8, style="italic", color=NAVY)

# Arrows inside Stage 3
arrow(S3_X + 1.55, 2.85, S3_X + 1.55, 1.95)
arrow(S2_X + S2_W, 3.00, S3_X + 0.15, 3.00)

# ========================================================= STAGE 4: Conformal Calibration & HF App
S4_X, S4_Y, S4_W, S4_H = 9.15, 0.40, 2.45, 5.20
box(S4_X, S4_Y, S4_W, S4_H, fc="white", ec=NAVY, lw=1.2)
txt(S4_X + S4_W/2, S4_Y + S4_H - 0.25, "STAGE 4: Calibration & App", 9.5, "bold", color=NAVY)

# Conformal Box
box(S4_X + 0.15, 3.35, 2.15, 1.80, fc="#FEF9E7", ec=ACCENT)
txt(S4_X + 1.22, 4.95, "Cross-Conformal Calibration", 8.2, "bold", color="#7D6608")
txt(S4_X + 1.22, 4.65, "Leave-One-Drug-Out (LODO)\nover 12 training drugs", 7.2)
box(S4_X + 0.25, 3.85, 1.95, 0.55, fc="#F9E79F", ec=ACCENT)
txt(S4_X + 1.22, 4.20, "Threshold q_hat = 0.6275", 7.8, "bold")
txt(S4_X + 1.22, 4.00, "(11th of 12 LODO scores)", 7.0, style="italic")
txt(S4_X + 1.22, 3.52, "Nominal Coverage: 84.6%", 7.2, "bold")

# App / Deployment Box
box(S4_X + 0.15, 0.55, 2.15, 2.50, fc=PALE, ec=MID)
txt(S4_X + 1.22, 2.85, "Hugging Face Space App", 8.5, "bold", color=NAVY)
txt(S4_X + 1.22, 2.60, "mitcheltastic/tdp-cipa-screening", 6.8, style="italic")

# Features of App
box(S4_X + 0.25, 1.80, 1.95, 0.65, fc="white", ec=MID)
txt(S4_X + 1.22, 2.25, "Conformal Prediction Set", 7.5, "bold")
txt(S4_X + 1.22, 2.00, "C(X) = {c: p_c >= 1 - q_hat}", 6.8, color=NAVY)

box(S4_X + 0.25, 1.15, 1.95, 0.55, fc="white", ec=MID)
txt(S4_X + 1.22, 1.50, "Clinical Safety Action", 7.5, "bold")
txt(S4_X + 1.22, 1.30, "Most severe class in C(X)", 6.8, color=WARN_COLOR, weight="bold")

box(S4_X + 0.25, 0.65, 1.95, 0.42, fc="white", ec=MID)
txt(S4_X + 1.22, 0.86, "SHAP Feature Attribution", 7.2, "bold")

# Arrows Stage 3 to Stage 4
arrow(S3_X + S3_W, 1.50, S4_X + 0.15, 1.50)
arrow(S3_X + S3_W, 4.25, S4_X + 0.15, 4.25)

# Save Outputs
out_pdf = os.path.join(os.path.dirname(__file__), "fig_architecture.pdf")
out_png = os.path.join(os.path.dirname(__file__), "fig_architecture.png")

fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
print(f"Saved Architecture Diagram PDF: {out_pdf}")
print(f"Saved Architecture Diagram PNG: {out_png}")
