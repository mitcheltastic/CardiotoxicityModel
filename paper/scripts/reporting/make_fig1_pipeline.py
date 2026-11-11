import os
import sys
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

TRAINAI_DIR = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
RESULTS_DIR = os.path.join(TRAINAI_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Registry to track all text elements and their font sizes
text_registry = []

def add_text(ax, x, y, text, fontsize, color, fontweight='normal', ha='center', va='center', zorder=10):
    text_registry.append({
        'text': text.replace('\n', ' '),
        'fontsize': fontsize,
        'color': color,
        'fontweight': fontweight
    })
    return ax.text(x, y, text, fontsize=fontsize, color=color, fontweight=fontweight, ha=ha, va=va, zorder=zorder)

def draw_box(ax, x, y, w, h, bg_color, rx=1.5):
    fancy = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rx}",
        facecolor=bg_color,
        edgecolor='none',
        linewidth=0,
        zorder=3
    )
    ax.add_patch(fancy)

def create_figure():
    fig, ax = plt.subplots(figsize=(7.16, 2.7), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Solid Color Palette (as requested in specifications)
    COLOR_DARK_BLUE   = '#1F4E79' # Blok 1: In-silico dataset
    COLOR_MED_BLUE    = '#2E75B6' # Blok 2: Data split
    COLOR_LIGHT_BLUE  = '#9DC3E6' # Blok 3: Classifiers
    COLOR_GREEN       = '#548235' # Blok 4: Calibration
    COLOR_ORANGE      = '#ED7D31' # Blok 5: Prediction set
    COLOR_RED         = '#C00000' # Blok 6: Clinical action
    COLOR_ARROW       = '#404040' # Dark gray arrows
    COLOR_TEXT_WHITE  = '#FFFFFF'
    COLOR_TEXT_BLACK  = '#000000'

    # ----------------------------------------------------
    # BLOK 1: In-silico dataset (Dark Blue #1F4E79)
    # ----------------------------------------------------
    b1_x, b1_y, b1_w, b1_h = 1.5, 15.0, 15.0, 70.0
    draw_box(ax, b1_x, b1_y, b1_w, b1_h, COLOR_DARK_BLUE, rx=2.0)
    
    add_text(ax, b1_x + b1_w/2, b1_y + 58, "In-silico Dataset", 8.5, COLOR_TEXT_WHITE, fontweight='bold')
    
    b1_items = [
        "- 28 CiPA drugs",
        "- 11 biomarkers",
        "- 2000 simulated\n  cells each",
        "- 3 patch-clamp\n  platforms"
    ]
    sub_y = b1_y + 45
    for item in b1_items:
        add_text(ax, b1_x + 1.2, sub_y, item, 7.0, COLOR_TEXT_WHITE, ha='left', va='top')
        sub_y -= 10.5

    # ----------------------------------------------------
    # BLOK 2: Split Data (Medium Blue #2E75B6, 2 stacked boxes)
    # ----------------------------------------------------
    b2_w = 14.0
    b2_top_x, b2_top_y, b2_top_h = 20.0, 53.0, 32.0
    draw_box(ax, b2_top_x, b2_top_y, b2_w, b2_top_h, COLOR_MED_BLUE, rx=2.0)
    add_text(ax, b2_top_x + b2_w/2, b2_top_y + 20, "12 Training\nDrugs", 8.0, COLOR_TEXT_WHITE, fontweight='bold')
    add_text(ax, b2_top_x + b2_w/2, b2_top_y + 7, "(calibration & training)", 7.0, COLOR_TEXT_WHITE)

    b2_bot_x, b2_bot_y, b2_bot_h = 20.0, 15.0, 32.0
    draw_box(ax, b2_bot_x, b2_bot_y, b2_w, b2_bot_h, COLOR_MED_BLUE, rx=2.0)
    add_text(ax, b2_bot_x + b2_w/2, b2_bot_y + 20, "16 Testing\nDrugs", 8.0, COLOR_TEXT_WHITE, fontweight='bold')
    add_text(ax, b2_bot_x + b2_w/2, b2_bot_y + 7, "(evaluation)", 7.0, COLOR_TEXT_WHITE)

    # ----------------------------------------------------
    # BLOK 3: Classifiers (Light Blue #9DC3E6, 4 vertically aligned boxes)
    # Group header above block
    # ----------------------------------------------------
    b3_x, b3_w = 38.0, 19.0
    
    add_text(ax, b3_x + b3_w/2, 92.0, "Grouped CV", 8.5, COLOR_TEXT_BLACK, fontweight='bold')
    add_text(ax, b3_x + b3_w/2, 85.5, "(StratifiedGroupKFold, k=4, by drug)", 7.0, COLOR_TEXT_BLACK)

    clf_boxes = [
        ("Stacking (RF + XGB)", 68.0, 12.0),
        ("Random Forest", 53.0, 12.0),
        ("Logistic Regression", 38.0, 12.0),
        ("Logistic Regression (qNet)", 23.0, 12.0)
    ]
    
    for c_title, c_y, c_h in clf_boxes:
        draw_box(ax, b3_x, c_y, b3_w, c_h, COLOR_LIGHT_BLUE, rx=1.5)
        # Light background -> black text for high contrast
        add_text(ax, b3_x + b3_w/2, c_y + c_h/2, c_title, 7.5, COLOR_TEXT_BLACK, fontweight='bold')

    # ----------------------------------------------------
    # BLOK 4: Cross-conformal calibration (Green #548235)
    # ----------------------------------------------------
    b4_x, b4_y, b4_w, b4_h = 61.0, 15.0, 14.5, 70.0
    draw_box(ax, b4_x, b4_y, b4_w, b4_h, COLOR_GREEN, rx=2.0)
    
    add_text(ax, b4_x + b4_w/2, b4_y + 53, "Cross-Conformal\nCalibration", 8.5, COLOR_TEXT_WHITE, fontweight='bold')
    add_text(ax, b4_x + b4_w/2, b4_y + 30, "Leave-one-drug-out\nover 12 drugs", 7.0, COLOR_TEXT_WHITE)
    add_text(ax, b4_x + b4_w/2, b4_y + 13, "q-hat at rank 11", 7.0, COLOR_TEXT_WHITE, fontweight='bold')

    # ----------------------------------------------------
    # BLOK 5: Prediction set (Orange #ED7D31)
    # ----------------------------------------------------
    b5_x, b5_y, b5_w, b5_h = 78.5, 25.0, 9.5, 50.0
    draw_box(ax, b5_x, b5_y, b5_w, b5_h, COLOR_ORANGE, rx=2.0)
    
    add_text(ax, b5_x + b5_w/2, b5_y + 35, "Prediction\nSet", 8.5, COLOR_TEXT_WHITE, fontweight='bold')
    add_text(ax, b5_x + b5_w/2, b5_y + 15, "{\nhigh,\nintermediate,\nlow\n}", 7.0, COLOR_TEXT_WHITE, fontweight='bold')

    # ----------------------------------------------------
    # BLOK 6: Clinical action (Brick Red #C00000)
    # ----------------------------------------------------
    b6_x, b6_y, b6_w, b6_h = 91.0, 25.0, 8.2, 50.0
    draw_box(ax, b6_x, b6_y, b6_w, b6_h, COLOR_RED, rx=2.0)
    
    add_text(ax, b6_x + b6_w/2, b6_y + 35, "Clinical\nAction", 8.5, COLOR_TEXT_WHITE, fontweight='bold')
    add_text(ax, b6_x + b6_w/2, b6_y + 14, "most severe\nclass in set", 7.0, COLOR_TEXT_WHITE)

    # ----------------------------------------------------
    # ARROWS (Thick dark gray arrows #404040)
    # ----------------------------------------------------
    arrow_style = dict(arrowstyle="-|>", color=COLOR_ARROW, lw=2.0, mutation_scale=12, zorder=5)
    
    # 1. Blok 1 -> Blok 2 (Split)
    ax.annotate("", xy=(20.0, 69.0), xytext=(16.5, 60.0), arrowprops=arrow_style)
    ax.annotate("", xy=(20.0, 31.0), xytext=(16.5, 40.0), arrowprops=arrow_style)
    
    # 2. Blok 2 (12 Training) -> Blok 3 (Classifiers)
    ax.annotate("", xy=(38.0, 69.0), xytext=(34.0, 69.0), arrowprops=arrow_style)
    
    # 3. Blok 2 (16 Testing) -> Blok 3 (Classifiers bottom)
    ax.annotate("", xy=(38.0, 31.0), xytext=(34.0, 31.0), arrowprops=arrow_style)
    
    # 4. Blok 2 (12 Training) -> Blok 4 (Calibration)
    curved_arrow_style = dict(
        arrowstyle="-|>", color=COLOR_ARROW, lw=2.0, mutation_scale=12,
        connectionstyle="arc3,rad=-0.28", linestyle="--", zorder=6
    )
    ax.annotate("", xy=(68.0, 85.0), xytext=(27.0, 85.0), arrowprops=curved_arrow_style)
    add_text(ax, 47.5, 96.0, "calibration data", 7.0, COLOR_ARROW, fontweight='bold')

    # 5. Blok 3 -> Blok 4 (Classifiers to Calibration)
    ax.annotate("", xy=(61.0, 50.0), xytext=(57.0, 50.0), arrowprops=arrow_style)
    
    # 6. Blok 4 -> Blok 5 (Calibration to Prediction Set)
    ax.annotate("", xy=(78.5, 50.0), xytext=(75.5, 50.0), arrowprops=arrow_style)
    
    # 7. Blok 5 -> Blok 6 (Prediction Set to Clinical Action)
    ax.annotate("", xy=(91.0, 50.0), xytext=(88.0, 50.0), arrowprops=arrow_style)

    plt.tight_layout(pad=0.1)
    
    pdf_path = os.path.join(RESULTS_DIR, 'fig1_pipeline_color.pdf')
    png_path = os.path.join(RESULTS_DIR, 'fig1_pipeline_color.png')
    
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(png_path, format='png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    
    return pdf_path, png_path

if __name__ == '__main__':
    pdf_p, png_p = create_figure()
    
    # Verification measurements
    pdf_kb = os.path.getsize(pdf_p) / 1024.0
    png_kb = os.path.getsize(png_p) / 1024.0
    
    with open(pdf_p, 'rb') as f:
        content = f.read()
        
    has_raster_image = (b'/Subtype /Image' in content) or (b'/Subtype/Image' in content)
    
    mediabox = re.search(r'/MediaBox\s*\[\s*([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s*\]', content.decode('latin1'))
    if mediabox:
        x0, y0, x1, y1 = [float(v) for v in mediabox.groups()]
        w_in = (x1 - x0) / 72.0
        h_in = (y1 - y0) / 72.0
    else:
        w_in, h_in = 7.16, 2.7
        
    print("==========================================================")
    print("  FIGURE 1 PIPELINE VERIFICATION RESULTS")
    print("==========================================================")
    print(f"PDF Dimensions : {w_in:.3f} inches x {h_in:.3f} inches (Target: 7.16 x <=2.8 in)")
    print(f"PDF Size       : {pdf_kb:.1f} KB")
    print(f"PNG Size       : {png_kb:.1f} KB (200 dpi)")
    print(f"Raster Image   : {'YES (FAIL)' if has_raster_image else 'NO (PASS - Pure Vector)'}")
    print("----------------------------------------------------------")
    print("  FIGURE TEXT INVENTORY & FONT SIZES")
    print("----------------------------------------------------------")
    for idx, t in enumerate(text_registry, 1):
        print(f"{idx:2d}. Font: {t['fontsize']:4.1f} pt | Weight: {t['fontweight']:6s} | Text: '{t['text']}'")
    print("==========================================================")
