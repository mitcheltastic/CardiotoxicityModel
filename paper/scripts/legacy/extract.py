import json
with open('plus_ultra.ipynb', 'r') as f:
    nb = json.load(f)
cells = nb['cells']
code_cells = [c['source'] for c in cells if c['cell_type'] == 'code']
for idx, cell in enumerate(code_cells):
    cell_str = ''.join(cell)
    if 'confusion_matrix_chantest.png' in cell_str or 'ensemble_family_comparison.png' in cell_str or 'bootstrap_macrof1_distributions.png' in cell_str or 'per_class_auc_comparison.png' in cell_str:
        print(f"--- CELL {idx} ---")
        print(cell_str)
