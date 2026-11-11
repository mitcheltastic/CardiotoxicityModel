import os

base_dir = r"d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah"
trainai_dir = os.path.join(base_dir, "TrainAI")

dirs_to_check = {
    'docs': os.path.join(base_dir, "docs"),
    'results': os.path.join(trainai_dir, "results"),
    'models': os.path.join(trainai_dir, "models"),
    'logs': os.path.join(trainai_dir, "logs"),
    'src': os.path.join(trainai_dir, "src"),
}

all_files = {}

for label, dpath in dirs_to_check.items():
    if os.path.exists(dpath):
        files = os.listdir(dpath)
        all_files[label] = [f for f in files if os.path.isfile(os.path.join(dpath, f))]
    else:
        all_files[label] = []

print("=== VERIFIED FILES ON DISK ===")
for label, files in all_files.items():
    print(f"\n[{label.upper()}] ({len(files)} files)")
    for f in sorted(files):
        p = os.path.join(dirs_to_check[label], f)
        print(f"  - {f} (Exists: {os.path.exists(p)})")
