import os

base_dir = r"d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah"
docs_path = os.path.join(base_dir, "docs", "STORYLINE.md")
trainai_docs_path = os.path.join(base_dir, "TrainAI", "docs", "STORYLINE.md")

with open(docs_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Print current Diagnostic Conclusion
old_diag = """### Diagnostic Conclusion
This pattern is consistent with `vrest` containing per-drug systematic offsets rather than strong discriminative signal across new drugs. The precise origin of these systematic offsets is not established in this repository."""

print("=== 1. CURRENT DIAGNOSTIC CONCLUSION IN SECTION 7 ===")
print(old_diag)

new_diag = """### Diagnostic Conclusion
On the 16 held-out test drugs, vrest is the least separable of the features examined (108 of 120 drug pairs overlap, versus 54 of 120 for qNet), permuting it changes on average 1.0 of 16 drug-level predictions, and the XGBoost base learner does not split on it at all. Its stable rank-2 position by mean|SHAP| is therefore not supported by a corresponding effect on predictions. This is consistent with instability of the model-agnostic explainer on a feature whose coefficient of variation is on the order of 1e-4, rather than with discriminative signal. The interpretability discussion accordingly rests on qNet, which ranks first by mean|SHAP| and has the largest permutation effect."""

print("\n=== 2. NEW DIAGNOSTIC CONCLUSION ===")
print(new_diag)

# Replace Diagnostic Conclusion
text = text.replace(old_diag, new_diag)

# 3. Section 9 updates
old_sec9 = """## 9. OPEN QUESTIONS

The following technical questions remain open and require further investigation:
1. **Undocumented Normalization Formula**: The exact mathematical transform used to normalize raw biophysical outputs into `_rev` features is not recorded in this repository.
2. **Unexplained Performance Delta vs. Earlier Datasets**: The exact cause of performance differences between earlier unrevised datasets and `_rev` datasets is unestablished.
3. **Control Normalization Scope**: Whether each drug was normalized against its own vehicle control run or against a shared control run is not documented here. This bears directly on the origin of `vrest` systematic offsets.
4. **`vrest` Micro-Threshold Origin**: The exact mechanism causing Random Forest to create 1,533 micro-splits on `vrest` while XGBoost ignores it is unresolved.
5. **Formal Coverage Guarantee Class**: Whether the 12-refit jackknife calibration strictly satisfies split-conformal coverage guarantees or requires a cross-conformal theoretical framework is unestablished.
6. **Prospective Clinical Validation**: Because Rank 11 was selected using the 16 held-out test drugs, true prospective safety validation requires testing on a new, uninspected drug library."""

print("\n=== 3. OLD SECTION 9 ===")
print(old_sec9)

new_sec9 = """## 9. OPEN QUESTIONS

The following technical questions remain open and require further investigation:
1. **Undocumented Normalization Formula**: The exact mathematical transform used to normalize raw biophysical outputs into `_rev` features is not recorded in this repository.
2. **Unexplained Performance Delta vs. Earlier Datasets**: The exact cause of performance differences between earlier unrevised datasets and `_rev` datasets is unestablished.
3. **Control Normalization Scope**: Whether each drug was normalized against its own vehicle control run or against a shared control run is not documented here.
4. **`vrest` Split & SHAP Origin**: Why the Random Forest base learner splits on vrest 1,533 times while XGBoost splits on it zero times is unresolved, as is why KernelExplainer assigns it a stable rank-2 importance despite its minimal permutation effect.
5. **Formal Coverage Guarantee Class**: Whether the 12-refit jackknife calibration strictly satisfies split-conformal coverage guarantees or requires a cross-conformal theoretical framework is unestablished.
6. **Prospective Clinical Validation**: Because Rank 11 was selected using the 16 held-out test drugs, true prospective safety validation requires testing on a new, uninspected drug library."""

print("\n=== NEW SECTION 9 ===")
print(new_sec9)

text = text.replace(old_sec9, new_sec9)

# 4. Section 8 entry j addition
old_sec8_i = """i. **Coverage-Miss Drug List Conflict**: A conflict between `STEP6` and `PAPER_NUMBERS` regarding which drugs missed coverage on Chantest and Nanion was resolved by independent recomputation, confirming `STEP6`'s list: Chantest misses `['domperidone', 'loratadine', 'pimozide', 'tamoxifen']`; Nanion misses `['domperidone', 'metoprolol', 'pimozide', 'tamoxifen']`. Corrected in `docs/PAPER_NUMBERS.md`."""

new_sec8_ij = old_sec8_i + """\nj. **vrest separability misreported**: An intermediate summary stated vrest had zero overlapping drug ranges, implying per-drug separability. Direct recomputation from the source data (`docs/STEP10_VREST_OVERLAP_RECHECK.md`) established 40 overlapping pairs of 66 on the training split and 108 of 120 on the test split. The zero-overlap claim and the per-drug-offset interpretation derived from it are retracted."""

print("\n=== 4. SECTION 8 ENTRY J ADDITION ===")
print("BEFORE:")
print(old_sec8_i)
print("AFTER:")
print(new_sec8_ij)

text = text.replace(old_sec8_i, new_sec8_ij)

# Save to both docs locations
with open(docs_path, "w", encoding="utf-8") as f:
    f.write(text)

with open(trainai_docs_path, "w", encoding="utf-8") as f:
    f.write(text)

# 5. Check remaining text for terms: separability, offsets, fingerprint
terms_to_check = ['separab', 'offset', 'fingerprint']
locations_checked = []
for idx, line in enumerate(text.splitlines()):
    for term in terms_to_check:
        if term in line.lower():
            locations_checked.append((idx + 1, line))

print("\n=== 5. LOCATIONS CHECKED FOR RETRACTED CONCEPTS ===")
print(f"Total lines checked matching keywords ({terms_to_check}): {len(locations_checked)}")
for line_no, l_text in locations_checked:
    print(f"Line {line_no:4d}: {l_text}")

# 6. Extract full Sections 7, 8j, 9
sec7_idx = text.find("## 7. VALIDITY CHECKS")
sec8_idx = text.find("## 8. ERRORS CAUGHT DURING THE WORK")
sec9_idx = text.find("## 9. OPEN QUESTIONS")
sec10_idx = text.find("## 10. ARTIFACT INDEX")

full_sec7 = text[sec7_idx:sec8_idx].strip()
full_sec8_j = text[text.find("j. **vrest separability"):sec9_idx].strip()
full_sec9 = text[sec9_idx:sec10_idx].strip()

print("\n==========================================")
print("=== FULL CORRECTED SECTION 7 ===")
print("==========================================")
print(full_sec7)

print("\n==========================================")
print("=== FULL CORRECTED SECTION 8 ENTRY J ===")
print("==========================================")
print(full_sec8_j)

print("\n==========================================")
print("=== FULL CORRECTED SECTION 9 ===")
print("==========================================")
print(full_sec9)
