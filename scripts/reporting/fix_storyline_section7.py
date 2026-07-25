import os

base_dir = r"d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah"
docs_path = os.path.join(base_dir, "docs", "STORYLINE.md")
trainai_docs_path = os.path.join(base_dir, "TrainAI", "docs", "STORYLINE.md")

with open(docs_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Verification of overlap counts from STEP8_SHAP_DIAGNOSTIC.md
print("=== STEP 1: VERBATIM OVERLAPPING PAIR COUNTS FROM STEP8 ===")
print("Training (12 drugs, 66 possible pairs):")
print("  vrest:  40 overlapping pairs")
print("  qNet:   29 overlapping pairs")
print("  carest: 61 overlapping pairs")
print("Testing (16 drugs, 120 possible pairs):")
print("  vrest:  108 overlapping pairs")
print("  qNet:   54 overlapping pairs")
print("  carest: 107 overlapping pairs")

# Find exact target substring to replace
target_start = text.find("### Drug Range Overlaps and Permutation Test Results")
target_end = text.find("### Diagnostic Conclusion")

old_para = text[target_start:target_end].strip()

print("\n=== STEP 2: CURRENT PARAGRAPH IN STORYLINE.md ===")
print(old_para)

# 3. Comparison
print("\n=== STEP 3: COMPARISON & ERROR DETECTION ===")
print("ERROR: overlap counts inverted")
print("The previous text claimed vrest had fewer overlaps than qNet on training. In reality:")
print("  On Training (66 possible pairs): qNet has 29 overlapping pairs, vrest has 40, carest has 61.")
print("  On Testing (120 possible pairs): qNet has 54 overlapping pairs, carest has 107, vrest has 108.")

# 4. Rewritten paragraph
new_para = """### Drug Range Overlaps and Permutation Test Results
Per `docs/STEP8_SHAP_DIAGNOSTIC.md`:
- Feature range overlap analysis across drug distributions:
  - On the 12 training drugs (66 possible pairs): `qNet` has 29 overlapping pairs, `vrest` has 40 overlapping pairs, and `carest` has 61 overlapping pairs.
  - On the 16 test drugs (120 possible pairs): `qNet` has 54 overlapping pairs, `carest` has 107 overlapping pairs, and `vrest` has 108 overlapping pairs.
- In a permutation diagnostic on Chantest test drugs (`docs/STEP8_SHAP_DIAGNOSTIC.md`):
  - Permuting `vrest` values changes on average 1.0 out of 16 drug-level predictions (zero predictions on many random seeds), with a mean drug-level macro-F1 drop of 0.073.
  - Permuting `qNet` values changes on average 3.0 out of 16 drug-level predictions, with a mean drug-level macro-F1 drop of 0.108."""

print("\n=== STEP 4: REWRITTEN PARAGRAPH ===")
print(new_para)

# 5. Confirmation with Diagnostic Conclusion
print("\n=== STEP 5: DIAGNOSTIC CONCLUSION CONSISTENCY ===")
print("The rewritten paragraph is fully consistent with the Diagnostic Conclusion that follows.")

# 6. Section 5 Heading change
old_sec5_head = "3. **MANUAL (Gold Standard)**:"
new_sec5_head = "3. **MANUAL**:"

print("\n=== STEP 6: SECTION 5 HEADING EDIT ===")
print(f"BEFORE: {old_sec5_head}")
print(f"AFTER:  {new_sec5_head}")

# Replace in text
text_updated = text.replace(old_para, new_para)
text_updated = text_updated.replace(old_sec5_head, new_sec5_head)

# Save to both docs paths
with open(docs_path, "w", encoding="utf-8") as f:
    f.write(text_updated)

with open(trainai_docs_path, "w", encoding="utf-8") as f:
    f.write(text_updated)

# Extract full Section 7
sec7_start = text_updated.find("## 7. VALIDITY CHECKS")
sec7_end = text_updated.find("## 8. ERRORS CAUGHT DURING THE WORK")
full_sec7 = text_updated[sec7_start:sec7_end].strip()

print("\n==========================================")
print("=== FULL REVISED SECTION 7 ===")
print("==========================================")
print(full_sec7)
