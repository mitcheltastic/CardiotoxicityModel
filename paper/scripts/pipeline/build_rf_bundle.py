import os
import sys
import joblib
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.data_loader_rev import load_rev

def build_rf_bundle():
    dep_path = os.path.join(base_dir, "hf_space", "deployable_chantest_3class.joblib")
    out_path = os.path.join(base_dir, "hf_space", "deployable_chantest_rf.joblib")

    print(f"Loading base bundle from: {dep_path}")
    b = joblib.load(dep_path)

    # Confirm model type before extraction
    stacking_model = b["model"]
    print(f"Existing bundle model type: {type(stacking_model)}")
    rf_estimator = stacking_model.estimators_[0]
    print(f"Extracted estimator type: {type(rf_estimator)}")
    print(f"Number of trees (n_estimators): {rf_estimator.n_estimators} (len={len(rf_estimator.estimators_)})")

    # Replace model and update conformal calibration parameters
    b["model"] = rf_estimator
    b["architecture"] = "RandomForestClassifier (400 trees, entropy, random_state=42) + split-conformal (LODO, rank 11)"
    b["q_hat"] = 0.533987        # Rank-11 LODO, calibration drug sotalol
    b["alpha"] = 2 / 13          # nominal coverage 84.6%

    print(f"Saving RF bundle to: {out_path}...")
    joblib.dump(b, out_path)
    print("Bundle saved successfully.")

    # Verification against research RF
    print("\n--- Verifying Bundle ---")
    b_new = joblib.load(out_path)
    model_new = b_new["model"]
    print(f"model.classes_: {model_new.classes_}")

    res_model_path = os.path.join(base_dir, "models", "chantest_stacking_model_rev.joblib")
    res_bundle = joblib.load(res_model_path)
    rf_res = res_bundle.named_estimators_['rf']

    X_test, y_test, drugs_test, report = load_rev('chantest', 'testing', data_dir=os.path.join(base_dir, 'data'))
    print(f"Loaded {len(X_test)} ChanTest test samples.")

    probs_new = model_new.predict_proba(X_test.values)
    probs_res = rf_res.predict_proba(X_test.values)

    max_diff = float(np.max(np.abs(probs_new - probs_res)))
    print(f"Max absolute difference vs research RF on ChanTest test set: {max_diff}")
    assert max_diff == 0.0, f"Max difference is {max_diff}, expected 0.0!"
    print("Verification PASSED: exactly 0.0 difference.")

if __name__ == "__main__":
    build_rf_bundle()
