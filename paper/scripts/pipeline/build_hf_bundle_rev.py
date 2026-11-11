import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

# Add workspace root to sys.path to import src.data_loader_rev
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.data_loader_rev import load_rev, CANONICAL_BIOMARKERS, LABEL_MAP

def build_bundle():
    print("=== Building Updated Hugging Face Model Bundle (_rev) ===")
    
    # 1. Load Chantest Training Dataset (_rev)
    print("Loading Chantest training data...")
    X_train, y_train, drugs_train, report = load_rev('chantest', 'training')
    
    print(f"Loaded {len(X_train)} training rows across {len(np.unique(drugs_train))} drugs.")
    print(f"Features ({len(CANONICAL_BIOMARKERS)}): {CANONICAL_BIOMARKERS}")
    
    # 2. Define Base Learners & Meta Learner
    rf_base = RandomForestClassifier(
        n_estimators=400,
        criterion='entropy',
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_base = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=10,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    
    # Logistic Regression meta-learner produces smooth calibrated probabilities
    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    
    # Grouped CV splits to prevent drug leakage during out-of-fold meta-feature generation
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=False)
    cv_splits = list(sgkf.split(X_train.values, y_train.values, groups=drugs_train))
    
    stacking_model = StackingClassifier(
        estimators=[('RF', rf_base), ('XGB', xgb_base)],
        final_estimator=meta_learner,
        cv=cv_splits,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    # 3. Fit Stacking Classifier
    print("Fitting StackingClassifier (RF + XGB -> LogisticRegression meta)...")
    stacking_model.fit(X_train.values, y_train.values)
    print("Model fitting complete!")
    
    # 4. Prepare Bundle Dictionary
    inverse_label_map = {v: k for k, v in LABEL_MAP.items()}
    
    bundle = {
        'model': stacking_model,
        'biomarkers': CANONICAL_BIOMARKERS,
        'class_map': LABEL_MAP,
        'inverse_class_map': inverse_label_map,
        'q_hat': 0.6275,  # 11th of 12 LODO scores
        'alpha': 0.154,   # 1 - 0.154 = 84.6% nominal coverage
        'dataset': 'Chantest_rev',
        'architecture': 'stacking base={RF,XGB} meta=LogisticRegression',
        'cv_scheme': 'StratifiedGroupKFold(n_splits=4, by drug)'
    }
    
    # 5. Export to hf_space/
    hf_dir = os.path.join(base_dir, 'hf_space')
    os.makedirs(hf_dir, exist_ok=True)
    
    bundle_path = os.path.join(hf_dir, 'deployable_chantest_3class.joblib')
    print(f"Saving model bundle to {bundle_path}...")
    joblib.dump(bundle, bundle_path)
    
    # 6. Generate and Export SHAP Background
    print("Generating SHAP background data (shap.kmeans, k=15)...")
    background = shap.kmeans(X_train.values, 15)
    bg_path = os.path.join(hf_dir, 'shap_background.joblib')
    print(f"Saving SHAP background to {bg_path}...")
    joblib.dump(background, bg_path)
    
    print("\n[SUCCESS] Bundle successfully built and exported to hf_space/!")

if __name__ == "__main__":
    build_bundle()
