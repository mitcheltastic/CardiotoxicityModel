> [WARNING] STALE - OLD DATA. All numbers in this file were computed on the pre-`_rev` datasets and are
> SUPERSEDED. Do not cite, quote, or reuse any figure from this document in the ICCEREC 2026
> paper. Authoritative numbers come from the `_rev` re-run only.

> ⚠️ STALE — OLD DATA. All numbers in this file were computed on the pre-`_rev` datasets and are
> SUPERSEDED. Do not cite, quote, or reuse any figure from this document in the ICCEREC 2026
> paper. Authoritative numbers come from the `_rev` re-run only.

# context.md — Python Training Pipeline Knowledge Dump

> Complete consolidation of what's known about Mitchel's Python ML/DL training
> pipeline as of this session. Attach this when starting a new chat for model
> coding work on the application prototype.

---

## 1. Project Identity

- **Project**: HCiM TA (Capstone Final Project) — in silico cardiotoxicity (TdP) risk
  prediction using CiPA biomarkers
- **Primary direction (per sempro)**: 3-tier consensus (voting) is the chosen approach;
  3-tier stacking, binary stacking, and binary consensus are baselines
- **Companion paper**: iBioMed 2026 submitted (paper #1571297362), awaiting verdict
- **Language**: English

---

## 2. Dataset Specification

### Drug partition (12/16 split per Fuadah et al. 2024)

- **Training set**: 12 drugs (4 per risk class: high / intermediate / low)
- **Test set**: 16 drugs (4 high-risk, 7 intermediate-risk, 5 low-risk)
- **Disjoint**: zero drug overlap between train and test
- **Samples per drug**: 2,000 (from virtual-population simulation)
- **Total**: 24,000 training samples, 32,000 testing samples

### Files (Google Drive, mounted at `/content/drive/My Drive/TA_Datasets/`)

| File | Size | Description |
|---|---|---|
| `chantest_training.csv` | 24,000 rows | Training data, 12 drugs |
| `chantest_testing.csv` | 32,000 rows | Test data, 16 drugs |
| `Revision/*.csv` | 9 files | Alt virtual-population datasets (3 cell models × 3 cell types) |
| `Revision/validation_results/master_results.csv` | — | Cross-dataset macro-F1 results |

### The 9 alternative datasets (in `Revision/`)

Cross-product of:
- AP cell models: ORd static, ORd dynamic, ToR-ORd
- Cell types: endocardial (endo), epicardial (epi), mid-myocardial (m)

Cross-dataset macro-F1 results (from iBioMed paper):
- ORd static: 0.36 / 0.33 / 0.30 (endo/epi/m)
- ORd dynamic: 0.76 / 0.77 / 0.74
- ToR-ORd: 0.59 / 0.37 / 0.20

Key finding: dynamic-hERG configurations cluster at 0.74–0.77; static-hERG configurations
collapse toward 3-class chance.

---

## 3. Features — 11 Biomarkers

### Canonical biomarker list

```python
BIOMARKERS = [
    'qNet', 'dvdtmax', 'vmax', 'vrest',
    'APD50', 'APD90', 'max_dv',
    'camax', 'carest', 'CaTD50', 'CaTD90'
]
```

### Name mapping (ChanTest columns ↔ scientific notation)

| CSV column name | Scientific name |
|---|---|
| `qNet` | qNet |
| `dvdtmax` | $dV_m/dt_{\max}$ |
| `vmax` | $V_{m,\text{peak}}$ |
| `vrest` | $V_{m,\text{rest}}$ |
| `APD50` | APD$_{50}$ |
| `APD90` | APD$_{90}$ |
| `max_dv` | $dV_m/dt_{\text{repol}}$ |
| `camax` | Ca$_{\text{peak}}$ |
| `carest` | Ca$_{\text{rest}}$ |
| `CaTD50` | CaD$_{50}$ |
| `CaTD90` | CaD$_{90}$ |

**Important**: ChanTest column names differ from Metaheart aliases. Don't mix them.

### Biomarker correlation structure

From the iBioMed paper analysis (verified):
- `r = 0.995` between APD50 and APD90
- `r = 0.993` between APD90 and `max_dv`
- `r = 0.992` between APD50 and `max_dv`

These three carry near-identical information. Gini vs SHAP rankings disagree
on these three because of the collinearity, not because of true contribution difference.
Supports a seven-biomarker subset reduction.

---

## 4. Labels

### Class encoding

```python
{'high': 0, 'intermediate': 1, 'low': 2}
```

### Binary (cautionary) scenario

Merge high + intermediate → "cautionary" class; low stays as "low".

---

## 5. Preprocessing Pipeline

```python
from sklearn.preprocessing import StandardScaler

# Standard 3-step preprocessing
# Step 1: select 11 biomarkers, drop metadata
X_train = df_train[BIOMARKERS]
X_test = df_test[BIOMARKERS]

# Step 2: integer-encode labels
label_map = {'high': 0, 'intermediate': 1, 'low': 2}
y_train = df_train['label'].map(label_map)
y_test = df_test['label'].map(label_map)

# Step 3: standardize using TRAIN-set statistics only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Critical**: Always `fit_transform` on train, `transform` on test. Never refit on test.

---

## 6. Base Learner Architectures (from iBioMed paper)

### Seven base learners across two paradigms

**ML side (4 learners):**

| Learner | Hyperparameters (selected) |
|---|---|
| Random Forest | criterion=entropy, max_depth=None, n_estimators=400 |
| XGBoost | learning_rate=0.001, max_depth=10, n_estimators=500 |
| Logistic Regression | C=0.01, penalty=L2, solver=lbfgs |
| SVM (RBF) | C=10.0, gamma=auto |

**DL side (3 learners):**

| Learner | Architecture | Selected hyperparameters |
|---|---|---|
| ANN1 | 1 hidden layer, ReLU, softmax output | nodes=32, lr=0.0001 |
| ANN2 | 2 hidden layers, ReLU, softmax output | nodes=64, lr=0.01 |
| ANN3 | 3 hidden layers, ReLU, softmax output | nodes=128, lr=0.01 |

### ANN common architecture

- Fully-connected feedforward
- ReLU hidden activations
- Softmax output layer
- Adam optimizer
- Sparse categorical cross-entropy loss
- Identical node count across hidden layers within each network
- `random_state=42` for reproducibility

### Hyperparameter search

- Method: GridSearchCV with 3-fold stratified cross-validation
- Optimization criterion: macro-averaged F1
- Performed independently per base learner before ensemble construction

---

## 7. Ensemble Configurations

### Three paradigms tested in iBioMed paper

#### Extended stacking (7 base learners at Level-1)
```python
from sklearn.ensemble import StackingClassifier
StackingClassifier(
    estimators=[...all 7 base learners...],
    final_estimator=<meta>,
    cv=5,
    stack_method='predict_proba',
    passthrough=False  # meta sees ONLY base outputs, not raw features
)
```
4 meta-classifier variants tested: LR, SVM, RF, XGBoost

#### Paper-faithful stacking (3 ANN base learners only)
Same structure as extended stacking, but only ANN1, ANN2, ANN3 at Level-1.
Replicates Fuadah et al. 2024 architecture exactly.
4 meta-classifier variants tested: LR, SVM, RF, XGBoost

#### Voting ensembles (7 base learners)
```python
from sklearn.ensemble import VotingClassifier
# Hard voting: majority class
VotingClassifier(estimators=[...], voting='hard')
# Soft voting: average predicted probabilities
VotingClassifier(estimators=[...], voting='soft')
```

### Final leaderboard (3-class scenario, iBioMed Table II)

| Configuration | Macro-F1 | Inter. F1 | Inter. Recall |
|---|---|---|---|
| Hard Voting (7 base) | **0.826** | 0.768 | 0.677 |
| Soft Voting (7 base) | 0.824 | 0.767 | 0.683 |
| Paper-faithful Stack (LR) | 0.820 | 0.758 | 0.646 |
| Paper-faithful Stack (SVM) | 0.818 | 0.752 | 0.636 |
| Paper-faithful Stack (RF) | 0.816 | 0.751 | 0.631 |
| Paper-faithful Stack (XGB) | 0.807 | 0.759 | 0.695 |
| Extended Stack (LR meta) | 0.802 | 0.750 | 0.672 |
| Extended Stack (RF meta) | 0.797 | 0.749 | 0.678 |
| Extended Stack (SVM meta) | 0.788 | 0.740 | 0.676 |
| Extended Stack (XGB meta) | 0.788 | 0.740 | 0.682 |

### Final leaderboard (2-class binary scenario, iBioMed Table III)

| Configuration | Macro-F1 |
|---|---|
| Hard Voting (7 base) | **0.902** |
| Extended Stack (XGB meta) | 0.901 |
| Soft Voting (7 base) | 0.901 |
| Paper-faithful Stack (LR) | 0.901 |
| Extended Stack (LR meta) | 0.900 |
| Paper-faithful Stack (SVM) | 0.899 |
| Paper-faithful Stack (RF) | 0.898 |
| Extended Stack (RF meta) | 0.896 |
| Extended Stack (SVM meta) | 0.888 |
| Paper-faithful Stack (XGB) | 0.806 |

### Single Random Forest baseline (in text only, not in leaderboard tables)

- macro-F1 = 0.822 on full test set
- CV macro-F1 = 0.847
- Selected as production model for CiPA bootstrap (computational efficiency,
  within 0.5pp of leading voting ensembles)

---

## 8. Evaluation Protocols

### Protocol A — Standard test-set evaluation
- Fit on training set (24,000 samples, 12 drugs)
- Evaluate on test set (32,000 samples, 16 drugs)
- Report macro-F1, per-class F1, per-class recall

### Protocol B — CiPA bootstrap evaluation (per Li et al. 2019)

```python
# 10,000 iterations
for i in range(10000):
    # Each iteration: 1 sample drawn with replacement per test drug
    # Yields 16-sample evaluation set per iteration
    sample = []
    for drug in test_drugs_16:
        s = drug_samples.sample(n=1, replace=True)
        sample.append(s)
    # Compute per-class AUC (one-vs-rest), LR+, LR-, accuracy
    # ...
# Aggregate: mean + 95% CI (empirical 2.5th and 97.5th percentiles)
```

#### Bootstrap exclusion handling (important for paper correctness)

Under Mitchel's sampling scheme (1 sample × 16 drugs per iteration):
- All 3 classes represented in every bootstrap sample → AUC defined 100% of iterations
- LR− defined 100% of iterations for all classes
- LR+ defined 100% for HIGH; 37.2% for INTERMEDIATE; 44.9% for LOW
- Excluded iterations have **perfect specificity** (zero FP), making LR+ infinite
- Reported LR+ values for INTERMEDIATE and LOW are **conservative lower bounds**

### Final CiPA bootstrap results (Random Forest, 10,000 iterations)

| Class | AUC | LR+ | LR− |
|---|---|---|---|
| HIGH | 0.88 (0.85–0.94) Good | 6.09 (4.00–12.0) Good | 0.016 (0.00–0.30) **Excellent** |
| INTERMEDIATE | 0.75 (0.68–0.87) Acceptable | 5.52 (2.57–6.43) Good | 0.37 (0.16–0.57) Acceptable |
| LOW | 0.99 (0.95–1.00) **Excellent** | 9.61 (4.40–11.0) Good | 0.09 (0.00–0.40) **Excellent** |

### CiPA tier criteria thresholds

| Metric | Excellent | Good | Acceptable |
|---|---|---|---|
| AUC | ≥0.90 | ≥0.80 | ≥0.70 |
| LR+ | ≥10.0 | ≥5.0 | ≥2.0 |
| LR− | ≤0.10 | ≤0.20 | ≤0.50 |

---

## 9. Probability Calibration & Threshold Optimization (Ablation)

Tried but UNDERPERFORMED the uncalibrated argmax — keep as ablation, not primary:

```python
from sklearn.calibration import CalibratedClassifierCV

# Isotonic regression calibration on drug-level holdout
# Drug-level: 1 drug per risk class held out = 6,000 samples total
# (instead of sample-level random split, to preserve across-drug generalization)
calibrated_rf = CalibratedClassifierCV(rf, method='isotonic', cv='prefit')
calibrated_rf.fit(X_holdout, y_holdout)

# Threshold search: τ ∈ [0.15, 0.60] at 0.01 resolution
# Objective: maximize intermediate-class F1
# Constraint: macro-F1 degradation ≤ 1 percentage point
```

### Result of calibration ablation
- Reduced training set: 12 → 9 drugs (24,000 → 18,000 samples)
- macro-F1 dropped from 0.822 → 0.767
- Conclusion: uncalibrated argmax is near-optimal under present class distribution
- BUT: macro-F1 drop also reflects training set reduction, not just calibration

---

## 10. PCA Drug-Level Analysis (for error analysis)

### PCA setup
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
PC_train = pca.fit_transform(X_train_scaled)
PC_test = pca.transform(X_test_scaled)
# Variance: PC1 = 53.3%, PC2 = 26.2%
```

### Per-drug centroid calculation
```python
def per_drug_centroid(PC, df):
    rows = []
    for drug in df['drug_name'].unique():
        m = df['drug_name'].values == drug
        rows.append({
            'drug': drug.title(),
            'label': df.loc[m, 'label'].iloc[0],
            'pc1': float(PC[m, 0].mean()),
            'pc2': float(PC[m, 1].mean()),
        })
    return pd.DataFrame(rows)
```

### Three problematic drugs (account for most intermediate-class misclassifications)

| Drug | True label | PC1 coordinate | Displaced to |
|---|---|---|---|
| Domperidone | intermediate | +0.97 | high-risk cluster |
| Pimozide | intermediate | +0.59 | high-risk cluster |
| Risperidone | intermediate | −1.80 | low-risk cluster |

These are LABEL-FEATURE INCONSISTENCY cases — clinical labels and mechanistic
biomarker profiles diverge. Not a model capacity issue.

---

## 11. Matplotlib Configuration (critical for paper-quality figures)

### Always use TrueType fonts (avoid Type 3 — EDAS will warn)

```python
import matplotlib.pyplot as plt

# Force TrueType fonts (Type 42)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Use serif fonts to match LaTeX rendering
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
```

### Color scheme for risk classes

```python
CLR = {
    'high': '#E63946',           # red
    'intermediate': '#F4A261',   # orange
    'low': '#06A77D',            # green
}
```

### Single-column IEEE figure size

```python
fig, ax = plt.subplots(figsize=(3.9, 4.4), dpi=300)
# OR for heatmap
fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=300)
```

---

## 12. The Open Methodological Issues

### Issue 1: AUC gap (HIGH PRIORITY)
- Fuadah et al. 2024 reports intermediate AUC = 0.94
- Mitchel's implementation gets intermediate AUC = 0.75
- 0.19 gap, unresolved
- Top hypothesis: preprocessing or bootstrap protocol difference

### Issue 2: ML/DL paradigm mixing (sempro feedback)
- Current 7-learner ensemble mixes ANN (DL) with RF/XGBoost/LR/SVM (ML)
- Examiners want either pure ML ensemble OR pure DL ensemble, not mixed
- Plan: run two clean ablations (Goal 2A pure ML, Goal 2B pure DL)

---

## 13. Recommended Libraries for Future Work

```bash
# Pure ML
pip install scikit-learn xgboost lightgbm catboost

# Pure DL (for Goal 2B and the application prototype)
pip install tensorflow keras scikeras           # ANN + sklearn ensemble compatibility
pip install torch                                # if using PyTorch path
pip install pytorch-tabnet rtdl                  # TabNet, FT-Transformer (optional)

# Standard data science
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### scikeras wrapper (needed if using Keras ANNs in sklearn ensembles)

```python
from scikeras.wrappers import KerasClassifier

def build_ann(n_hidden_layers=2, nodes=64, lr=0.001):
    model = keras.Sequential()
    for _ in range(n_hidden_layers):
        model.add(layers.Dense(nodes, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))  # 3 classes
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

ann2_wrapped = KerasClassifier(
    model=build_ann,
    model__n_hidden_layers=2,
    model__nodes=64,
    model__lr=0.01,
    epochs=50,
    batch_size=32,
    verbose=0,
    random_state=42
)
# Now ann2_wrapped behaves like a sklearn classifier — can go in StackingClassifier or VotingClassifier
```

---

## 14. Reproducibility Settings

```python
import numpy as np
import random
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# For sklearn: pass random_state=SEED to every estimator
```

---

## 15. Key References (for methodology debugging)

| Citation | Why it matters |
|---|---|
| Fuadah et al. 2024 (CPT Pharmacometrics, 13(12):2159–2170) | THE paper to debug against — claims 0.94 intermediate AUC |
| Fuadah et al. 2023 (Frontiers Physiol., 14:1266084) | Earlier RF paper, same group |
| Pramudito et al. 2024 (Sci. Reports, 14:24045) | SHAP biomarker importance |
| Li et al. 2019 (Clin. Pharm. Therap., 105(2):466–475) | CiPA bootstrap protocol |
| Dutta et al. 2017 (Frontiers Physiol., 8:616) | ORd-CiPAv1.0 cell model |

---

## 16. Application Prototype — Notes for Next Session

Since you mentioned coding for an **application prototype** in the next session, here
are the prototype-relevant items to think about:

### What "production model" means
- The Random Forest was selected as the production model in the iBioMed paper
- Justification: macro-F1 within 0.5pp of leading voting ensembles, much faster inference
- For an application prototype, this is the model to serve at the API/UI

### Inference pipeline (what the app needs)

```python
# Pseudo-flow for app prototype
def predict_risk(biomarker_values: dict) -> dict:
    # 1. Convert input dict to array in correct column order
    x = np.array([biomarker_values[b] for b in BIOMARKERS]).reshape(1, -1)
    # 2. Standardize using FITTED scaler from training time
    x_scaled = scaler.transform(x)
    # 3. Predict class + probabilities
    pred_class = production_model.predict(x_scaled)[0]
    pred_proba = production_model.predict_proba(x_scaled)[0]
    # 4. Map back to labels
    label_map_inv = {0: 'high', 1: 'intermediate', 2: 'low'}
    return {
        'predicted_class': label_map_inv[pred_class],
        'probabilities': {
            'high': float(pred_proba[0]),
            'intermediate': float(pred_proba[1]),
            'low': float(pred_proba[2]),
        }
    }
```

### Model persistence

```python
import joblib

# Save model + scaler together (BOTH needed for inference)
joblib.dump(production_model, 'production_rf.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Load for inference
production_model = joblib.load('production_rf.joblib')
scaler = joblib.load('scaler.joblib')
```

**Critical**: Always save the scaler alongside the model. Without it, raw biomarker
inputs from a user/app can't be standardized to match the training distribution.

---

## 17. What to Tell the Next Chat

Start the next chat with:

```
Hi, I'm Mitchel (HCiM account). Attaching context.md with all the Python
training pipeline knowledge. Today I want to start coding [X] for the
application prototype.
```

Where [X] could be:
- The training script (cleaned up, modular)
- The inference pipeline (with saved scaler + model)
- A Flask/FastAPI backend serving the model
- A Streamlit UI for biomarker input
- A dockerized version of the above
- Whatever your app prototype needs

The next assistant will read this file, confirm the goal, ask for specific files
(notebook, current code, etc.), and dive in.

---

**End of context.md.** Save this with guide.md as your TA handoff toolkit.

🫡
