import sys

ALPHA = 0.154
Q_HAT = 0.6275
SEVERITY = {"low": 0, "intermediate": 1, "high": 2}
INV_SEVERITY = {0: "low", 1: "intermediate", 2: "high"}
CALIBRATION_NOTE = (
    "Conformal threshold calibrated at drug level on probability\n"
    "vectors averaged over 2000 replicate samples per drug (leave-one-drug-out over 12\n"
    "training drugs). This response is based on a single input row; the coverage\n"
    "guarantee does not strictly transfer to single-row input."
)

def conformal_from_proba(prob_dict):
    # prob_dict is like {'high': 0.1, 'intermediate': 0.2, 'low': 0.7}
    # class included iff p[c] >= 1 - Q_HAT = 0.3725
    threshold = 1.0 - Q_HAT
    
    included_classes = []
    for c, p in prob_dict.items():
        if p >= threshold:
            included_classes.append(c)
            
    # sort by severity descending
    included_classes.sort(key=lambda c: SEVERITY[c.lower()], reverse=True)
    
    set_size = len(included_classes)
    is_ambiguous = set_size > 1
    
    if set_size == 0:
        # Empty set fallback
        argmax_class = max(prob_dict, key=prob_dict.get)
        return {
            "prediction_set": [],
            "recommended_action": argmax_class.lower(),
            "set_size": 0,
            "is_ambiguous": True,
            "out_of_distribution": True,
            "warning": "No class met the conformal threshold. Result is not covered by the guarantee.",
            "alpha": ALPHA,
            "q_hat": Q_HAT,
            "calibration_note": CALIBRATION_NOTE
        }
    
    recommended_action = included_classes[0]
    
    return {
        "prediction_set": included_classes,
        "recommended_action": recommended_action.lower(),
        "set_size": set_size,
        "is_ambiguous": is_ambiguous,
        "out_of_distribution": False,
        "alpha": ALPHA,
        "q_hat": Q_HAT,
        "calibration_note": CALIBRATION_NOTE
    }

def run_test(name, prob_dict, expected_set, expected_action, expected_ambiguous, expected_ood=False):
    res = conformal_from_proba(prob_dict)
    
    match = (
        res["prediction_set"] == expected_set and
        res["recommended_action"] == expected_action and
        res["is_ambiguous"] == expected_ambiguous and
        res["out_of_distribution"] == expected_ood
    )
    
    print(f"--- {name} ---")
    print(f"Probabilities: {prob_dict}")
    print(f"Expected: set {expected_set}, action {expected_action}, ambiguous {expected_ambiguous}, ood {expected_ood}")
    print(f"Actual:   set {res['prediction_set']}, action {res['recommended_action']}, ambiguous {res['is_ambiguous']}, ood {res['out_of_distribution']}")
    print(f"Match: {'YES' if match else 'NO'}")
    if not match:
        sys.exit(1)

run_test(
    "clarithromycin/manual", 
    {"high": 0.0028, "intermediate": 0.4734, "low": 0.5238},
    ["intermediate", "low"], "intermediate", True
)

run_test(
    "domperidone/manual",
    {"high": 0.9991, "intermediate": 0.0000, "low": 0.0009},
    ["high"], "high", False
)

run_test(
    "loratadine/chantest",
    {"high": 0.0, "intermediate": 0.0, "low": 1.0},
    ["low"], "low", False
)

run_test(
    "synthetic empty-set check",
    {"high": 0.35, "intermediate": 0.35, "low": 0.30},
    [], "high", True, expected_ood=True
)

print("\nALL TESTS PASSED.")
