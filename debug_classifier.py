from sklearn.base import BaseEstimator, ClassifierMixin, get_tags, is_classifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class FittedHolder:
    def __init__(self, estimator):
        self.estimator = estimator

class PreFittedEstimatorWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, holder=None):
        self.holder = holder
        if holder is not None:
            self.classes_ = holder.estimator.classes_

rf = RandomForestClassifier(n_estimators=10).fit(np.random.randn(5, 2), [0, 1, 0, 1, 0])
w = PreFittedEstimatorWrapper(FittedHolder(rf))

print("isinstance(w, ClassifierMixin):", isinstance(w, ClassifierMixin))
print("w estimator_type tag:", get_tags(w).estimator_type)
print("is_classifier(w):", is_classifier(w))
