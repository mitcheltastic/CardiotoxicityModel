from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class FittedHolder:
    def __init__(self, estimator):
        self.estimator = estimator

class PreFittedEstimatorWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, holder=None):
        self.holder = holder
        if holder is not None:
            self.classes_ = holder.estimator.classes_
            
    def fit(self, X, y, sample_weight=None, groups=None):
        if self.holder is not None:
            self.classes_ = self.holder.estimator.classes_
        return self
        
    def predict(self, X):
        return self.holder.estimator.predict(X)
        
    def predict_proba(self, X):
        return self.holder.estimator.predict_proba(X)

class FinalEstimatorWrapper(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    def __init__(self, estimator=None):
        self.estimator = estimator
        if estimator is not None and hasattr(estimator, 'classes_'):
            self.classes_ = estimator.classes_
            
    def fit(self, X, y, sample_weight=None, groups=None, **kwargs):
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        self.estimator.fit(X, y, **fit_params)
        self.classes_ = self.estimator.classes_
        return self
        
    def predict(self, X):
        return self.estimator.predict(X)
        
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
