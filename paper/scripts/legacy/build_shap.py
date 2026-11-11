import joblib
import pandas as pd
import shap
import numpy as np

# STEP A: build background file
print("Loading bundle and data...")
bundle = joblib.load('deployable_chantest_3class.joblib')
df = pd.read_csv('../data/chantest_training.csv')

features = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
X_train = df[features].values

print("Scaling data...")
X_train_scaled = bundle['scaler'].transform(X_train)

print("Running shap.kmeans...")
background = shap.kmeans(X_train_scaled, 15)

print("Saving background to shap_background.joblib...")
joblib.dump(background, 'shap_background.joblib')
print("Done!")
