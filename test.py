import joblib
import pandas as pd

model = joblib.load(r"D:\MLDesignAl\TheFinal\XGBoost\ElementTreatmentEl-UTS\output-exceptEL\XGB_best_model.joblib")

one = {
    "Si": 0.05,
    "Fe": 0.06,
    "Cu": 0.20,
    "Mn": 0.02,
    "Mg": 2.60,
    "Cr": 0.19,
    "Zn": 7.28,
    "V" : 0.12,
    "Ti": 0.02,
    "Zr": 0.17,
    "Li": 0.00,
    "Ni": 0.60,
    "Be": 0.00,
    "Sc": 0.15,
    "Ag": 0.39,
    "Bi": 0.35,
    "Pb": 0.08,
    "Al": 87.72,
    "SS Temp"    : 461,
    "Ageing Temp": 124,
    "Ageing Time": 24.2,
    # ...
}
X_one = pd.DataFrame([one])

if hasattr(model, "feature_names_in_"):
    X_one = X_one[model.feature_names_in_]

y_pred = model.predict(X_one)[0]
print("预测值:", y_pred)
