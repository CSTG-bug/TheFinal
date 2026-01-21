import joblib
import pandas as pd

model = joblib.load(r"D:\MLDesignAl\TheFinal\XGBoost\ElementTreatmentEl-UTS\output-exceptEL\XGB_best_model.joblib")

one = {
    "Si": 0.00,
    "Fe": 0.00,
    "Cu": 2.44,
    "Mn": 0.00,
    "Mg": 2.80,
    "Cr": 0.00,
    "Zn": 7.27,
    "V" : 0.00,
    "Ti": 0.05,
    "Zr": 0.00,
    "Li": 0.00,
    "Ni": 0.00,
    "Be": 0.00,
    "Sc": 0.23,
    "Ag": 0.00,
    "Bi": 0.00,
    "Pb": 0.00,
    "Al": 87.21,
    "SS Temp"    : 462,
    "Ageing Temp": 123,
    "Ageing Time": 21.5,
    # ...
}
X_one = pd.DataFrame([one])

if hasattr(model, "feature_names_in_"):
    X_one = X_one[model.feature_names_in_]

y_pred = model.predict(X_one)[0]
print("预测值:", y_pred)
