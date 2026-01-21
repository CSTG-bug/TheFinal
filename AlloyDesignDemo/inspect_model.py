from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).parent / "models" / "EL_XGB_best_model.joblib"  

m = joblib.load(MODEL_PATH)
print("type:", type(m))

print("has feature_names_in_:", hasattr(m, "feature_names_in_"))
print("has get_booster:", hasattr(m, "get_booster"))

if hasattr(m, "n_features_in_"):
    print("n_features_in_:", m.n_features_in_)

if hasattr(m, "get_booster"):
    try:
        b = m.get_booster()
        print("booster.feature_names:", b.feature_names)
    except Exception as e:
        print("get_booster failed:", e)
