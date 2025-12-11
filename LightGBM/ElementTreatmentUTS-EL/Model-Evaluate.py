#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM：手动调参 & 最终定型
================================
1) MANUAL_TUNE=True + MANUAL_TRIALS → 手动跑多组参数，看排名
2) MANUAL_TUNE=False + FINAL_PARAMS → 只跑最终参数，并按阈值选择是否落盘
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# =========【必改 1】四个预处理 CSV 路径 =========
X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\ElementTreatmentUTS-EL\output\ElementTreatmentUTS-EL-X_train.csv"
Y_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\ElementTreatmentUTS-EL\output\ElementTreatmentUTS-EL-y_train.csv"
X_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\ElementTreatmentUTS-EL\output\ElementTreatmentUTS-EL-X_test.csv"
Y_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\ElementTreatmentUTS-EL\output\ElementTreatmentUTS-EL-y_test.csv"

# =========【必改 2】运行模式与输出 =========
MANUAL_TUNE = False   # 调参期设 True；确定参数后改 False
OUTPUT_DIR  = Path(r"D:\MLDesignAl\TheFinal\LightGBM\ElementTreatmentUTS-EL\output")

# ========= 保存策略（只在 MANUAL_TUNE=False 时生效）=========
MASTER_SAVE_SWITCH = True
AUTO_SAVE_RULE = {
    "min_test_r2": None,
    "max_test_mape": None,
    "min_cv_r2": None
}
# AUTO_SAVE_RULE = None

# ========= 随机性与交叉验证 =========
SEED = 42
CV_SPLITS = 5
CV_REPEATS = 2

# =========【必改 3A】手动调参参数组 =========
MANUAL_TRIALS = [
    {
        "n_estimators": 811,"num_leaves": 10,"max_depth": 10,"learning_rate": 0.08,"subsample": 0.6,
        "colsample_bytree": 0.9,"min_child_samples": 30,"reg_lambda": 5.0,"reg_alpha": 0.7,
    },

    {
        "n_estimators": 811,"num_leaves": 10,"max_depth": 10,"learning_rate": 0.08,"subsample": 0.6,
        "colsample_bytree": 0.9,"min_child_samples": 30,"reg_lambda": 5.0,"reg_alpha": 0.6,
    },

    {
        "n_estimators": 811,"num_leaves": 10,"max_depth": 10,"learning_rate": 0.08,"subsample": 0.6,
        "colsample_bytree": 0.9,"min_child_samples": 30,"reg_lambda": 5.0,"reg_alpha": 0.5,
    },

    {
        "n_estimators": 811,"num_leaves": 10,"max_depth": 10,"learning_rate": 0.08,"subsample": 0.6,
        "colsample_bytree": 0.9,"min_child_samples": 30,"reg_lambda": 5.0,"reg_alpha": 0.4,
    },

    {
        "n_estimators": 811,"num_leaves": 10,"max_depth": 10,"learning_rate": 0.08,"subsample": 0.6,
        "colsample_bytree": 0.9,"min_child_samples": 30,"reg_lambda": 5.0,"reg_alpha": 0.3,
    },
]

# =========【必改 3B】最终参数 =========
FINAL_PARAMS = {
    "n_estimators": 811,
    "num_leaves": 10,
    "max_depth": 10,
    "learning_rate": 0.08,
    "subsample": 0.6,
    "colsample_bytree": 0.9,
    "min_child_samples": 30,
    "reg_lambda": 5.0,
    "reg_alpha": 0.4,
}

# ========= 工具函数 =========
def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

def read_xy_files(x_train_path, y_train_path, x_test_path, y_test_path):
    X_train_df = pd.read_csv(x_train_path)
    X_test_df  = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0].to_numpy().reshape(-1)
    y_test  = pd.read_csv(y_test_path).iloc[:, 0].to_numpy().reshape(-1)
    feature_names = X_train_df.columns.tolist()
    X_train = X_train_df.to_numpy()
    X_test = X_test_df.to_numpy()
    return X_train, y_train, X_test, y_test, feature_names

def _should_save(metrics: dict) -> (bool, str):
    if not MASTER_SAVE_SWITCH:
        return False, "MASTER_SAVE_SWITCH=False（总开关关闭）"
    if AUTO_SAVE_RULE is None:
        return True, ""

    reasons = []
    thr = AUTO_SAVE_RULE.get("min_test_r2")
    if thr is not None and metrics.get("test_r2", -1) < thr:
        reasons.append(f"test_r2={metrics['test_r2']:.4f} < min_test_r2={thr}")
    thr = AUTO_SAVE_RULE.get("max_test_mape")
    if thr is not None and metrics.get("test_mape", 1e9) > thr:
        reasons.append(f"test_mape={metrics['test_mape']:.2f}% > max_test_mape={thr}%")
    thr = AUTO_SAVE_RULE.get("min_cv_r2")
    if thr is not None and metrics.get("cv_r2", -1) < thr:
        reasons.append(f"cv_r2={metrics['cv_r2']:.4f} < min_cv_r2={thr}")

    if reasons:
        return False, "；".join(reasons)
    return True, ""

# ========= 核心：给定一组参数，训练一次 LightGBM 并评估 =========
def evaluate_lgbm_once(
    X_train, y_train, X_test, y_test, feature_names,
    params: dict,
    cv_splits: int = CV_SPLITS,
    cv_repeats: int = CV_REPEATS,
):
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    local_params = params.copy()
    local_params.setdefault("random_state", SEED)
    local_params.setdefault("n_estimators", 500)
    local_params.setdefault("n_jobs", -1)
    local_params.setdefault("objective", "regression")
    local_params.setdefault("boosting_type", "gbdt")

    model = LGBMRegressor(**local_params)
    model.fit(X_train, y_train)

    cv = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=SEED)
    cv_scores = cross_val_score(model, X_train, y_train,
                                scoring="r2", cv=cv, n_jobs=-1)
    cv_mean = float(cv_scores.mean())
    cv_std  = float(cv_scores.std())

    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)

    metrics = {
        "params": local_params,
        "cv_r2": cv_mean,
        "cv_std": cv_std,
        "train_r2": float(r2_score(y_train, y_pred_tr)),
        "train_mape": mape(y_train, y_pred_tr),
        "train_mae": float(mean_absolute_error(y_train, y_pred_tr)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_tr))),
        "test_r2": float(r2_score(y_test, y_pred_te)),
        "test_mape": mape(y_test, y_pred_te),
        "test_mae": float(mean_absolute_error(y_test, y_pred_te)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_te))),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": len(feature_names),
        "n_estimators": int(local_params.get("n_estimators", 0)),
        "num_leaves": local_params.get("num_leaves", None),
        "max_depth": local_params.get("max_depth", None),
        "learning_rate": local_params.get("learning_rate", None),
        "subsample": local_params.get("subsample", None),
        "colsample_bytree": local_params.get("colsample_bytree", None),
        "min_child_samples": local_params.get("min_child_samples", None),
        "reg_lambda": local_params.get("reg_lambda", None),
        "reg_alpha": local_params.get("reg_alpha", None),
    }
    return metrics, model, y_pred_te

def save_bundle_lgbm(outdir: Path, metrics: dict, model, y_test, y_pred_test, feature_names):
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "LGBM_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    ape = np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
    pd.DataFrame({
        "y_test": y_test,
        "y_pred": y_pred_test,
        "APE_percent": ape
    }).to_csv(outdir / "LGBM_predictions.csv", index=False)

    importances = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
    pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).to_csv(
        outdir / "LGBM_importances.csv", index=False
    )

    dump(model, outdir / "LGBM_best_model.joblib")

# ========= 主程序 =========
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH
    )

    print("\n=== 数据概况 ===")
    print(f"n_train / n_test / n_features : {len(y_train)} / {len(y_test)} / {len(feature_names)}")
    print(f"y_train: mean={np.mean(y_train):.3f}, std={np.std(y_train):.3f}, "
          f"min={np.min(y_train):.3f}, max={np.max(y_train):.3f}")

    if MANUAL_TUNE:
        # -------- 手动调参模式 --------
        rows = []
        packs = []
        print("\n================ LGBM 手动调参 ================\n")
        for i, p in enumerate(MANUAL_TRIALS, 1):
            print(f"[{i}/{len(MANUAL_TRIALS)}] params: {p}")
            metrics, model, y_pred_te = evaluate_lgbm_once(
                X_train, y_train, X_test, y_test, feature_names, p
            )
            row = {
                "idx": i,
                "test_r2": metrics["test_r2"],
                "cv_r2": metrics["cv_r2"],
                "cv_std": metrics["cv_std"],
                "rmse": metrics["test_rmse"],
                "mape%": metrics["test_mape"],
                "n_estimators": metrics["n_estimators"],
                "num_leaves": metrics["num_leaves"],
                "max_depth": metrics["max_depth"],
                "min_child_samples":metrics["min_child_samples"],
                "lr": metrics["learning_rate"],
                "subsample": metrics["subsample"],
                "colsample": metrics["colsample_bytree"],
                "reg_lambda":metrics["reg_lambda"],
                "reg_alpha":metrics["reg_alpha"],
            }
            rows.append(row)
            packs.append((row, metrics, model, y_pred_te))

        table = pd.DataFrame(rows).sort_values("test_r2", ascending=False)
        print("\n=== 排名（按 Test R²）===\n")
        print(table.to_string(index=False, justify="center",
                              float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else f"{x}"))

        best_row, best_metrics, _, _ = max(packs, key=lambda t: t[0]["test_r2"])
        print("\n=== 最优组合（当前轮次） ===")
        print(best_row)
        print("\n调参提示：")
        print("      - num_leaves(最大叶子数) = 16,32,64,128")
        print("      - max_depth(深度) = -1,6,7,8")
        print("      - min_child_samples(叶子最小样本数) = 10,20,30,40,60")
        print("      - subsample = 0.6,0.7,0.8,0.9,1.0")
        print("      - colsample_bytree = 0.6,0.7,0.8,0.9,1.0")
        print("      - learning_rate = 0.02,0.03,0.05,0.10")
        print("      - n_estimators = 300,400,600,800,1200")
        print("      - reg_lambda = 1,2,5,10")
        print("      - reg_alpha = 0,0.1,0.5,1.0")
        print("      - min_split_gain(min_gain_to_split) = 0.0,0.1,0.3,0.5")
        print("\n下一步：把满意的参数写入 FINAL_PARAMS；MANUAL_TUNE=False 再运行。")

    else:
        # -------- 最终定型与保存模式 --------
        print("\n--- Final 将采用的 LGBM 参数 ---")
        key_order = [
            "n_estimators", "num_leaves", "max_depth",
            "learning_rate", "subsample", "colsample_bytree",
            "min_child_samples", "reg_lambda", "reg_alpha",
            "random_state", "n_jobs"
        ]
        for k in key_order:
            if k in FINAL_PARAMS:
                print(f"{k:>18}: {FINAL_PARAMS[k]}")

        metrics, model, y_pred_te = evaluate_lgbm_once(
            X_train, y_train, X_test, y_test, feature_names, FINAL_PARAMS
        )

        print("\n=== 训练集性能 ===")
        print(f"R²   : {metrics['train_r2']:.4f}")
        print(f"MAPE : {metrics['train_mape']:.2f}%")
        print(f"MAE  : {metrics['train_mae']:.4f}")
        print(f"RMSE : {metrics['train_rmse']:.4f}")

        print("\n=== 测试集性能 ===")
        print(f"R²   : {metrics['test_r2']:.4f}")
        print(f"MAPE : {metrics['test_mape']:.2f}%")
        print(f"MAE  : {metrics['test_mae']:.4f}")
        print(f"RMSE : {metrics['test_rmse']:.4f}")

        print(f"\nCV R² (RepeatedKFold): {metrics['cv_r2']:.4f} ± {metrics['cv_std']:.4f}")

        allow, reason = _should_save(metrics)
        if not allow:
            print(f"\n⚠️ 未保存任何文件：{reason}")
        else:
            save_bundle_lgbm(OUTPUT_DIR, metrics, model, y_test, y_pred_te, feature_names)
            print(f"\n✅ 已保存到：{OUTPUT_DIR.resolve()}")
