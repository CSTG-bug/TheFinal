#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机森林：手动调参 & 最终定型
================================================
你的工作流：
  1) 当需要手动调参：把 MANUAL_TUNE = True，然后在 MANUAL_TRIALS 里写你要试的参数组合，运行；
     - 脚本会打印每组的 Train/Test 指标 + CV R²，并按 Test R² 排名
     - 你根据结果挑一组最满意的参数
  2) 当你确定好参数：把 MANUAL_TUNE = False，并把 FINAL_PARAMS 填好，运行；
     - 脚本会仅训练这组“最终参数”，打印摘要；若 MASTER_SAVE_SWITCH=True 且满足阈值，就落盘结果
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# =========【必改 1】你的四个预处理 CSV 路径 =========
X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-EL\output\Element-EL-X_train.csv"
Y_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-EL\output\Element-EL-y_train.csv"
X_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-EL\output\Element-EL-X_test.csv"
Y_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-EL\output\Element-EL-y_test.csv"

# =========【必改 2】运行模式与输出 =========
MANUAL_TUNE = False                      # 调参期设 True；确定参数后改 False
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\RF\Element-EL\output")

# 保存策略（只在 MANUAL_TUNE=False 时生效）
MASTER_SAVE_SWITCH = True              # 满意后改 True 才会真正落盘
AUTO_SAVE_RULE = {                      # 设阈值“达标才保存”；不需要就设 None
    "min_test_r2": None,                # 例：0.75
    "max_test_mape": None,              # 例：16.0
    "min_cv_r2": None                   # 例：0.70
}
# AUTO_SAVE_RULE = None

# ========= 随机性与交叉验证 =========
SEED = 42
CV_SPLITS = 5
CV_REPEATS = 2

# =========【必改 3A】手动调参参数组（MANUAL_TUNE=True 才会用）=========
MANUAL_TRIALS = [

    {"n_estimators": 271, "max_features": 0.1, "max_depth": 31,
     "min_samples_leaf": 1, "min_samples_split": 4, "bootstrap": True, "oob_score": True,
     "random_state": SEED, "n_jobs": -1},

    {"n_estimators": 271, "max_features": 0.1, "max_depth": 31,
     "min_samples_leaf": 1, "min_samples_split": 4, "bootstrap": True, "oob_score": True,
     "random_state": SEED, "n_jobs": -1},

    {"n_estimators": 271, "max_features": 0.1, "max_depth": 31,
     "min_samples_leaf": 1, "min_samples_split": 4, "bootstrap": True, "oob_score": True,
     "random_state": SEED, "n_jobs": -1},

    {"n_estimators": 271, "max_features": 0.1, "max_depth": 31,
     "min_samples_leaf": 1, "min_samples_split": 4, "bootstrap": True, "oob_score": True,
     "random_state": SEED, "n_jobs": -1},

    {"n_estimators": 271, "max_features": 0.1, "max_depth": 31,
     "min_samples_leaf": 1, "min_samples_split": 4, "bootstrap": True, "oob_score": True,
     "random_state": SEED, "n_jobs": -1},

]

# =========【必改 3B】最终参数（MANUAL_TUNE=False 才会用）=========
FINAL_PARAMS = {
    "n_estimators": 271,
    "max_features": 0.1,
    "max_depth": 31,
    "min_samples_leaf": 1,
    "min_samples_split": 4,
    "bootstrap": True,
    "oob_score": True,
    "random_state": SEED,
    "n_jobs": -1,
}

# ========= 常用工具 =========
def mape(y_true, y_pred) -> float:
    """MAPE（百分比），对 0 做极小值保护避免除零。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

def read_xy_files(x_train_path, y_train_path, x_test_path, y_test_path):
    """读取四个 CSV，返回 numpy 矩阵与列名。"""
    Xtr_df = pd.read_csv(x_train_path)
    Xte_df = pd.read_csv(x_test_path)
    ytr = pd.read_csv(y_train_path).iloc[:, 0].to_numpy().reshape(-1)
    yte = pd.read_csv(y_test_path).iloc[:, 0].to_numpy().reshape(-1)
    feat_names = Xtr_df.columns.tolist()
    return Xtr_df.to_numpy(), ytr, Xte_df.to_numpy(), yte, feat_names

def _should_save(metrics: dict) -> (bool, str):
    """根据总开关与阈值判断是否保存（只在 Final 分支里用）。"""
    if not MASTER_SAVE_SWITCH:
        return False, "MASTER_SAVE_SWITCH=False（总开关关闭）"
    if AUTO_SAVE_RULE is None:
        return True, ""
    reasons = []
    thr = AUTO_SAVE_RULE.get("min_test_r2")
    if thr is not None and metrics.get("test_r2", -1) < thr:
        reasons.append(f"test_r2={metrics['test_r2']:.4f} < {thr}")
    thr = AUTO_SAVE_RULE.get("max_test_mape")
    if thr is not None and metrics.get("test_mape", 1e9) > thr:
        reasons.append(f"test_mape={metrics['test_mape']:.2f}% > {thr}%")
    thr = AUTO_SAVE_RULE.get("min_cv_r2")
    if thr is not None and metrics.get("cv_r2", -1) < thr:
        reasons.append(f"cv_r2={metrics['cv_r2']:.4f} < {thr}")
    if reasons:
        return False, "；".join(reasons)
    return True, ""

def evaluate_rf_once(X_train, y_train, X_test, y_test, feature_names, params: dict):
    """
    用给定 params 训练一遍 RF，返回 (metrics, model, y_pred_test)。
    - 使用 RepeatedKFold 做 CV（R²）
    - 打印/保存逻辑由上层控制
    """
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # 交叉验证（R²）：更稳健
    cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)
    cv_scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=cv, n_jobs=-1)
    cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())

    # 训练/测试集指标
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)

    metrics = {
        "params": params,
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
        "oob_r2": float(getattr(model, "oob_score_", np.nan)),
        "n_estimators": int(params.get("n_estimators", 100)),
        "n_features": int(len(feature_names))
    }
    return metrics, model, y_pred_te

def save_bundle(outdir: Path, metrics: dict, model, y_test, y_pred_test, feature_names):
    """落盘：metrics.json / predictions.csv / importances.csv / 模型.joblib"""
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 指标
    with open(outdir / "RF_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 2) 预测（含 APE%）
    ape = np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
    pd.DataFrame({"y_test": y_test, "y_pred": y_pred_test, "APE_percent": ape}).to_csv(
        outdir / "RF_predictions.csv", index=False
    )

    # 3) 特征重要性
    importances = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
    pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    ).to_csv(outdir / "RF_importances.csv", index=False)

    # 4) 模型
    dump(model, outdir / "RF_best_model.joblib")

# ========= 主程序 =========
if __name__ == "__main__":
    # 读取数据 + 概况
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH
    )
    print("\n=== 数据概况 ===")
    print(f"n_train / n_test / n_features : {len(y_train)} / {len(y_test)} / {len(feature_names)}")
    print(f"y_train: mean={np.mean(y_train):.3f}, std={np.std(y_train):.3f}, "
          f"min={np.min(y_train):.3f}, max={np.max(y_train):.3f}")

    if MANUAL_TUNE:
        # ============ 手动调参模式 ============
        rows, packs = [], []
        print("\n================ RF 手动调参 ================\n")
        for i, p in enumerate(MANUAL_TRIALS, 1):
            print(f"[{i}/{len(MANUAL_TRIALS)}] params: {p}")
            metrics, model, y_pred_te = evaluate_rf_once(
                X_train, y_train, X_test, y_test, feature_names, p
            )
            row = {
                "idx": i,
                "test_r2": metrics["test_r2"],
                "cv_r2": metrics["cv_r2"],
                "cv_std": metrics["cv_std"],
                "oob": metrics["oob_r2"],
                "rmse": metrics["test_rmse"],
                "mape%": metrics["test_mape"],
                "n_estimators": metrics["n_estimators"],
                "max_depth": p.get("max_depth", None),
                "min_samples_leaf": p.get("min_samples_leaf", None),
                "min_samples_split": p.get("min_samples_split", None),
                "max_features": p.get("max_features", None)
            }
            rows.append(row)
            packs.append((row, metrics, model, y_pred_te))

        # 排名展示（按 Test R²）
        table = pd.DataFrame(rows).sort_values("test_r2", ascending=False)
        print("\n=== 排名（按 Test R²）===\n")
        print(table.to_string(index=False, justify="center",
                              float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else f"{x}"))

        # 最优提示
        best_row, best_metrics, _, _ = max(packs, key=lambda t: t[0]["test_r2"])
        print("\n=== 最优组合（当前轮次） ===")
        print(best_row)
        print("\n提示")
        print("  • max_features(每次分裂可用特征数):'sqrt', 0.5, 0.7, None")
        print("  • min_samples_leaf(叶子最小样本):1, 2, 4, 8")
        print("  • min_samples_split(分裂最小样本):2, 5, 10, 20")
        print("  • max_depth(最大深度):None, 20, 30, 40, 50")
        print("  • n_estimators(数的数量):600, 900, 1200, 1500, 1800")
        print("\n下一步：把上面你满意的那组参数复制到 FINAL_PARAMS；"
              "然后把 MANUAL_TUNE=False 再运行，即进入“最终定型与保存”模式。")

    else:
        # ============ 最终定型与保存模式 ============
        print("\n--- Final 将采用的参数 ---")
        key_order = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
                     "max_features", "bootstrap", "oob_score", "random_state", "n_jobs"]
        for k in key_order:
            if k in FINAL_PARAMS:
                print(f"{k:>18}: {FINAL_PARAMS[k]}")

        # 训练/评估
        metrics, model, y_pred_te = evaluate_rf_once(
            X_train, y_train, X_test, y_test, feature_names, FINAL_PARAMS
        )

        # 控制台摘要（训练/测试 + CV + OOB）
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
        if not np.isnan(metrics["oob_r2"]):
            print(f"OOB R²               : {metrics['oob_r2']:.4f}")

        # 是否保存
        allow, reason = _should_save(metrics)
        if not allow:
            print(f"\n⚠️ 未保存任何文件：{reason}")
        else:
            save_bundle(OUTPUT_DIR, metrics, model, y_test, y_pred_te, feature_names)
            print(f"\n✅ 已保存到：{OUTPUT_DIR.resolve()}")
