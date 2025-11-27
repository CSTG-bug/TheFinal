#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN：手动调参 & 最终定型
================================================
工作流：
  1) 手动调参：MANUAL_TUNE=True，在 MANUAL_TRIALS 写若干组参数 → 运行 → 看排名与最优
  2) 确定参数：MANUAL_TUNE=False，把最终参数填进 FINAL_PARAMS → 运行 → 打印摘要；若达标且开关允许则落盘
  3) 全流程含：数据读取、Pipeline(Scaler+KNN)、CV(R²)、训练/测试指标、可选保存

注意：
  - KNN 对尺度敏感，默认给出三种缩放器（standard/minmax/robust）；可为单组 trial 覆盖全局缩放器
  - 评价指标：R² / MAPE(%) / MAE / RMSE（训练/测试） + RepeatedKFold 的 CV R²（稳健）
  - 不满意不落盘：MASTER_SAVE_SWITCH + AUTO_SAVE_RULE
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# =========【必改 1】四个预处理 CSV 路径 =========
X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS-ZL\output\Element-UTS-ZLX_train_raw.csv"
Y_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS-ZL\output\Element-UTS-ZLy_train.csv"
X_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS-ZL\output\Element-UTS-ZLX_test_raw.csv"
Y_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS-ZL\output\Element-UTS-ZLy_test.csv"

# =========【必改 2】运行模式与输出目录 =========
MANUAL_TUNE = False                                 # 调参期 True；确定参数后改 False
OUTPUT_DIR  = Path(r"D:\MLDesignAl\TheFinal\KNN\Element-UTS-ZL\output")

# ========= 保存策略（只在 MANUAL_TUNE=False 分支里生效）=========
MASTER_SAVE_SWITCH = True                         # 满意后改 True 才允许落盘
AUTO_SAVE_RULE = {                                 # 达标才保存；不想限制就设为 None
    "min_test_r2": None,                           # 例：0.75
    "max_test_mape": None,                         # 例：16.0  （百分比）
    "min_cv_r2": None                              # 例：0.70
}
# AUTO_SAVE_RULE = None

# ========= 随机性与交叉验证 =========
SEED = 42
CV_SPLITS = 5
CV_REPEATS = 2

# ========= 全局缩放器（手动调参默认用它；单个 trial 可覆盖）=========
SCALER_CHOICE = 'standard'                         # 'standard' / 'minmax' / 'robust' / 'identity'
def get_scaler(name: str):
    name = name.lower()
    if name == 'standard': return StandardScaler()
    if name == 'minmax':   return MinMaxScaler()
    if name == 'robust':   return RobustScaler()
    if name == 'identity': return FunctionTransformer(lambda X: X, validate=False)  # 不做缩放
    raise ValueError(f"Unknown scaler: {name}")

# =========【必改 3A】手动调参参数组（MANUAL_TUNE=True 时使用）=========
MANUAL_TRIALS = [
    {"n_neighbors": 5, "weights": "uniform",  "p": 1,   "leaf_size": 10},

    {"n_neighbors": 5, "weights": "uniform",  "p": 1,   "leaf_size": 25},

    {"n_neighbors": 5, "weights": "uniform",  "p": 1,   "leaf_size": 45},

    {"n_neighbors": 5, "weights": "uniform",  "p": 1,   "leaf_size": 60},
]

# =========【必改 3B】最终参数（MANUAL_TUNE=False 时使用）=========
FINAL_PARAMS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "p": 1,
    "leaf_size": 10,
    "metric": "minkowski",
}
FINAL_SCALER = 'standard'   # 最终采用的缩放器：'standard' / 'minmax' / 'robust' / 'identity'

# ========= 工具函数 =========
def mape(y_true, y_pred) -> float:
    """MAPE（%），对 y_true==0 做极小值保护避免除零。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

def read_xy_files(x_train_path, y_train_path, x_test_path, y_test_path):
    """读取四个 CSV，返回 X/y 与特征名列表。"""
    X_train_df = pd.read_csv(x_train_path)
    X_test_df  = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0].astype(float).to_numpy().reshape(-1)
    y_test  = pd.read_csv(y_test_path).iloc[:, 0].astype(float).to_numpy().reshape(-1)
    feature_names = X_train_df.columns.tolist()
    return X_train_df.to_numpy(), y_train, X_test_df.to_numpy(), y_test, feature_names

def _should_save(metrics: dict) -> (bool, str):
    """根据总开关与阈值判断是否保存（仅 Final 分支使用）。"""
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

# ========= 单次评估（手动/最终共用）=========
def evaluate_knn_once(X_train, y_train, X_test, y_test, feature_names,
                      params: dict, scaler_name: str, cv_splits=5, cv_repeats=2):
    """
    用给定 params 跑一遍 KNN，返回 (metrics, model, y_pred_test)。
    - 使用 Pipeline([scaler, KNN])，避免数据泄漏
    - params 支持 n_neighbors / weights / p / leaf_size / metric
    """
    pipe = Pipeline([
        ("scaler", get_scaler(scaler_name)),
        ("knn", KNeighborsRegressor())
    ])
    knn_params = {
        "knn__n_neighbors": int(params.get("n_neighbors", 7)),
        "knn__weights": params.get("weights", "uniform"),
        "knn__p": int(params.get("p", 2)),
        "knn__leaf_size": int(params.get("leaf_size", 30)),
        "knn__metric": params.get("metric", "minkowski")
    }
    pipe.set_params(**knn_params)

    # CV R²（稳健）
    cv = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=SEED)
    cv_scores = cross_val_score(pipe, X_train, y_train, scoring="r2", cv=cv, n_jobs=-1)
    cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())

    # 训练/测试拟合与预测
    pipe.fit(X_train, y_train)
    y_pred_tr = pipe.predict(X_train)
    y_pred_te = pipe.predict(X_test)

    metrics = {
        "scaler": scaler_name,
        "params": {k.replace("knn__", ""): v for k, v in knn_params.items()},
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
        "n_features": int(len(feature_names)),
    }
    return metrics, pipe, y_pred_te

# ========= 落盘（仅在允许保存时调用）=========
def save_bundle(outdir: Path, tag: str, metrics: dict, model, y_test, y_pred_test):
    """保存：metrics.json / predictions.csv / 模型.joblib"""
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"KNN_{tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    ape = np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
    pd.DataFrame({"y_test": y_test, "y_pred": y_pred_test, "APE_percent": ape}).to_csv(
        outdir / f"KNN_{tag}_predictions.csv", index=False
    )
    dump(model, outdir / f"KNN_{tag}_best_model.joblib")

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
        print(f"全局缩放器（默认）：{SCALER_CHOICE}")
        packs = []
        print("\n================ KNN 手动调参 ================\n")
        for i, p in enumerate(MANUAL_TRIALS, 1):
            this_scaler = p.get("scaler", SCALER_CHOICE)
            pretty = {**p, "scaler": this_scaler}
            print(f"[{i}/{len(MANUAL_TRIALS)}] 训练参数：{pretty}")
            metrics, model, y_pred_te = evaluate_knn_once(
                X_train, y_train, X_test, y_test, feature_names,
                params=p, scaler_name=this_scaler,
                cv_splits=CV_SPLITS, cv_repeats=CV_REPEATS
            )
            row = {
                "idx": i, "scaler": metrics["scaler"],
                "k": metrics["params"]["n_neighbors"], "weights": metrics["params"]["weights"],
                "p": metrics["params"]["p"], "leaf": metrics["params"]["leaf_size"],
                "test_r2": metrics["test_r2"], "cv_r2": metrics["cv_r2"], "cv_std": metrics["cv_std"],
                "rmse": metrics["test_rmse"], "mape%": metrics["test_mape"]
            }
            packs.append((row, metrics, model, y_pred_te))

        table = pd.DataFrame([r for r, *_ in packs]).sort_values("test_r2", ascending=False)
        print("\n=== 排名（按 Test R²）===\n")
        print(table.to_string(index=False, justify="center",
                              float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else f"{x}"))

        best_row, best_metrics, _, _ = max(packs, key=lambda t: t[0]["test_r2"])
        print("\n=== 最优组合（当前轮次） ===")
        print(best_row)
        print("\n提示：k=n_neighbors(3,5,7,9,11,13,15)")
        print("      weight(uniform,distance)")
        print("      p(2,1)")
        print("      leaf_size(10,20,30,40,50,60)")
        print("\n下一步：把你满意的那组参数写入 FINAL_PARAMS / FINAL_SCALER；将 MANUAL_TUNE=False 再运行。")

    else:
        # ============ 最终定型与保存模式 ============
        print(f"\n--- Final 将采用的缩放器：{FINAL_SCALER} ---")
        print("--- Final 将采用的 KNN 参数 ---")
        for k in ["n_neighbors", "weights", "p", "leaf_size", "metric"]:
            if k in FINAL_PARAMS:
                print(f"{k:>12}: {FINAL_PARAMS[k]}")

        metrics, model, y_pred_te = evaluate_knn_once(
            X_train, y_train, X_test, y_test, feature_names,
            params=FINAL_PARAMS, scaler_name=FINAL_SCALER,
            cv_splits=CV_SPLITS, cv_repeats=CV_REPEATS
        )

        # 控制台摘要（训练/测试 + CV）
        print("\n=== 训练集性能 ===")
        print(f"R²   : {metrics['train_r2']:.6f}")
        print(f"MAPE : {metrics['train_mape']:.2f}%")
        print(f"MAE  : {metrics['train_mae']:.6f}")
        print(f"RMSE : {metrics['train_rmse']:.6f}")

        print("\n=== 测试集性能 ===")
        print(f"R²   : {metrics['test_r2']:.6f}")
        print(f"MAPE : {metrics['test_mape']:.2f}%")
        print(f"MAE  : {metrics['test_mae']:.6f}")
        print(f"RMSE : {metrics['test_rmse']:.6f}")

        print(f"\nCV R² (RepeatedKFold): {metrics['cv_r2']:.4f} ± {metrics['cv_std']:.4f}")

        allow, reason = _should_save(metrics)
        if not allow:
            print(f"\n⚠️ 未保存任何文件：{reason}")
        else:
            tag = (f"final_scaler{metrics['scaler']}_k{metrics['params']['n_neighbors']}"
                   f"_w{metrics['params']['weights']}_p{metrics['params']['p']}"
                   f"_leaf{metrics['params']['leaf_size']}")
            save_bundle(OUTPUT_DIR, tag, metrics, model, y_test, y_pred_te)
            print(f"\n✅ 已保存到：{OUTPUT_DIR.resolve()}")
