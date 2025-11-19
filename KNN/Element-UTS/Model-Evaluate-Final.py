#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN 固定参数版（极简评估 + 训练/测试双摘要 + 可选保存）
====================================================
用法：
  1) 修改“【需修改】路径设置”的 4 个 CSV
  2) 若需保存结果文件，将 SAVE_OUTPUT=True
  3) 如要改参数，直接改 KNN_PARAMS / SCALER 即可
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# ==========【需修改】路径设置 ==========
X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_train.csv"
Y_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_train.csv"
X_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_test.csv"
Y_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_test.csv"

# 输出目录 & 总保存开关（不满意就先 False）
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\KNN\Element-UTS\output")
SAVE_OUTPUT = False   # ← 满意后改 True 再运行

# ========== 固定的缩放器与 KNN 超参 ==========
SCALER = FunctionTransformer(lambda X: X, validate=False)
KNN_PARAMS = {
    "n_neighbors": 7,         # k
    "weights": "uniform",     # 不进行距离加权
    "p": 2,                   # 欧氏距离
    "leaf_size": 30,          # 近似搜索的叶子大小（对精度几乎无影响）
    "metric": "minkowski"
}

# ========== 小工具 ==========
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
    y_train = pd.read_csv(y_train_path).iloc[:, 0].to_numpy().reshape(-1)
    y_test  = pd.read_csv(y_test_path).iloc[:, 0].to_numpy().reshape(-1)
    feature_names = X_train_df.columns.tolist()
    return X_train_df.to_numpy(), y_train, X_test_df.to_numpy(), y_test, feature_names

# ========== 训练与评估 ==========
def train_and_eval_fixed_knn(X_train, y_train, X_test, y_test, feature_names, outdir: Path):
    """按照固定超参训练 KNN，打印并（可选）保存结果。"""
    # 1) Pipeline：缩放器 + KNN（避免数据泄漏）
    pipe = Pipeline([
        ("scaler", SCALER),
        ("knn", KNeighborsRegressor(**KNN_PARAMS))
    ])
    pipe.fit(X_train, y_train)

    # 2) 预测
    y_pred_tr = pipe.predict(X_train)
    y_pred_te = pipe.predict(X_test)

    # 3) 训练集目标统计（快速感知尺度/异常）
    ytr_mean = float(np.mean(y_train))
    ytr_std  = float(np.std(y_train))
    ytr_min  = float(np.min(y_train))
    ytr_max  = float(np.max(y_train))

    # 4) 指标
    metrics = {
        "model": "KNeighborsRegressor",
        "scaler": "StandardScaler",
        "params": KNN_PARAMS,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(len(feature_names)),
        "ytrain_mean": ytr_mean,
        "ytrain_std": ytr_std,
        "ytrain_min": ytr_min,
        "ytrain_max": ytr_max,
        "train_r2": float(r2_score(y_train, y_pred_tr)),
        "train_mape": mape(y_train, y_pred_tr),
        "train_mae": float(mean_absolute_error(y_train, y_pred_tr)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_tr))),
        "test_r2": float(r2_score(y_test, y_pred_te)),
        "test_mape": mape(y_test, y_pred_te),
        "test_mae": float(mean_absolute_error(y_test, y_pred_te)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_te))),
    }

    # 5) 控制台摘要 —— 数据概况 + 训练集性能 + 测试集性能 + 参数清单
    print("\n=== 数据概况 ===")
    print(f"n_train / n_test / n_features : {metrics['n_train']} / {metrics['n_test']} / {metrics['n_features']}")
    print(f"y_train: mean={metrics['ytrain_mean']:.3f}, std={metrics['ytrain_std']:.3f}, "
          f"min={metrics['ytrain_min']:.3f}, max={metrics['ytrain_max']:.3f}")

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

    print("\n--- 使用的 KNN 设置 ---")
    key_order = ["n_neighbors", "weights", "p", "leaf_size", "metric"]
    for k in key_order:
        if k in KNN_PARAMS:
            print(f"{k:>12}: {KNN_PARAMS[k]}")
    print(f"{'scaler':>12}: StandardScaler")

    # 6) （可选）落盘
    if SAVE_OUTPUT:
        outdir.mkdir(parents=True, exist_ok=True)

        # 6.1 指标
        with open(outdir / "KNN_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # 6.2 预测（含 APE%）
        ape = np.abs((y_test - y_pred_te) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
        pd.DataFrame({"y_test": y_test, "y_pred": y_pred_te, "APE_percent": ape}).to_csv(
            outdir / "KNN_predictions.csv", index=False
        )

        # 6.3 模型（含缩放器）
        dump(pipe, outdir / "KNN_best_model.joblib")
        print(f"\n✅ 已保存到：{outdir.resolve()}")

    return metrics

# ========== 主程序 ==========
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH
    )
    train_and_eval_fixed_knn(X_train, y_train, X_test, y_test, feature_names, outdir=OUTPUT_DIR)
