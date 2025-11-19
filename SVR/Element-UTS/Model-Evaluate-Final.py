#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVR 固定参数版（修复：可保存 joblib，训练/测试双摘要 + 可选落盘）
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
# ★ 修复点：不再使用 lambda 恒等变换，直接用 'passthrough' 跳过缩放器
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# ==========【需修改】数据路径 ==========
X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_train.csv"
Y_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_train.csv"
X_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_test.csv"
Y_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_test.csv"

# 输出目录 & 总保存开关
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\SVR\Element-UTS\output")
SAVE_OUTPUT = False  # 满意后设为 True

# ========== 固定参数 ==========
SVR_PARAMS = {
    "kernel": "rbf",
    "C": 600.0,
    "epsilon": 0.25,   # 你最后一次运行显示的是 0.25
    "gamma": 0.03
}

# ★ 修复点：CSV 已缩放 → 缩放器用 'passthrough'，可被安全序列化
SCALER = 'passthrough'

# ========== 工具 ==========
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
    return X_train_df.to_numpy(), y_train, X_test_df.to_numpy(), y_test, feature_names

# ========== 训练与评估 ==========
def train_and_eval_fixed_svr(X_train, y_train, X_test, y_test, feature_names, outdir: Path):
    pipe = Pipeline([
        ("scaler", SCALER),          # 'passthrough'：跳过缩放
        ("svr", SVR(**SVR_PARAMS))
    ])
    pipe.fit(X_train, y_train)

    y_pred_tr = pipe.predict(X_train)
    y_pred_te = pipe.predict(X_test)

    # 数据概况
    ytr_mean, ytr_std = float(np.mean(y_train)), float(np.std(y_train))
    ytr_min, ytr_max = float(np.min(y_train)), float(np.max(y_train))

    metrics = {
        "model": "SVR",
        "scaler": "identity(passthrough)",
        "params": SVR_PARAMS,
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

    # 控制台摘要
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

    print("\n--- 使用的 SVR 参数 ---")
    for k in ["kernel", "C", "epsilon", "gamma"]:
        print(f"{k:>8}: {SVR_PARAMS[k]}")
    print(f"{'scaler':>8}: passthrough")

    # 可选落盘
    if SAVE_OUTPUT:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "SVR_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        ape = np.abs((y_test - y_pred_te) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
        pd.DataFrame({"y_test": y_test, "y_pred": y_pred_te, "APE_percent": ape}).to_csv(
            outdir / "SVR_predictions.csv", index=False
        )

        dump(pipe, outdir / "SVR_best_model.joblib")  # ★ 现在可以正常保存
        print(f"\n✅ 已保存到：{outdir.resolve()}")

    return metrics

# ========== 主程序 ==========
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH
    )
    train_and_eval_fixed_svr(X_train, y_train, X_test, y_test, feature_names, outdir=OUTPUT_DIR)
