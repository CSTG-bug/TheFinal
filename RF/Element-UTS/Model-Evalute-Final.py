#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RandomForest 固定参数版（极简评估 + 训练/测试双摘要 + 可选保存）
================================================================
你可以：
  1) 在顶部“【需修改】路径设置”处填写 4 个 CSV 路径
  2) 如需落盘，改 SAVE_OUTPUT=True
  3) 若要换一组超参，只改 CHOSEN_PARAMS 字典即可
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# ==========【需修改】路径设置 ==========
X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_train.csv"
Y_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_train.csv"
X_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_test.csv"
Y_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_test.csv"

# 输出目录 & 总保存开关（不满意就先 False）
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\RF\Element-UTS\output")
SAVE_OUTPUT = False   # ← 满意后改 True 再运行

# ========== 固定的 RF 超参（你给定的最终方案） ==========
CHOSEN_PARAMS = {
    "n_estimators": 1500,
    "max_features": None,   # 使用全部特征
    "max_depth": 15,
    "min_samples_leaf": 1,
    "min_samples_split": 20,
    "bootstrap": True,
    "oob_score": True,
    "random_state": 42,     # 固定种子以保证可复现
    "n_jobs": -1            # 并行加速
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
def train_and_eval_fixed_rf(X_train, y_train, X_test, y_test, feature_names, outdir: Path):
    """按照固定超参训练 RF，打印并（可选）保存结果。"""
    # 1) 建模与拟合
    rf = RandomForestRegressor(**CHOSEN_PARAMS)
    rf.fit(X_train, y_train)

    # 2) 预测
    y_pred_tr = rf.predict(X_train)
    y_pred_te = rf.predict(X_test)

    # 2.1 森林结构（每棵树的深度/叶子数统计，研判是否“深度被卡”很有用）
    depths = np.array([est.get_depth() for est in rf.estimators_])
    leaves = np.array([est.get_n_leaves() for est in rf.estimators_])
    depth_min, depth_mean, depth_max = int(depths.min()), float(depths.mean()), int(depths.max())
    leaves_mean = float(leaves.mean())

    # 2.2 训练集目标的统计（看数据尺度是否合理）
    ytr_mean = float(np.mean(y_train))
    ytr_std  = float(np.std(y_train))
    ytr_min  = float(np.min(y_train))
    ytr_max  = float(np.max(y_train))

    # 3) 指标
    metrics = {
        "model": "RandomForestRegressor",
        "params": CHOSEN_PARAMS,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(len(feature_names)),
        "ytrain_mean": ytr_mean,
        "ytrain_std": ytr_std,
        "ytrain_min": ytr_min,
        "ytrain_max": ytr_max,
        "forest_depth_min": depth_min,
        "forest_depth_mean": depth_mean,
        "forest_depth_max": depth_max,
        "forest_leaves_mean": leaves_mean,
        "train_r2": float(r2_score(y_train, y_pred_tr)),
        "train_mape": mape(y_train, y_pred_tr),
        "train_mae": float(mean_absolute_error(y_train, y_pred_tr)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_tr))),
        "test_r2": float(r2_score(y_test, y_pred_te)),
        "test_mape": mape(y_test, y_pred_te),
        "test_mae": float(mean_absolute_error(y_test, y_pred_te)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_te))),
    }
    # OOB（仅在 bootstrap=True & oob_score=True 时可用）
    if getattr(rf, "oob_score_", None) is not None:
        metrics["oob_r2"] = float(rf.oob_score_)

    # 4) 控制台摘要 —— 数据概况 + 训练集性能 + 测试集性能
    print("\n=== 数据概况 ===")
    print(f"n_train / n_test / n_features : {metrics['n_train']} / {metrics['n_test']} / {metrics['n_features']}")
    print(f"y_train: mean={metrics['ytrain_mean']:.3f}, std={metrics['ytrain_std']:.3f}, "
          f"min={metrics['ytrain_min']:.3f}, max={metrics['ytrain_max']:.3f}")
    print(f"Forest depth(min/mean/max) = {metrics['forest_depth_min']} / {metrics['forest_depth_mean']:.2f} / {metrics['forest_depth_max']}; "
          f"leaves(mean) = {metrics['forest_leaves_mean']:.1f}")

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
    if "oob_r2" in metrics:
        print(f"OOB R²: {metrics['oob_r2']:.6f}")

    print("\n--- 使用的 RF 参数 ---")
    key_order = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
                 "max_features", "bootstrap", "oob_score", "random_state", "n_jobs"]
    for k in key_order:
        if k in CHOSEN_PARAMS:
            print(f"{k:>18}: {CHOSEN_PARAMS[k]}")

    # 5) （可选）落盘
    if SAVE_OUTPUT:
        outdir.mkdir(parents=True, exist_ok=True)

        # 5.1 指标
        with open(outdir / "RF_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # 5.2 预测（含APE%）
        ape = np.abs((y_test - y_pred_te) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
        pd.DataFrame({"y_test": y_test, "y_pred": y_pred_te, "APE_percent": ape}).to_csv(
            outdir / "RF_predictions.csv", index=False
        )

        # 5.3 特征重要性
        importances = getattr(rf, "feature_importances_", np.zeros(len(feature_names)))
        pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        ).to_csv(outdir / "RF_importances.csv", index=False)

        # 5.4 模型
        dump(rf, outdir / "RF_best_model.joblib")
        print(f"\n✅ 已保存到：{outdir.resolve()}")

    return metrics

# ========== 主程序 ==========
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH
    )
    train_and_eval_fixed_rf(X_train, y_train, X_test, y_test, feature_names, outdir=OUTPUT_DIR)
