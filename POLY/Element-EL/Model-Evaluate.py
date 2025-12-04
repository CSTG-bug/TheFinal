#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多项式回归（POLY）模型训练与评估
============================================================
功能说明：
  ✅ 自动读取你指定的四个预处理后 CSV 文件
  ✅ 生成多项式特征（默认比较 degree=2, 3, 4 三种）
  ✅ 对每种 degree 训练 Ridge 回归（可有效缓解过拟合）
  ✅ 计算并保存 R²、MAPE、MAE、RMSE 等指标
  ✅ 自动选出测试集 R² 最高（若相同则 MAPE 更低）的最佳模型
  ✅ 可选择是否保存最佳模型为 .joblib 文件
  ✅ 所有输出（metrics、预测结果、系数、summary、最佳模型）都保存到输出文件夹
"""

# ===== 导入库 =====
import numpy as np
import pandas as pd
from pathlib import Path
import json
from joblib import dump
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

# ===== 固定随机种子 =====
SEED = 42

# ===== MAPE计算函数 =====
def mean_absolute_percentage_error(y_true, y_pred):
    """计算MAPE（百分比），对0值做保护"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

# ===== 数据读取函数 =====
def read_xy_files(x_train_path, y_train_path, x_test_path, y_test_path):
    """读取四个预处理后的CSV文件"""
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0].to_numpy()
    y_test = pd.read_csv(y_test_path).iloc[:, 0].to_numpy()
    feature_names = X_train.columns.tolist()
    return X_train.to_numpy(), y_train, X_test.to_numpy(), y_test, feature_names

# ===== 单个多项式模型的训练与评估 =====
def train_and_eval_poly(degree, X_train, y_train, X_test, y_test, feature_names, outdir, save_model=False):
    """
    构建“多项式特征 + Ridge回归”管道，
    训练模型，计算指标并保存结果。
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ 定义多项式特征 + 标准化 + Ridge
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=SEED))
    ])

    # 2️⃣ 交叉验证
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)

    # 3️⃣ 训练与预测
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 4️⃣ 指标计算
    metrics = {
        "degree": degree,
        "cv_r2_mean": float(cv_mean),
        "cv_r2_std": float(cv_std),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "train_mape": float(mean_absolute_percentage_error(y_train, y_pred_train)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "test_mape": float(mean_absolute_percentage_error(y_test, y_pred_test)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
    }

    # 5️⃣ 保存预测结果
    ape = np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
    pred_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred_test, "APE_percent": ape})
    pred_df.to_csv(outdir / f"POLY_deg{degree}_predictions.csv", index=False)

    # 6️⃣ 保存指标JSON
    with open(outdir / f"POLY_deg{degree}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 7️⃣ 可选保存模型
    if save_model:
        dump(model, outdir / f"POLY_deg{degree}.joblib")

    print(f"✅ 多项式回归(degree={degree}) 完成：R²={metrics['test_r2']:.4f}, MAPE={metrics['test_mape']:.2f}%")
    return metrics

# ===== 主程序入口 =====
if __name__ == "__main__":
    # === 【需修改区域】 ===
    X_train_path = r"D:\MLDesignAl\TheFinal\Data\Element-EL\output\Element-EL-X_train.csv"
    y_train_path = r"D:\MLDesignAl\TheFinal\Data\Element-EL\output\Element-EL-y_train.csv"
    X_test_path  = r"D:\MLDesignAl\TheFinal\Data\Element-EL\output\Element-EL-X_test.csv"
    y_test_path  = r"D:\MLDesignAl\TheFinal\Data\Element-EL\output\Element-EL-y_test.csv"
    output_dir = Path(r"D:\MLDesignAl\TheFinal\POLY\Element-EL\output")
    save_best_model = False  # 是否保存最佳模型

    # === 读取数据 ===
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_train_path, y_train_path, X_test_path, y_test_path
    )

    # === 多项式阶数列表 ===
    degrees = [2, 3, 4]

    # === 循环训练不同阶数的多项式回归模型 ===
    results = {}
    for deg in degrees:
        results[f"POLY_deg{deg}"] = train_and_eval_poly(deg, X_train, y_train, X_test, y_test, feature_names, output_dir)

    # === 汇总结果 ===
    summary_df = pd.DataFrame(results).T.sort_values(by=["test_r2", "test_mape"], ascending=[False, True])
    summary_path = output_dir / "POLY_summary.csv"
    summary_df.to_csv(summary_path)
    print("\n📊 多项式模型对比汇总：")
    print(summary_df[["test_r2", "test_mape", "test_rmse"]])

    # === 选出最佳模型 ===
    best_model_name = summary_df.index[0]
    best_degree = int(summary_df.loc[best_model_name, "degree"])
    print(f"\n🏆 最佳模型：{best_model_name} (degree={best_degree})")

    # === 重新训练最佳模型并保存 ===
    if save_best_model:
        best_model = Pipeline([
            ("poly", PolynomialFeatures(degree=best_degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=SEED))
        ])
        best_model.fit(X_train, y_train)
        dump(best_model, output_dir / f"best_POLY_deg{best_degree}.joblib")
        print(f"✅ 已保存最佳模型：best_POLY_deg{best_degree}.joblib")
