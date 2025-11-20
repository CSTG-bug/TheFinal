#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线性模型（LIN / Ridge / Lasso）训练与评估
===============================================================
功能说明：
  ✅ 自动读取你手动指定的4个CSV文件（X_train、y_train、X_test、y_test）
  ✅ 一次运行三个模型：LinearRegression、Ridge、Lasso
  ✅ 计算训练集与测试集的 R²、MAPE、MAE、RMSE 等指标
  ✅ 自动找出“测试集R²最高（若相同则MAPE更低）”的最佳模型
  ✅ 输出结果文件（metrics、预测结果、系数、汇总、最佳模型）
  ✅ 可选择是否保存最佳模型为 .joblib 文件（方便后续载入）
"""

# ===== 导入库 =====
import numpy as np
import pandas as pd
from pathlib import Path
import json
from joblib import dump

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

# ===== 固定随机种子（让结果可复现） =====
SEED = 42

# ===== 自定义MAPE指标（百分比） =====
def mean_absolute_percentage_error(y_true, y_pred):
    """计算MAPE（百分比），避免除0错误"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

# ===== 读取CSV数据 =====
def read_xy_files(x_train_path, y_train_path, x_test_path, y_test_path):
    """读取预处理后的四个CSV文件"""
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0].to_numpy()
    y_test = pd.read_csv(y_test_path).iloc[:, 0].to_numpy()
    feature_names = X_train.columns.tolist()
    return X_train.to_numpy(), y_train, X_test.to_numpy(), y_test, feature_names

# ===== 训练并评估一个模型 =====
def train_and_eval(model_name, model, X_train, y_train, X_test, y_test, feature_names, outdir, save_model=False):
    """训练单个模型、计算指标、保存结果"""
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 1️⃣ 五折交叉验证 ---
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)

    # --- 2️⃣ 模型训练 ---
    model.fit(X_train, y_train)

    # --- 3️⃣ 预测 ---
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # --- 4️⃣ 计算评估指标 ---
    metrics = {
        "model": model_name,
        "cv_r2_mean": float(cv_mean),
        "cv_r2_std": float(cv_std),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "train_mape": float(mean_absolute_percentage_error(y_train, y_pred_train)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "test_mape": float(mean_absolute_percentage_error(y_test, y_pred_test)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
    }

    # --- 5️⃣ 保存预测结果（含误差） ---
    ape = np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
    pred_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred_test, "APE_percent": ape})
    pred_df.to_csv(outdir / f"{model_name}_predictions.csv", index=False)

    # --- 6️⃣ 保存系数（线性模型） ---
    coef = getattr(model, "coef_", np.zeros(len(feature_names)))
    intercept = getattr(model, "intercept_", 0)
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coef})
    coef_df.loc[len(coef_df)] = ["<intercept>", intercept]
    coef_df.to_csv(outdir / f"{model_name}_coefficients.csv", index=False)

    # --- 7️⃣ 保存指标JSON ---
    with open(outdir / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # --- 8️⃣ 可选保存模型 ---
    if save_model:
        dump(model, outdir / f"{model_name}.joblib")

    print(f"✅ {model_name} 训练完成：R²={metrics['test_r2']:.4f}, MAPE={metrics['test_mape']:.2f}%")
    return metrics

# ===== 主程序入口 =====
if __name__ == "__main__":
    # === 【需修改区域】 ===
    # 请在此处填写你的四个CSV文件路径（用 r"路径" 或 正斜杠 /）
    X_train_path = r"D:\MLDesignAl\TheFinal\Data\Element-UTS-ZL\output\Element-UTS-ZLX_train.csv"
    y_train_path = r"D:\MLDesignAl\TheFinal\Data\Element-UTS-ZL\output\Element-UTS-ZLy_train.csv"
    X_test_path  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS-ZL\output\Element-UTS-ZLX_test.csv"
    y_test_path  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS-ZL\output\Element-UTS-ZLy_test.csv"

    # 输出文件夹
    output_dir = Path(r"D:\MLDesignAl\TheFinal\LIN\Element-UTS-ZL\output")

    # 是否保存最佳模型（True 会生成 .joblib 文件）
    save_best_model = True

    # === 读取数据 ===
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_train_path, y_train_path, X_test_path, y_test_path
    )

    # === 定义三个模型 ===
    models = {
        "LIN": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=SEED),
        "Lasso": Lasso(alpha=1.0, random_state=SEED, max_iter=10000),
    }

    # === 循环训练每个模型 ===
    results = {}
    for name, model in models.items():
        results[name] = train_and_eval(name, model, X_train, y_train, X_test, y_test, feature_names, output_dir, save_model=False)

    # === 汇总对比 ===
    summary_df = pd.DataFrame(results).T.sort_values(by=["test_r2", "test_mape"], ascending=[False, True])
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path)
    print("\n📊 三个模型对比汇总：")
    print(summary_df[["test_r2", "test_mape", "test_rmse"]])

    # === 选出最佳模型 ===
    best_model_name = summary_df.index[0]
    print(f"\n🏆 最佳模型：{best_model_name}")

    # === 重新训练并保存最佳模型 ===
    if save_best_model:
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)
        dump(best_model, output_dir / f"best_{best_model_name}.joblib")
        print(f"已保存最佳模型到 {output_dir / f'best_{best_model_name}.joblib'}")
