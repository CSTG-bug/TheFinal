#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决策树回归（单模型版）自动训练与评估脚本
================================================
功能：
  ✅ 从你指定的 4 个预处理 CSV 文件读取数据（X_train, y_train, X_test, y_test）
  ✅ 网格搜索（GridSearchCV）+ 5 折交叉验证（KFold）寻找最佳超参数
  ✅ 计算并保存指标：R²、MAPE、MAE、RMSE（训练集与测试集）
  ✅ 保存测试集预测结果（含 APE%）
  ✅ 保存特征重要性（feature_importances_）
  ✅ 可选：导出树的文本规则（便于可解释）与保存最佳模型 .joblib
  ✅ 全流程中文注释，便于你理解每一步

说明：
  - 决策树容易过拟合，建议后期与随机森林/梯度提升对比；本脚本仅做“决策树单模”。
  - 训练集是否标准化对树模型影响不大（树基于阈值划分），可直接使用你预处理后的 CSV。
  - 思路参考并升级自你之前的脚本（固定随机种子、核心剪枝参数调优）。 ← 保留优点
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# sklearn 组件
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# 固定随机种子（保证结果可复现）
SEED = 42


# ===== MAPE 指标（与前面脚本保持一致，对 y=0 做保护）=====
def mean_absolute_percentage_error(y_true, y_pred) -> float:
    """计算 MAPE（百分比），避免 y_true=0 导致除零"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


# ===== 读取四个预处理 CSV（与之前风格一致）=====
def read_xy_files(x_train_path, y_train_path, x_test_path, y_test_path):
    """
    从四个 CSV 读取数据：
      - X_train, X_test: 保留列名（用于后面特征重要性输出）
      - y_train, y_test: 读取第一列为目标变量
    返回：X_train(np.array), y_train(np.array), X_test(np.array), y_test(np.array), feature_names(list)
    """
    X_train_df = pd.read_csv(x_train_path)
    X_test_df = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0].to_numpy().reshape(-1)
    y_test = pd.read_csv(y_test_path).iloc[:, 0].to_numpy().reshape(-1)

    feature_names = X_train_df.columns.tolist()
    return X_train_df.to_numpy(), y_train, X_test_df.to_numpy(), y_test, feature_names


# ===== 训练 + 评估 + 输出保存（单次，以最佳参数为准）=====
def train_eval_and_save(
    X_train, y_train, X_test, y_test, feature_names,
    outdir: Path,
    save_model: bool = True,
    save_rules: bool = True
):
    """
    - 使用网格搜索 + 5 折交叉验证挑选最佳超参数
    - 用最佳模型拟合全训练集，评估训练与测试
    - 保存：metrics.json、predictions.csv、importances.csv
    - 可选：保存最佳模型 .joblib 与导出树规则文本
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 定义超参数网格（核心剪枝相关），防止过拟合
    #   - max_depth：限制树深度
    #   - min_samples_split：节点评分裂所需最小样本数
    #   - min_samples_leaf：叶子最小样本数
    #   - max_features：每次分裂考虑的特征数（sqrt/None）
    param_grid = {
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", None],
    }

    # 2) 基模型（记得设置 random_state，保证可复现）
    base_model = DecisionTreeRegressor(random_state=SEED)

    # 3) 交叉验证与网格搜索（以 R² 为评分标准）
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # 4) 交叉验证结果（最优参数与 CV R²）
    best_params = grid.best_params_
    cv_best_r2 = float(grid.best_score_)
    print("\n=== 最佳参数组合 ===")
    print(best_params)
    print(f"交叉验证最佳R²: {cv_best_r2:.4f}")

    # 5) 用最佳参数得到最终模型并拟合
    best_dt = grid.best_estimator_
    best_dt.fit(X_train, y_train)

    # 6) 训练集与测试集预测
    y_pred_train = best_dt.predict(X_train)
    y_pred_test = best_dt.predict(X_test)

    # 7) 计算训练/测试指标
    metrics = {
        "model": "DecisionTreeRegressor",
        "best_params": best_params,
        "cv_best_r2": cv_best_r2,
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "train_mape": mean_absolute_percentage_error(y_train, y_pred_train),
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "test_mape": mean_absolute_percentage_error(y_test, y_pred_test),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "random_state": SEED
    }

    # 8) 保存指标 JSON
    metrics_path = outdir / "DT_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 9) 保存测试集预测（含绝对百分比误差 APE%）
    ape = np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
    pred_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred_test, "APE_percent": ape})
    pred_df.to_csv(outdir / "DT_predictions.csv", index=False)

    # 10) 保存特征重要性（可解释性，越大影响越大）
    importances = getattr(best_dt, "feature_importances_", np.zeros(len(feature_names)))
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df.sort_values("importance", ascending=False).to_csv(outdir / "DT_importances.csv", index=False)

    # 11) 可选：导出文本规则（人类可读的树结构）
    if save_rules:
        try:
            rules = export_text(best_dt, feature_names=feature_names)
            (outdir / "DT_rules.txt").write_text(rules, encoding="utf-8")
        except Exception as e:
            # 某些情况下（特征名包含特殊字符或过长）可能失败，做容错
            print(f"导出规则失败（不影响训练结果）：{e}")

    # 12) 可选：保存最佳模型（后续直接加载预测）
    if save_model:
        dump(best_dt, outdir / "DT_best_model.joblib")

    # 13) 控制台摘要
    print("\n=== 测试集性能（最佳参数） ===")
    print(f"R²:   {metrics['test_r2']:.4f}")
    print(f"MAPE: {metrics['test_mape']:.2f}%")
    print(f"MAE:  {metrics['test_mae']:.4f}")
    print(f"RMSE: {metrics['test_rmse']:.4f}")
    print(f"\n已保存：\n- {metrics_path}\n- DT_predictions.csv\n- DT_importances.csv"
          f"\n- {'DT_rules.txt' if save_rules else '(未导出规则)'}\n- {'DT_best_model.joblib' if save_model else '(未保存模型)'}")


# ===== 主程序（在代码里直接指定路径并运行）=====
if __name__ == "__main__":
    # === 【需修改区域】把下面四个路径改成你的预处理结果 CSV ===
    X_train_path = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_train.csv"
    y_train_path = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_train.csv"
    X_test_path  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_test.csv"
    y_test_path  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_test.csv"

    # 输出目录（可改）
    output_dir = Path(r"D:\MLDesignAl\TheFinal\DT\Element-UTS\output")

    # 是否保存模型与树规则文本
    SAVE_MODEL = True
    SAVE_RULES = True

    # === 读取数据 ===
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_train_path, y_train_path, X_test_path, y_test_path
    )

    # === 训练、评估并保存结果 ===
    train_eval_and_save(
        X_train, y_train, X_test, y_test, feature_names,
        outdir=output_dir,
        save_model=SAVE_MODEL,
        save_rules=SAVE_RULES
    )
