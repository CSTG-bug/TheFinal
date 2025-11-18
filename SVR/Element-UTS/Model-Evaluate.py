#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVR 支持向量回归 —— 训练与评估脚本（含总保存开关）
================================================
功能：
  ✅ 读取你预处理后的四个 CSV（X_train, y_train, X_test, y_test）
  ✅ 标准化 + SVR（Pipeline，防止数据泄漏）
  ✅ 网格搜索（RBF/Linear 两种核）+ 5 折交叉验证，评分用 R²
  ✅ 指标：R²、MAPE%、MAE、RMSE（训练/测试）
  ✅ “不满意就不落盘”的总开关（MASTER_SAVE_SWITCH）+ 可选阈值（AUTO_SAVE_RULE）
  ✅ 可选保存：metrics.json / predictions.csv / cv_results.csv / best_model.joblib
  ✅ 中文逐段注释，便于理解与后续迁移
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# ============== 全局随机种子（保证可复现） ==============
SEED = 42

# ============== 总保存开关 & 自动保存阈值 ==============
# 1) 总开关：False=不落盘，True=允许落盘（仍会受阈值限制）
MASTER_SAVE_SWITCH = False   # ← 调参期建议 False；满意后改 True

# 2) 自动保存阈值（可选）：只有满足阈值才保存；不想限制就设为 None
AUTO_SAVE_RULE = {
    "min_test_r2": None,    # 例如 0.85
    "max_test_mape": None,  # 例如 5.0（百分比）
    "min_cv_r2": None       # 例如 0.80
}
# AUTO_SAVE_RULE = None  # ← 如需“总开关开了就一定保存”，把上面注释掉，用这一行

def _should_save(metrics: dict) -> (bool, str):
    """根据总开关与阈值判断是否保存。返回：(是否保存, 若不保存的原因)"""
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
    if thr is not None and metrics.get("cv_best_r2", -1) < thr:
        reasons.append(f"cv_best_r2={metrics['cv_best_r2']:.4f} < min_cv_r2={thr}")

    if reasons:
        return False, "；".join(reasons)
    return True, ""


# ============== 指标函数：MAPE（百分比，带 0 保护） ==============
def mean_absolute_percentage_error(y_true, y_pred) -> float:
    """计算 MAPE（百分比），对 y_true==0 做极小值保护避免除零。"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


# ============== 读取四个预处理 CSV ==============
def read_xy_files(x_train_path, y_train_path, x_test_path, y_test_path):
    """
    读取预处理输出的 4 个 CSV：
      - X_train.csv, X_test.csv：作为特征矩阵（保留列名仅用于调试/记录）
      - y_train.csv, y_test.csv：第一列为目标变量
    返回：
      X_train(np.ndarray), y_train(np.ndarray), X_test(np.ndarray), y_test(np.ndarray), feature_names(list)
    """
    X_train_df = pd.read_csv(x_train_path)
    X_test_df = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0].to_numpy().reshape(-1)
    y_test = pd.read_csv(y_test_path).iloc[:, 0].to_numpy().reshape(-1)
    feature_names = X_train_df.columns.tolist()
    return X_train_df.to_numpy(), y_train, X_test_df.to_numpy(), y_test, feature_names


# ============== 训练 + 评估 +（按需）保存输出 ==============
def train_eval_and_maybe_save(
    X_train, y_train, X_test, y_test, feature_names,
    outdir: Path,
    save_model: bool = True,
    save_cv_results: bool = True
):
    """
    核心流程：
      1) 建立 pipeline：标准化（防止数据泄漏）+ SVR
      2) 网格搜索两类核函数（rbf / linear），交叉验证评分 R²
      3) 用最佳参数拟合全训练集，评估训练/测试
      4) 只有在“should_save=True”时，才会落盘所有结果文件
    """
    # ---------- 1) 定义 Pipeline ----------
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])

    # ---------- 2) 参数网格 ----------
    # 说明：用“两组字典”分别对应 rbf 与 linear（避免无效参数）
    param_grid = [
        {   # RBF 核：需要调 gamma
            "svr__kernel": ["rbf"],
            "svr__C": [1, 3, 10, 30, 100, 300],
            "svr__epsilon": [0.005, 0.01, 0.05, 0.1, 0.2],
            "svr__gamma": ["scale", 0.1, 0.03, 0.01, 0.003, 0.001],
        },
        {   # 线性核：没有 gamma
            "svr__kernel": ["linear"],
            "svr__C": [1, 3, 10, 30, 100, 300],
            "svr__epsilon": [0.005, 0.01, 0.05, 0.1, 0.2],
        }
    ]

    # ---------- 3) 交叉验证配置 ----------
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---------- 4) 网格搜索 ----------
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        refit=True,
        verbose=1
    )
    grid.fit(X_train, y_train)

    # ---------- 5) 交叉验证最优结果 ----------
    best_params = grid.best_params_
    cv_best_r2 = float(grid.best_score_)
    cv_best_std = float(grid.cv_results_["std_test_score"][grid.best_index_])

    print("\n=== 最佳参数组合（交叉验证） ===")
    print(best_params)
    print(f"CV R²: {cv_best_r2:.4f} ± {cv_best_std:.4f}")

    # ---------- 6) 最优模型拟合与预测 ----------
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred_train = best_model.predict(X_train)
    y_pred_test  = best_model.predict(X_test)

    # ---------- 7) 计算指标 ----------
    metrics = {
        "model": "SVR",
        "best_params": best_params,
        "cv_best_r2": cv_best_r2,
        "cv_best_std": cv_best_std,
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

    print("\n=== 测试集性能（最佳参数） ===")
    print(f"R²   : {metrics['test_r2']:.4f}")
    print(f"MAPE : {metrics['test_mape']:.2f}%")
    print(f"MAE  : {metrics['test_mae']:.4f}")
    print(f"RMSE : {metrics['test_rmse']:.4f}")

    # ---------- 8) 判断是否保存 ----------
    allow_save, reason = _should_save(metrics)
    if not allow_save:
        print(f"\n⚠️ 未保存任何文件：{reason}")
        return metrics

    # ---------- 9) 达标 → 真正开始落盘 ----------
    outdir.mkdir(parents=True, exist_ok=True)

    # 9.1 metrics.json
    metrics_path = outdir / "SVR_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 9.2 predictions.csv（测试集）
    ape = np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
    pd.DataFrame({"y_test": y_test, "y_pred": y_pred_test, "APE_percent": ape}).to_csv(
        outdir / "SVR_predictions.csv", index=False
    )

    # 9.3 完整 CV 结果（便于复现/审稿）
    cv_df = pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False)
    if save_cv_results:
        cv_df.to_csv(outdir / "SVR_cv_results.csv", index=False)

    # 9.4 最佳模型
    if save_model:
        dump(best_model, outdir / "SVR_best_model.joblib")

    print(f"\n✅ 已保存到：{outdir.resolve()}")
    print("- SVR_metrics.json\n- SVR_predictions.csv")
    if save_cv_results: print("- SVR_cv_results.csv")
    if save_model: print("- SVR_best_model.joblib")

    return metrics


# ============== 主程序入口（直接在代码里设路径即可运行） ==============
if __name__ == "__main__":
    # ===== 【需修改区域】：把下面四个路径改为你的预处理 CSV =====
    X_train_path = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_train.csv"
    y_train_path = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_train.csv"
    X_test_path  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_test.csv"
    y_test_path  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_test.csv"

    # 输出目录（可自定义）
    output_dir = Path(r"D:\MLDesignAl\TheFinal\SVR\Element-UTS\output")

    # 是否保存模型 & CV 明细（只有在“允许保存”时才会生效）
    SAVE_MODEL = True
    SAVE_CV_RESULTS = True

    # ===== 读取数据 =====
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_train_path, y_train_path, X_test_path, y_test_path
    )

    # ===== 训练、评估（根据开关决定是否落盘） =====
    train_eval_and_maybe_save(
        X_train, y_train, X_test, y_test, feature_names,
        outdir=output_dir,
        save_model=SAVE_MODEL,
        save_cv_results=SAVE_CV_RESULTS
    )
