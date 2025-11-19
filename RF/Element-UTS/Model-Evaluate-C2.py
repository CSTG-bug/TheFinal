#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机森林回归（RF）
======================================================================
特性：
  - 直接在代码里指定 4 个预处理 CSV（X_train/X_test/y_train/y_test）
  - 默认不保存任何文件（调参阶段更干净）；满意后切换 SAVE_OUTPUT=True 再落盘
  - RandomizedSearchCV（默认）或 GridSearchCV（二选一）
  - RepeatedKFold = 5x2 次交叉验证，更稳
  - 更宽的参数搜索空间：n_estimators / max_depth / max_features / min_samples_* / bootstrap
  - 可选目标对数变换（对偏态/重尾的目标常有用）
  - 训练/测试：R²、MAPE、MAE、RMSE；可选保存最佳模型与CV明细、稳定性复跑
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump
from math import exp

# ===================== 全局设置（可改） =====================
# 【需修改区域】：四个CSV路径
X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_train.csv"
Y_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_train.csv"
X_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_test.csv"
Y_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_test.csv"

# 输出目录 & 保存开关（默认不落盘）
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\RF\Element-UTS\output")
SAVE_OUTPUT = False        # ← 调参阶段建议 False；满意后改 True
SAVE_MODEL = True          # 保存最佳模型 .joblib（在 SAVE_OUTPUT=True 时生效）
SAVE_CV_RESULTS = True     # 保存完整CV结果（在 SAVE_OUTPUT=True 时生效）
STABILITY_RUNS = 0         # 稳定性复跑次数（0=不复跑；建议满意后开 5 或 10）

# 搜索方式：随机/网格（随机更快、空间更大）
USE_RANDOMIZED_SEARCH = True
N_ITER_RANDOM_SEARCH = 50  # 随机搜索抽样次数（越大越稳，越慢）

# 交叉验证：5 折重复 2 次（更稳）
CV_SPLITS = 5
CV_REPEATS = 2

# 目标对数变换（对偏态目标常见增益；不想用就设 False）
DO_LOG_TARGET = True

# 固定随机种子
SEED = 42

# ===================== 指标函数 =====================
def mean_absolute_percentage_error(y_true, y_pred) -> float:
    """MAPE（百分比），对 0 做保护避免除零。"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

# ===================== 读取数据 =====================
def read_xy_files(x_train_path, y_train_path, x_test_path, y_test_path):
    """
    读取预处理输出的四个 CSV：
      - X_train/X_test：保留列名（重要性输出用）
      - y_train/y_test：第一列为目标
    返回：X_train(np.ndarray), y_train(np.ndarray), X_test(np.ndarray), y_test(np.ndarray), feature_names(list)
    """
    X_train_df = pd.read_csv(x_train_path)
    X_test_df = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0].to_numpy().reshape(-1)
    y_test = pd.read_csv(y_test_path).iloc[:, 0].to_numpy().reshape(-1)
    feature_names = X_train_df.columns.tolist()
    return X_train_df.to_numpy(), y_train, X_test_df.to_numpy(), y_test, feature_names

# ===================== 训练 & 评估 =====================
def train_eval_rf(
    X_train, y_train, X_test, y_test, feature_names,
    outdir: Path
):
    """
    1) 选择搜索器（随机/网格），定义参数空间
    2) RepeatedKFold 交叉验证
    3) 搜索最佳参数 → 拟合全训练集
    4) 计算训练/测试指标（R²、MAPE、MAE、RMSE）
    5) 视开关决定是否落盘（metrics / predictions / importances / cv_results / model）
    """

    # ---- 参数空间（随机搜索用列表做分布，方便控制）----
    param_space = {
        "n_estimators":   [300, 500, 700, 900, 1200],
        "max_depth":      [None, 15, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf":  [1, 2, 4, 8],
        "max_features":   ["sqrt", "log2", None, 0.7, 0.5],
        "bootstrap":      [True],  # OOB 评估只在 bootstrap=True 生效
        "random_state":   [SEED],  # 固定随机性（搜索器不会改它）
        "n_jobs":         [-1],    # 并行
    }

    # ---- 构建基模型 ----
    base_rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, bootstrap=True, oob_score=True)

    # ---- 可选目标变换：log1p ↔ expm1（避免负值问题建议先确认 y≥0）----
    if DO_LOG_TARGET:
        # TransformedTargetRegressor 会在内部对 y 做 log1p，预测后再 expm1 还原
        model = TransformedTargetRegressor(
            regressor=base_rf,
            func=np.log1p,
            inverse_func=np.expm1
        )
        # 需要把参数名加前缀 'regressor__'
        param_space = {f"regressor__{k}": v for k, v in param_space.items()}
        oob_attr = "regressor__oob_score_"  # 仅用于提示
    else:
        model = base_rf
        oob_attr = "oob_score_"

    # ---- 交叉验证器（更稳）----
    cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)

    # ---- 搜索器 ----
    if USE_RANDOMIZED_SEARCH:
        searcher = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_space,
            n_iter=N_ITER_RANDOM_SEARCH,
            cv=cv,
            scoring="r2",
            random_state=SEED,
            n_jobs=-1,
            refit=True
        )
    else:
        searcher = GridSearchCV(
            estimator=model,
            param_grid=param_space,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            refit=True
        )

    # ---- 搜索 + 拟合 ----
    searcher.fit(X_train, y_train)

    # ---- 提取最佳模型与参数 ----
    best_model = searcher.best_estimator_
    best_params = searcher.best_params_
    cv_best_r2 = float(searcher.best_score_)

    # ---- 最终拟合（searcher.refit=True 已做过；这里确保拿到训练好的对象）----
    # 预测
    y_pred_train = best_model.predict(X_train)
    y_pred_test  = best_model.predict(X_test)

    # ---- 指标 ----
    metrics = {
        "model": "RandomForestRegressor",
        "use_randomized_search": USE_RANDOMIZED_SEARCH,
        "cv_scheme": f"RepeatedKFold({CV_SPLITS}x{CV_REPEATS})",
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
        "log_target": DO_LOG_TARGET
    }

    # ---- 打印 OOB 分数（仅在 bootstrap=True 时可用；在 TTR 包裹下需取内部 rf）----
    try:
        if DO_LOG_TARGET:
            rf_inner = best_model.regressor_
        else:
            rf_inner = best_model
        if getattr(rf_inner, "oob_score_", None) is not None:
            print(f"OOB R²: {rf_inner.oob_score_:.4f}")
            metrics["oob_r2"] = float(rf_inner.oob_score_)
    except Exception:
        pass

    # ---- 控制台摘要 ----
    print("\n=== RF 优化版：测试集性能 ===")
    print(f"R²   : {metrics['test_r2']:.4f}")
    print(f"MAPE : {metrics['test_mape']:.2f}%")
    print(f"MAE  : {metrics['test_mae']:.4f}")
    print(f"RMSE : {metrics['test_rmse']:.4f}")
    print(f"CV(best R²): {metrics['cv_best_r2']:.4f}")
    if "oob_r2" in metrics:
        print(f"OOB R²     : {metrics['oob_r2']:.4f}")

    # ====== 新增：把“最终采用的最佳参数”打印出来 ======
    def _strip_prefix_dict(d: dict, prefix: str) -> dict:
        """去掉字典键名中的指定前缀，便于阅读（TTR包装下键是 regressor__xxx）"""
        out = {}
        for k, v in d.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
            else:
                out[k] = v
        return out

    # 原始 best_params（严格复现用）
    print("\n--- best_params（原始键名，用于完全复现）---")
    print(best_params)

    # 可读版（去掉 regressor__ 前缀的 RF 参数）
    readable_params = _strip_prefix_dict(best_params, "regressor__") if DO_LOG_TARGET else best_params
    # 为了更直观，挑选常用关键超参按顺序打印；其余保留在 readable_params 里
    key_order = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
                 "max_features", "bootstrap", "oob_score", "random_state", "n_jobs"]
    print("\n--- 最终采用的 RF 参数（可读版）---")
    for k in key_order:
        if k in readable_params:
            print(f"{k:>18}: {readable_params[k]}")
    # 打印其余可能存在但不常用的键
    others = {k: v for k, v in readable_params.items() if k not in key_order}
    if others:
        print("其它参数：", others)


    # ---- 只在 SAVE_OUTPUT=True 时落盘 ----
    if SAVE_OUTPUT:
        outdir.mkdir(parents=True, exist_ok=True)

        # 1) metrics.json
        with open(outdir / "RF_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # 2) predictions.csv
        ape = np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
        pd.DataFrame({"y_test": y_test, "y_pred": y_pred_test, "APE_percent": ape}).to_csv(
            outdir / "RF_predictions.csv", index=False
        )

        # 3) importances.csv（特征重要性）
        importances = getattr(rf_inner, "feature_importances_", np.zeros(len(feature_names)))
        pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        ).to_csv(outdir / "RF_importances.csv", index=False)

        # 4) 完整 CV 结果（便于复现/审稿）
        if SAVE_CV_RESULTS:
            cv_df = pd.DataFrame(searcher.cv_results_).sort_values("mean_test_score", ascending=False)
            cv_df.to_csv(outdir / "RF_cv_results.csv", index=False)

        # 5) 保存最佳模型
        if SAVE_MODEL:
            dump(best_model, outdir / "RF_best_model.joblib")

        # 6) 稳定性复跑（不同 random_state）
        if STABILITY_RUNS and STABILITY_RUNS > 0:
            rows = []
            # 从 best_params 中抽出 rf 的参数字典
            def strip_prefix(d, prefix):
                res = {}
                for k, v in d.items():
                    if k.startswith(prefix):
                        res[k[len(prefix):]] = v
                return res

            if DO_LOG_TARGET:
                rf_params = strip_prefix(best_params, "regressor__")
            else:
                rf_params = strip_prefix(best_params, "rf__") if "rf__n_estimators" in best_params else best_params

            for i in range(STABILITY_RUNS):
                rf = RandomForestRegressor(**rf_params, random_state=SEED + i, n_jobs=-1)
                if DO_LOG_TARGET:
                    ttr = TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1)
                    ttr.fit(X_train, y_train)
                    y_pred = ttr.predict(X_test)
                else:
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                rows.append({"run_id": i + 1, "r2": float(r2_score(y_test, y_pred)),
                             "mape": mean_absolute_percentage_error(y_test, y_pred)})
            pd.DataFrame(rows).to_csv(outdir / "RF_stability_runs.csv", index=False)

    return metrics

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 读取数据
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH
    )

    # 训练 + 评估（默认不保存任何文件）
    metrics = train_eval_rf(X_train, y_train, X_test, y_test, feature_names, outdir=OUTPUT_DIR)