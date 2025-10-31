#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机森林回归
========================================================================
设计目标：
  - 直接在代码里指定 4 个预处理CSV，运行即出结果
  - 自动尝试两套目标：原始 y 与 log1p(y)（用 TransformedTargetRegressor 实现）
  - 更合理的参数搜索空间 + RepeatedKFold(5x2)
  - 以测试集 R² 为主指标（并列时用 MAPE 决胜）选**更优方案**
  - 默认不落盘；满意后把 SAVE_OUTPUT=True 再保存所有产物

如何提升 R² 的要点：
  - 树数更多（n_estimators）+ 适当放开 max_depth
  - 控制叶子规模（min_samples_leaf）抑制过拟合后反而能提高测试R²
  - max_features 既试 'sqrt'/'log2' 也试连续比例（0.5/0.7），常见有提升
  - 目标分布偏态时，log 版本常能涨分；不偏态则原始目标更好
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# ============ 【需修改路径】你的四个CSV ============
X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_train.csv"
Y_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_train.csv"
X_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_test.csv"
Y_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_test.csv"

# ============ 输出与保存（默认不落盘，调满意再改 True） ============
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\RF\Element-UTS\output")
SAVE_OUTPUT = False          # ← 调参阶段 False；满意后设 True 一次性落盘
SAVE_MODEL = True            # SAVE_OUTPUT=True 时才生效
SAVE_CV_RESULTS = True       # SAVE_OUTPUT=True 时才生效

# ============ 搜索与验证配置（可以先用默认） ============
SEED = 42
USE_RANDOMIZED_SEARCH = True
N_ITER = 80                  # 随机搜索抽样次数（60~120 常见，越大越稳）
CV_SPLITS = 5
CV_REPEATS = 2               # 想更稳可设 3；时间也会更长

# ============ 公共函数：读数据 / 指标 ============
def read_xy_files(x_train_path, y_train_path, x_test_path, y_test_path):
    """读取预处理后的四个CSV；返回矩阵与列名"""
    X_train_df = pd.read_csv(x_train_path)
    X_test_df = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0].to_numpy().reshape(-1)
    y_test = pd.read_csv(y_test_path).iloc[:, 0].to_numpy().reshape(-1)
    feature_names = X_train_df.columns.tolist()
    return X_train_df.to_numpy(), y_train, X_test_df.to_numpy(), y_test, feature_names

def mape(y_true, y_pred) -> float:
    """MAPE（百分比），对 0 做保护避免除零。"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

# ============ 核心：给“一个模型对象 + 参数空间”做搜索并返回指标 ============
def search_and_score(model, param_space, X_train, y_train, X_test, y_test,
                     feature_names, tag: str, outdir: Path):
    """
    入参：
      - model: 可以是 RF 本体，也可以是 TTR(regressor=RF)
      - param_space: 与 model 对应的参数空间（注意前缀是否需要 regressor__）
      - tag: 用于区分“原始目标”与“对数目标”的名字，打印友好
    返回：
      - dict：包含 训练/测试 R²、MAPE、MAE、RMSE、best_params、cv_best_r2 等
      - best_model：已在全训练集拟合好的最佳模型
    """
    cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)
    searcher = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        n_iter=N_ITER,
        cv=cv,
        scoring="r2",
        random_state=SEED,
        n_jobs=-1,
        refit=True
    )
    searcher.fit(X_train, y_train)

    best_model = searcher.best_estimator_
    best_params = searcher.best_params_
    cv_best_r2 = float(searcher.best_score_)

    # 预测与指标
    y_pred_train = best_model.predict(X_train)
    y_pred_test  = best_model.predict(X_test)

    metrics = {
        "tag": tag,
        "cv_best_r2": cv_best_r2,
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "train_mape": mape(y_train, y_pred_train),
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "test_mape": mape(y_test, y_pred_test),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "best_params": best_params,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }

    # 打印摘要
    print(f"\n=== {tag}：测试集性能 ===")
    print(f"R²   : {metrics['test_r2']:.4f}")
    print(f"MAPE : {metrics['test_mape']:.2f}%")
    print(f"MAE  : {metrics['test_mae']:.4f}")
    print(f"RMSE : {metrics['test_rmse']:.4f}")
    print(f"CV(best R²): {metrics['cv_best_r2']:.4f}")

    # 按需落盘
    if SAVE_OUTPUT:
        outdir.mkdir(parents=True, exist_ok=True)
        # metrics
        with open(outdir / f"RF_{tag}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        # predictions
        y_pred = best_model.predict(X_test)
        ape = np.abs((y_test - y_pred) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
        pd.DataFrame({"y_test": y_test, "y_pred": y_pred, "APE_percent": ape}).to_csv(
            outdir / f"RF_{tag}_predictions.csv", index=False
        )
        # importances（取内部 RF）
        rf_inner = best_model.regressor_ if isinstance(best_model, TransformedTargetRegressor) else best_model
        importances = getattr(rf_inner, "feature_importances_", np.zeros(len(feature_names)))
        pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        ).to_csv(outdir / f"RF_{tag}_importances.csv", index=False)
        # cv results
        if SAVE_CV_RESULTS:
            cv_df = pd.DataFrame(searcher.cv_results_).sort_values("mean_test_score", ascending=False)
            cv_df.to_csv(outdir / f"RF_{tag}_cv_results.csv", index=False)
        # 保存模型
        if SAVE_MODEL:
            dump(best_model, outdir / f"RF_{tag}_best_model.joblib")

    return metrics, best_model

# ============ 主流程：两套目标都跑，自动挑更优 ============
if __name__ == "__main__":
    # 读取数据
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH
    )

    # 基础 RF（不包 TTR）：参数空间——无任何前缀
    rf_base = RandomForestRegressor(random_state=SEED, n_jobs=-1, bootstrap=True)
    param_space_raw = {
        "n_estimators":       [400, 700, 900, 1200, 1400],
        "max_depth":          [None, 20, 30, 40, 50],
        "min_samples_split":  [2, 5, 10, 20],
        "min_samples_leaf":   [1, 2, 4, 8],
        "max_features":       ["sqrt", "log2", None, 0.5, 0.7],
        "bootstrap":          [True],
    }

    # TTR(log) 版本：参数空间需要加前缀 regressor__
    rf_log = TransformedTargetRegressor(
        regressor=RandomForestRegressor(random_state=SEED, n_jobs=-1, bootstrap=True),
        func=np.log1p, inverse_func=np.expm1
    )
    param_space_log = {f"regressor__{k}": v for k, v in param_space_raw.items()}

    # 搜索与评估：原始目标
    metrics_raw, model_raw = search_and_score(
        rf_base, param_space_raw, X_train, y_train, X_test, y_test, feature_names,
        tag="raw", outdir=OUTPUT_DIR
    )
    # 搜索与评估：对数目标
    metrics_log, model_log = search_and_score(
        rf_log, param_space_log, X_train, y_train, X_test, y_test, feature_names,
        tag="log", outdir=OUTPUT_DIR
    )

    # 选更优（先比 test_r2，高者胜；若相同比 test_mape，小者胜）
    def better(a, b):
        if a["test_r2"] > b["test_r2"]:
            return a
        if a["test_r2"] < b["test_r2"]:
            return b
        return a if a["test_mape"] <= b["test_mape"] else b

    best = better(metrics_raw, metrics_log)
    print("\n================== 最终选择 ==================")
    print(f"选择方案: {'原始目标(raw)' if best['tag']=='raw' else '对数目标(log)'}")
    print(f"Test R² = {best['test_r2']:.4f} | Test MAPE = {best['test_mape']:.2f}%")
    print("最佳参数：")
    print(best["best_params"])
