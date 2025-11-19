#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN 手动调参器（与 RF 手动调参同款风格）
========================================================
你要做的事：
  1) 修改顶部“【需修改】路径设置”的 4 个 CSV
  2) 选择缩放器 SCALER_CHOICE = 'standard' / 'minmax' / 'robust'
  3) 在 MANUAL_TRIALS 里写入若干你想试的 KNN 参数（每个是 dict）
  4) 运行，查看每组：Train/Test 四指标 + CV R²，并按 Test R² 排名
  5) 满意后把 MASTER_SAVE_SWITCH=True，再运行即可落盘最佳组
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# ==========【需修改】路径设置 ==========
X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_train.csv"
Y_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_train.csv"
X_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_test.csv"
Y_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_test.csv"

# 输出目录 & 保存控制
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\KNN\Element-UTS\output")
MASTER_SAVE_SWITCH = False   # 调参期建议 False；满意后改 True 落盘
AUTO_SAVE_RULE = {           # 达标才保存（不想用就设为 None）
    "min_test_r2": None,     # 例如 0.82
    "max_test_mape": None,   # 例如 8.0   （百分比）
    "min_cv_r2": None        # 例如 0.78
}
# AUTO_SAVE_RULE = None

# ========== 随机性与交叉验证 ==========
SEED = 42
CV_SPLITS = 5
CV_REPEATS = 2               # 想更稳可改 3（耗时更长）

# ========== 缩放器选择（很关键，KNN对尺度敏感） ==========
SCALER_CHOICE = 'standard'   # 'standard' / 'minmax' / 'robust'
def get_scaler(name: str):
    name = name.lower()
    if name == 'standard': return StandardScaler()
    if name == 'minmax':   return MinMaxScaler()
    if name == 'robust':   return RobustScaler()
    raise ValueError(f"Unknown scaler: {name}")

# ========== 你要【手动尝试】的 KNN 参数组（每个 dict 一组） ==========
# 字段说明：n_neighbors（K）、weights（uniform/distance）、p（1=曼哈顿,2=欧氏）、leaf_size（默认30）
# 可选地为某一组指定 "scaler": "minmax"/"robust" 覆盖全局缩放器
MANUAL_TRIALS = [
    # 基线（便于对照）
    {"n_neighbors": 7,  "weights": "uniform",  "p": 2,   "leaf_size": 30},

    # A. 扩大 k（平滑噪声；回归里偶数也可以）
    {"n_neighbors": 9,  "weights": "uniform",  "p": 2,   "leaf_size": 30},
    {"n_neighbors": 11, "weights": "uniform",  "p": 2,   "leaf_size": 30},
    {"n_neighbors": 13, "weights": "uniform",  "p": 2,   "leaf_size": 30},
    {"n_neighbors": 15, "weights": "uniform",  "p": 2,   "leaf_size": 30},
    {"n_neighbors": 17, "weights": "uniform",  "p": 2,   "leaf_size": 30},
    {"n_neighbors": 19, "weights": "uniform",  "p": 2,   "leaf_size": 30},

    # B. 轻度改距离度量（仍用 uniform），有时 p=1.5 更稳
    {"n_neighbors": 11, "weights": "uniform",  "p": 1,   "leaf_size": 30},
    {"n_neighbors": 13, "weights": "uniform",  "p": 1.5, "leaf_size": 30},
    {"n_neighbors": 15, "weights": "uniform",  "p": 1.5, "leaf_size": 30},

    # C. 缩放器对比（只在较大 k 上测一遍 MinMax）
    {"n_neighbors": 13, "weights": "uniform",  "p": 2,   "leaf_size": 30, "scaler": "minmax"},
    {"n_neighbors": 15, "weights": "uniform",  "p": 2,   "leaf_size": 30, "scaler": "minmax"},
]

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

def _should_save(metrics: dict) -> (bool, str):
    """根据总开关与阈值判断是否保存。"""
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

# ========== 单组试验：构建 Pipeline → 交叉验证 → 训练/评估 ==========
def evaluate_knn_once(X_train, y_train, X_test, y_test, feature_names,
                      params: dict, scaler_name: str, cv_splits=5, cv_repeats=2):
    """
    用给定 params 跑一遍 KNN，返回 (metrics, model, y_pred_test)。
    说明：
      - 采用 Pipeline([scaler, KNN])，避免数据泄漏（缩放器只用训练折拟合）
      - params 支持 n_neighbors / weights / p / leaf_size / metric（默认 minkowski）
      - 可在 params 里放 "scaler" 覆盖全局缩放器
    """
    this_scaler = params.get("scaler", scaler_name)
    pipe = Pipeline([
        ("scaler", get_scaler(this_scaler)),
        ("knn", KNeighborsRegressor())
    ])

    # 把手动参数塞进 pipeline 的 knn 步骤
    knn_params = {
        "knn__n_neighbors": int(params.get("n_neighbors", 7)),
        "knn__weights": params.get("weights", "distance"),
        "knn__p": int(params.get("p", 2)),
        "knn__leaf_size": int(params.get("leaf_size", 30)),
        "knn__metric": params.get("metric", "minkowski")
    }
    pipe.set_params(**knn_params)

    # 交叉验证（R²），更稳健
    cv = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=SEED)
    cv_scores = cross_val_score(pipe, X_train, y_train, scoring="r2", cv=cv, n_jobs=-1)
    cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())

    # 拟合全训练集并预测
    pipe.fit(X_train, y_train)
    y_pred_tr = pipe.predict(X_train)
    y_pred_te = pipe.predict(X_test)

    # 训练/测试指标
    metrics = {
        "scaler": this_scaler,
        "n_neighbors": knn_params["knn__n_neighbors"],
        "weights": knn_params["knn__weights"],
        "p": knn_params["knn__p"],
        "leaf_size": knn_params["knn__leaf_size"],
        "metric": knn_params["knn__metric"],
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

# ========== 落盘（只有在允许保存时才会调用） ==========
def save_bundle(outdir: Path, tag: str, metrics: dict, model, y_test, y_pred_test):
    """保存：metrics.json / predictions.csv / 最佳模型.joblib"""
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"KNN_{tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    ape = np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
    pd.DataFrame({"y_test": y_test, "y_pred": y_pred_test, "APE_percent": ape}).to_csv(
        outdir / f"KNN_{tag}_predictions.csv", index=False
    )

    dump(model, outdir / f"KNN_{tag}_best_model.joblib")

# ========== 主程序：一次跑多组手动参数，打印排名表 ==========
if __name__ == "__main__":
    # 读取数据
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH
    )

    # 数据概况（只打印一次）
    print("\n=== 数据概况 ===")
    print(f"n_train / n_test / n_features : {len(y_train)} / {len(y_test)} / {len(feature_names)}")
    print(f"y_train: mean={np.mean(y_train):.3f}, std={np.std(y_train):.3f}, "
          f"min={np.min(y_train):.3f}, max={np.max(y_train):.3f}")
    print(f"全局缩放器（默认）：{SCALER_CHOICE}")

    # 跑多组手写参数
    packs = []
    print("\n================ KNN 手动调参 ================\n")
    for i, p in enumerate(MANUAL_TRIALS, 1):
        this_scaler = p.get("scaler", SCALER_CHOICE)
        print(f"[{i}/{len(MANUAL_TRIALS)}] 训练参数：{{scaler: {this_scaler}, n_neighbors: {p.get('n_neighbors')}, "
              f"weights: {p.get('weights')}, p: {p.get('p')}, leaf_size: {p.get('leaf_size', 30)}}}")
        metrics, model, y_pred_te = evaluate_knn_once(
            X_train, y_train, X_test, y_test, feature_names,
            params=p, scaler_name=SCALER_CHOICE,
            cv_splits=CV_SPLITS, cv_repeats=CV_REPEATS
        )
        row = {
            "idx": i,
            "scaler": metrics["scaler"],
            "k": metrics["n_neighbors"],
            "weights": metrics["weights"],
            "p": metrics["p"],
            "leaf": metrics["leaf_size"],
            "test_r2": metrics["test_r2"],
            "cv_r2": metrics["cv_r2"],
            "cv_std": metrics["cv_std"],
            "rmse": metrics["test_rmse"],
            "mape%": metrics["test_mape"]
        }
        packs.append((row, metrics, model, y_pred_te))

    # 排名打印（按 Test R²）
    table = pd.DataFrame([r for r, *_ in packs]).sort_values("test_r2", ascending=False)
    print("\n=== 排名（按 Test R²）===\n")
    print(table.to_string(index=False, justify="center", float_format=lambda x: f"{x:.6f}"))

    # 选最优并给出建议
    best_row, best_metrics, best_model, best_pred = max(packs, key=lambda t: t[0]["test_r2"])
    print("\n=== 最优组合（当前轮次） ===")
    print(best_row)
    print("\n提示：若 Train R² 高而 Test/CV 低 → 说明方差大，可：")
    print("  • 增大 k（如 7→9→11→15），或改用 weights='distance'")
    print("  • 用 MinMax/Robust 缩放器再试一遍（有异常值时 Robust 常更稳）")
    print("若 Train/Test 都偏低 → 说明偏差大，可：")
    print("  • 减小 k（如 7→5→3），或改 p=1（曼哈顿），有时能提升")
    print("  • 重新检查特征选择/工程（元素物性衍生特征常对 KNN 有帮助）")

    # 是否保存
    allow_save, reason = _should_save(best_metrics)
    if not allow_save:
        print(f"\n⚠️ 未保存任何文件：{reason}")
    else:
        tag = (f"top1_scaler{best_row['scaler']}_k{best_row['k']}_w{best_row['weights']}"
               f"_p{best_row['p']}_leaf{best_row['leaf']}")
        save_bundle(OUTPUT_DIR, tag, best_metrics, best_model, y_test, best_pred)
        print(f"\n✅ 已保存到：{OUTPUT_DIR.resolve()}")
        print("- KNN_*_metrics.json / KNN_*_predictions.csv / KNN_*_best_model.joblib")
