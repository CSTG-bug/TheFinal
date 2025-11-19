#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVR 手动调参器（与 RF/KNN 手动调参同款风格）
================================================
你要做的事：
  1) 修改“【需修改】路径设置”的 4 个 CSV
  2) 选择缩放器 SCALER_CHOICE = 'standard' / 'minmax' / 'robust' / 'identity'
     （如果 CSV 已经缩放过，就用 'identity' 避免重复缩放）
  3) 在 MANUAL_TRIALS 里手写若干 SVR 参数组合（每个是 dict）
  4) 运行，查看每组：Train/Test 四指标 + CV R²，并按 Test R² 排名
  5) 满意后把 MASTER_SAVE_SWITCH=True，再运行即可落盘最佳组
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# ==========【需修改】路径设置 ==========
X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_train.csv"
Y_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_train.csv"
X_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_test.csv"
Y_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_test.csv"

# 输出目录 & 保存控制
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\SVR\Element-UTS\output")
MASTER_SAVE_SWITCH = False   # 调参期建议 False；满意后改 True 落盘
AUTO_SAVE_RULE = {           # 达标才保存（不想用就设为 None）
    "min_test_r2": None,     # 例如 0.72
    "max_test_mape": None,   # 例如 15.5
    "min_cv_r2":   None      # 例如 0.68
}
# AUTO_SAVE_RULE = None

# ========== 随机性与交叉验证 ==========
SEED = 42
CV_SPLITS = 5
CV_REPEATS = 2               # 想更稳可改 3（耗时更长）

# ========== 缩放器选择 ==========
SCALER_CHOICE = 'identity'   # 'standard' / 'minmax' / 'robust' / 'identity'
def get_scaler(name: str):
    n = name.lower()
    if n == 'standard': return StandardScaler()
    if n == 'minmax':   return MinMaxScaler()
    if n == 'robust':   return RobustScaler()
    if n == 'identity': return FunctionTransformer(lambda X: X, validate=False)
    raise ValueError(f"Unknown scaler: {name}")

# ========== 你要【手动尝试】的 SVR 参数组 ==========
# 说明：
#   - 对 RBF：需要 kernel='rbf'，建议给 C / epsilon / gamma
#   - 对 Linear：kernel='linear'，不要给 gamma（给了也会被忽略）
#   - 下面以你的网格最佳（rbf, C=300, eps=0.3, gamma=0.03）为中位点做“扇形扩散”
MANUAL_TRIALS = [
    # 基线（便于对照）
    {"kernel": "rbf", "C": 600, "epsilon": 0.30, "gamma": 0.03, "scaler": "identity"},

    # A. C 细化（上下轻微扩张）
    {"kernel": "rbf", "C": 500, "epsilon": 0.30, "gamma": 0.03, "scaler": "identity"},
    {"kernel": "rbf", "C": 700, "epsilon": 0.30, "gamma": 0.03, "scaler": "identity"},
    {"kernel": "rbf", "C": 900, "epsilon": 0.30, "gamma": 0.03, "scaler": "identity"},

    # B. epsilon 细化（抑制/放宽噪声带）
    {"kernel": "rbf", "C": 600, "epsilon": 0.25, "gamma": 0.03, "scaler": "identity"},
    {"kernel": "rbf", "C": 600, "epsilon": 0.35, "gamma": 0.03, "scaler": "identity"},

    # C. gamma 细化（核宽度微调）
    {"kernel": "rbf", "C": 600, "epsilon": 0.30, "gamma": 0.02, "scaler": "identity"},
    {"kernel": "rbf", "C": 600, "epsilon": 0.30, "gamma": 0.04, "scaler": "identity"},

    # D. 交互（把看起来更稳的方向做两两组合）
    {"kernel": "rbf", "C": 700, "epsilon": 0.25, "gamma": 0.03, "scaler": "identity"},
    {"kernel": "rbf", "C": 700, "epsilon": 0.30, "gamma": 0.02, "scaler": "identity"},
    {"kernel": "rbf", "C": 500, "epsilon": 0.35, "gamma": 0.04, "scaler": "identity"},

    # E. 对照：gamma='scale' 在 C=600 附近再看一眼
    {"kernel": "rbf", "C": 600, "epsilon": 0.30, "gamma": "scale", "scaler": "identity"},
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
def evaluate_svr_once(X_train, y_train, X_test, y_test, feature_names,
                      params: dict, scaler_name: str, cv_splits=5, cv_repeats=2):
    """
    用给定 params 跑一遍 SVR，返回 (metrics, model, y_pred_test)。
    说明：
      - 采用 Pipeline([scaler, SVR])，避免数据泄漏（缩放器仅用训练折拟合）
      - params 支持 kernel / C / epsilon / gamma（linear 核会忽略 gamma）
    """
    this_scaler = params.get("scaler", scaler_name)
    pipe = Pipeline([
        ("scaler", get_scaler(this_scaler)),
        ("svr", SVR())
    ])

    # 把手动参数塞进 pipeline 的 svr 步骤
    svr_params = {
        "svr__kernel": params.get("kernel", "rbf"),
        "svr__C": float(params.get("C", 300.0)),
        "svr__epsilon": float(params.get("epsilon", 0.3)),
    }
    if "gamma" in params:
        svr_params["svr__gamma"] = params["gamma"]  # 允许 'scale' 或数值
    pipe.set_params(**svr_params)

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
        "kernel": svr_params["svr__kernel"],
        "C":      svr_params["svr__C"],
        "epsilon":svr_params["svr__epsilon"],
        "gamma":  params.get("gamma", None),
        "cv_r2":  cv_mean,
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
    with open(outdir / f"SVR_{tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    ape = np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
    pd.DataFrame({"y_test": y_test, "y_pred": y_pred_test, "APE_percent": ape}).to_csv(
        outdir / f"SVR_{tag}_predictions.csv", index=False
    )

    dump(model, outdir / f"SVR_{tag}_best_model.joblib")

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
    print(f"Scaler（默认）: {SCALER_CHOICE}")

    # 跑多组手写参数
    packs = []
    print("\n================ SVR 手动调参 ================\n")
    for i, p in enumerate(MANUAL_TRIALS, 1):
        this_scaler = p.get("scaler", SCALER_CHOICE)
        # 打印本组设置
        show = {"kernel": p.get("kernel"), "C": p.get("C"),
                "epsilon": p.get("epsilon"), "gamma": p.get("gamma", None),
                "scaler": this_scaler}
        print(f"[{i}/{len(MANUAL_TRIALS)}] 训练参数：{show}")

        metrics, model, y_pred_te = evaluate_svr_once(
            X_train, y_train, X_test, y_test, feature_names,
            params=p, scaler_name=SCALER_CHOICE,
            cv_splits=CV_SPLITS, cv_repeats=CV_REPEATS
        )
        row = {
            "idx": i, "scaler": metrics["scaler"], "kernel": metrics["kernel"],
            "C": metrics["C"], "eps": metrics["epsilon"], "gamma": metrics["gamma"],
            "test_r2": metrics["test_r2"], "cv_r2": metrics["cv_r2"], "cv_std": metrics["cv_std"],
            "rmse": metrics["test_rmse"], "mape%": metrics["test_mape"]
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
    print("\n提示：若 Train R² 明显高于 Test/CV → 过拟合：")
    print("  • 降低 C（如 300→100），或增大 epsilon（如 0.3→0.5）")
    print("  • 减小 gamma（如 0.03→0.01/'scale'）使函数更平滑")
    print("若 Train/Test 都偏低 → 偏差大：")
    print("  • 提高 C（如 300→600），或减小 epsilon（如 0.3→0.2）")
    print("  • 增大 gamma（如 0.01→0.03/0.05）适度增加非线性")

    # 是否保存
    allow_save, reason = _should_save(best_metrics)
    if not allow_save:
        print(f"\n⚠️ 未保存任何文件：{reason}")
    else:
        tag = (f"top1_{best_row['kernel']}_C{best_row['C']}_eps{best_row['eps']}"
               f"_g{best_row['gamma']}_scaler{best_row['scaler']}")
        save_bundle(OUTPUT_DIR, tag, best_metrics, best_model, y_test, best_pred)
        print(f"\n✅ 已保存到：{OUTPUT_DIR.resolve()}")
        print("- SVR_*_metrics.json / SVR_*_predictions.csv / SVR_*_best_model.joblib")
