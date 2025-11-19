#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF 手动调参器（面向 PyCharm 一次跑多组参数，人工选最优）
================================================================
你要做的事：
  1) 在 MANUAL_TRIALS 里写入你想尝试的超参组合（每个是一个 dict）
  2) 运行脚本，查看每组的 CV R²、训练/测试 R²、MAPE、RMSE、OOB（若启用）
  3) 依据打印的排序结果与建议，修改参数继续跑
  4) 满意后把 MASTER_SAVE_SWITCH = True，再次运行保存模型与结果

备注：
  - RF 不需要标准化；保留你预处理后的四个 CSV 直接使用
  - 同步给出 RepeatedKFold 的交叉验证 R²，提升稳健性
  - 支持 OOB（袋外分数），便于快速判断是否过拟合
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# =========【必须改】你的四个预处理 CSV 路径 =========
X_train_path = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_train.csv"
y_train_path = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_train.csv"
X_test_path  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSX_test.csv"
y_test_path  = r"D:\MLDesignAl\TheFinal\Data\Element-UTS\output\Element-UTSy_test.csv"

# ========= 输出与保存：总开关 + 阈值（不满意不落盘） =========
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\RF\Element-UTS\output")
MASTER_SAVE_SWITCH = False      # 调参阶段 False；满意后改 True
AUTO_SAVE_RULE = {              # 想“达标才保存”就填阈值；不要就设为 None
    "min_test_r2": None,        # 例如 0.85
    "max_test_mape": None,      # 例如 5.0
    "min_cv_r2": None           # 例如 0.80
}
# AUTO_SAVE_RULE = None

# ========= 交叉验证与随机性 =========
SEED = 42
CV_SPLITS = 5
CV_REPEATS = 2                  # 想更稳可改 3；时间会更长

# ========= 你要【手动尝试】的参数组（每个 dict 一组） =========
# 参考顺序：先调 max_features → min_samples_leaf/min_samples_split → max_depth → n_estimators
MANUAL_TRIALS = [
    # B. 叶/分裂的正则化扩散（控制过拟合，提升测试R²稳定性）
    {"n_estimators": 1500, "max_features": None, "max_depth": 15,
     "min_samples_leaf": 1, "min_samples_split": 20, "bootstrap": True, "oob_score": True},
    {"n_estimators": 1500, "max_features": None, "max_depth": 15,
     "min_samples_leaf": 1, "min_samples_split": 20, "bootstrap": True, "oob_score": True},
    {"n_estimators": 1500, "max_features": None, "max_depth": 15,
     "min_samples_leaf": 1, "min_samples_split": 20, "bootstrap": True, "oob_score": True},
    {"n_estimators": 1500, "max_features": None, "max_depth": 15,
     "min_samples_leaf": 1, "min_samples_split": 20, "bootstrap": True, "oob_score": True},
]

# ========= 常用工具函数 =========
def read_xy_files(x_train_path, y_train_path, x_test_path, y_test_path):
    """读取四个 CSV，返回 numpy 矩阵与列名"""
    X_train_df = pd.read_csv(x_train_path)
    X_test_df = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0].to_numpy().reshape(-1)
    y_test = pd.read_csv(y_test_path).iloc[:, 0].to_numpy().reshape(-1)
    feature_names = X_train_df.columns.tolist()
    return X_train_df.to_numpy(), y_train, X_test_df.to_numpy(), y_test, feature_names

def mape(y_true, y_pred) -> float:
    """MAPE（百分比），对 0 做极小值保护避免除零。"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

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
    if thr is not None and metrics.get("cv_best_r2", -1) < thr:
        reasons.append(f"cv_best_r2={metrics['cv_best_r2']:.4f} < {thr}")
    if reasons:
        return False, "；".join(reasons)
    return True, ""

def evaluate_rf_once(X_train, y_train, X_test, y_test,
                     feature_names, params: dict, cv_splits=5, cv_repeats=2):
    """
    用给定 params 训练一遍 RF，返回指标与模型。
    关键说明：
      - 强制设置 random_state/n_jobs，其他从 params 读
      - 若 params['oob_score']=True，则自动启用 bootstrap=True
    """
    local_params = params.copy()
    local_params.setdefault("random_state", SEED)
    local_params.setdefault("n_jobs", -1)
    if local_params.get("oob_score", False):
        local_params["bootstrap"] = True  # OOB 需要自助采样

    model = RandomForestRegressor(**local_params)
    model.fit(X_train, y_train)

    # 交叉验证（R²），更稳健
    cv = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=SEED)
    cv_scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=cv, n_jobs=-1)
    cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())

    # 训练/测试集指标
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    metrics = {
        "params": local_params,
        "cv_best_r2": cv_mean,
        "cv_best_std": cv_std,
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "train_mape": mape(y_train, y_pred_train),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "test_mape": mape(y_test, y_pred_test),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "oob_score": float(getattr(model, "oob_score_", np.nan)),
        "n_estimators": int(local_params.get("n_estimators", 100)),
        "n_features": len(feature_names)
    }
    return metrics, model, y_pred_test

def save_bundle(outdir: Path, tag: str, metrics: dict, model, y_test, y_pred_test, feature_names):
    """落盘：metrics / predictions / importances / model"""
    outdir.mkdir(parents=True, exist_ok=True)
    # 1) 指标
    with open(outdir / f"RF_{tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    # 2) 预测
    ape = np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1e-10, y_test)) * 100.0
    pd.DataFrame({"y_test": y_test, "y_pred": y_pred_test, "APE_percent": ape}).to_csv(
        outdir / f"RF_{tag}_predictions.csv", index=False
    )
    # 3) 重要性
    importances = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
    pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    ).to_csv(outdir / f"RF_{tag}_importances.csv", index=False)
    # 4) 模型
    dump(model, outdir / f"RF_{tag}_best_model.joblib")

# ========= 主程序：一次跑多组手动参数，打印排序表 =========
if __name__ == "__main__":
    # 读取数据
    X_train, y_train, X_test, y_test, feature_names = read_xy_files(
        X_train_path, y_train_path, X_test_path, y_test_path
    )

    all_rows = []
    best_pack = None

    print("\n================ RF 手动调参 ================\n")
    for i, p in enumerate(MANUAL_TRIALS, 1):
        print(f"[{i}/{len(MANUAL_TRIALS)}] 训练参数：{p}")
        metrics, model, y_pred_test = evaluate_rf_once(
            X_train, y_train, X_test, y_test, feature_names, p,
            cv_splits=CV_SPLITS, cv_repeats=CV_REPEATS
        )
        row = {
            "idx": i,
            "test_r2": metrics["test_r2"],
            "cv_r2": metrics["cv_best_r2"],
            "cv_std": metrics["cv_best_std"],
            "oob": metrics["oob_score"],
            "rmse": metrics["test_rmse"],
            "mape%": metrics["test_mape"],
            "n_estimators": metrics["n_estimators"],
            "max_depth": p.get("max_depth", None),
            "min_samples_leaf": p.get("min_samples_leaf", None),
            "min_samples_split": p.get("min_samples_split", None),
            "max_features": p.get("max_features", None)
        }
        all_rows.append((row, metrics, model, y_pred_test))

    # 按测试 R² 排序展示
    table = pd.DataFrame([r for r, *_ in all_rows]).sort_values("test_r2", ascending=False)
    print("\n=== 排名（按 Test R²）===\n")
    print(table.to_string(index=False, justify="center", float_format=lambda x: f"{x:.4f}"))

    # 选最优并给保存建议
    best_row, best_metrics, best_model, best_pred = max(all_rows, key=lambda t: t[0]["test_r2"])
    print("\n=== 最优组合（当前轮次） ===")
    print(best_row)
    print("\n提示：若 Train R² 很高而 Test R² 明显低，或 OOB << Train R²，通常是过拟合：")
    print("  • 增大 min_samples_leaf（如 2→4→8）或 min_samples_split（如 2→5→10）")
    print("  • 限制 max_depth（如 None→30→20）")
    print("  • 适当调大 max_features（如 'sqrt'→0.7）可提高泛化；或调小以去耦合")
    print("  • n_estimators 增大带来稳健性，但收益递减：观察 CV R² 是否已平台化")

    # 是否保存
    allow_save, reason = _should_save(best_metrics)
    if not allow_save:
        print(f"\n⚠️ 未保存任何文件：{reason}")
    else:
        tag = f"mantrial_top1_ne{best_row['n_estimators']}_mf{best_row['max_features']}_md{best_row['max_depth']}_ml{best_row['min_samples_leaf']}"
        save_bundle(OUTPUT_DIR, tag, best_metrics, best_model, y_test, best_pred, feature_names)
        print(f"\n✅ 已保存到：{(OUTPUT_DIR / ('RF_'+tag+'_*.{json,csv,joblib}')).parent.resolve()}")
