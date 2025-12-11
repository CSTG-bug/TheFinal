#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多输出随机森林（成分 + 工艺 → UTS & EL）
================================================
功能汇总：
  1）同时预测 UTS 和 EL（多输出回归）
  2）对 X（成分+工艺）和 Y（UTS, EL）都做 StandardScaler 标准化
  3）支持“手动调参模式”：一次跑多组参数，只打印结果，不保存
  4）支持“定型模式”：确定好一组参数后，只用这一组训练 + 评估 + 保存结果/模型
  5）保存内容（定型模式且 MASTER_SAVE_SWITCH=True 时）：
      - metrics.json：包含 UTS / EL 训练和测试的 R²、MAPE、MAE、RMSE 等
      - predictions.csv：测试集中每个样本的真值/预测值/APE%
      - importances.csv：特征重要性
      - MultiRF_best_model.joblib：已经包含 X/Y 标准化器和 RF 模型的打包对象

用法说明（建议按顺序来）：
  A. 调参阶段：
     - 把 MANUAL_TUNE = True
     - 在 MANUAL_TRIALS 中写不同的超参数组合
     - 运行脚本，只会打印每组的 UTS/EL 训练/测试指标 + CV R² 排名，不会落盘
  B. 定型阶段：
     - 选中你最满意的一组参数，填到 FINAL_PARAMS 里
     - 把 MANUAL_TUNE = False
     - 把 MASTER_SAVE_SWITCH = True
     - 再运行一次，脚本会用 FINAL_PARAMS 训练 + 评估 + 保存结果文件
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from joblib import dump

# ====================== 1. 文件路径与模式开关（需要你改） ======================

X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\ElementTreatment-UTSEL\ElementTreatment-UTSEL-X_train_raw.csv"
Y_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\ElementTreatment-UTSEL\ElementTreatment-UTSEL-y_train.csv"
X_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\ElementTreatment-UTSEL\ElementTreatment-UTSEL-X_test_raw.csv"
Y_TEST_PATH  = r"D:\MLDesignAl\TheFinal\Data\ElementTreatment-UTSEL\ElementTreatment-UTSEL-y_test.csv"

# 输出目录（定型模式下保存 json / csv / joblib）
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\RF\ElementTreatment-UTSEL\output")

# 是否处于“手动调参模式”
#   True  : 只跑 MANUAL_TRIALS 中的参数组合 → 打印结果，不保存
#   False : 只用 FINAL_PARAMS 这一组参数 → 训练 + 评估 + （可选）保存
MANUAL_TUNE: bool = True

# 定型模式下是否真的把文件落盘
MASTER_SAVE_SWITCH: bool = False   # 调参时建议 False；定型满意后改 True

# ====================== 2. 交叉验证 & 随机性参数 ======================

SEED = 42
CV_SPLITS = 5
CV_REPEATS = 2  # 想更稳可以改 3；时间会更长一些

# ====================== 3. 手动调参参数列表 & 最终参数 ======================

MANUAL_TRIALS: List[Dict[str, Any]] = [
    {"n_estimators": 800, "max_depth": 15, "min_samples_split": 2,
     "min_samples_leaf": 1, "max_features": 0.5, "bootstrap": True},

    {"n_estimators": 800, "max_depth": 15, "min_samples_split": 2,
     "min_samples_leaf": 1, "max_features": 0.7, "bootstrap": True},

    {"n_estimators": 800, "max_depth": 15, "min_samples_split": 2,
     "min_samples_leaf": 1, "max_features": None, "bootstrap": True},

    {"n_estimators": 800, "max_depth": 15, "min_samples_split": 2,
     "min_samples_leaf": 1, "max_features": "sqrt", "bootstrap": True},
]

FINAL_PARAMS: Dict[str, Any] = {
    "n_estimators": 1500,
    "max_depth": 15,
    "min_samples_split": 20,
    "min_samples_leaf": 1,
    "max_features": 0.5,
    "bootstrap": True,
}

# ====================== 4. 工具函数：读数据 & 指标计算 ======================

def read_xy_files(
    x_train_path: str,
    y_train_path: str,
    x_test_path: str,
    y_test_path: str,
    target_cols: List[str] = ("UTS", "EL"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    从四个 CSV 中读取 X_train / Y_train / X_test / Y_test。

    要求：
      - X_* 的列名就是特征名（元素 + 工艺）
      - Y_* 至少包含 target_cols 中的列（这里默认 ["UTS", "EL"]）

    返回：
      X_train, Y_train, X_test, Y_test, feature_names, target_names
    """
    X_train_df = pd.read_csv(x_train_path)
    X_test_df = pd.read_csv(x_test_path)
    Y_train_df = pd.read_csv(y_train_path)
    Y_test_df = pd.read_csv(y_test_path)

    # 确认目标列存在
    for c in target_cols:
        if c not in Y_train_df.columns or c not in Y_test_df.columns:
            raise KeyError(
                f"在 y_train / y_test 中未找到目标列 {c}。\n"
                f"y_train 列：{list(Y_train_df.columns)}\n"
                f"y_test  列：{list(Y_test_df.columns)}"
            )

    Y_train = Y_train_df[list(target_cols)].to_numpy()
    Y_test = Y_test_df[list(target_cols)].to_numpy()

    # 简单去除缺失值（X 或 Y 中任一存在 NaN 就丢弃该样本）
    mask_tr = ~np.isnan(X_train_df.to_numpy()).any(axis=1) & ~np.isnan(Y_train).any(axis=1)
    mask_te = ~np.isnan(X_test_df.to_numpy()).any(axis=1) & ~np.isnan(Y_test).any(axis=1)

    X_train = X_train_df.to_numpy()[mask_tr]
    Y_train = Y_train[mask_tr]
    X_test = X_test_df.to_numpy()[mask_te]
    Y_test = Y_test[mask_te]

    feature_names = list(X_train_df.columns)
    target_names = list(target_cols)

    return X_train, Y_train, X_test, Y_test, feature_names, target_names

def mape(y_true, y_pred) -> float:
    """MAPE（百分比），对 0 做极小值保护。"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

def print_target_metrics(
    name: str,
    y_tr: np.ndarray,
    y_tr_pred: np.ndarray,
    y_te: np.ndarray,
    y_te_pred: np.ndarray,
) -> Dict[str, float]:
    """
    打印单个目标（UTS 或 EL）的训练/测试指标，并把结果打包成 dict 返回。
    """
    train_r2 = r2_score(y_tr, y_tr_pred)
    test_r2 = r2_score(y_te, y_te_pred)

    train_mape = mape(y_tr, y_tr_pred)
    test_mape = mape(y_te, y_te_pred)

    train_mae = mean_absolute_error(y_tr, y_tr_pred)
    test_mae = mean_absolute_error(y_te, y_te_pred)

    train_rmse = float(np.sqrt(mean_squared_error(y_tr, y_tr_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(y_te, y_te_pred)))

    print(f"\n=== 目标：{name} ===")
    print(f"[训练集] R²   : {train_r2:.4f}")
    print(f"[训练集] MAPE : {train_mape:.2f}%")
    print(f"[训练集] MAE  : {train_mae:.4f}")
    print(f"[训练集] RMSE : {train_rmse:.4f}")
    print(f"[测试集] R²   : {test_r2:.4f}")
    print(f"[测试集] MAPE : {test_mape:.2f}%")
    print(f"[测试集] MAE  : {test_mae:.4f}")
    print(f"[测试集] RMSE : {test_rmse:.4f}")

    return {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_mape": float(train_mape),
        "test_mape": float(test_mape),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
    }

# ====================== 5. 多输出 + 标准化 的 RF 封装类 ======================

class ScaledMultiOutputRF(BaseEstimator, RegressorMixin):
    """
    一个小封装：把
      - X 标准化（StandardScaler）
      - Y 标准化（StandardScaler）
      - RandomForestRegressor（支持多输出）
    三者打包成一个“模型”。

    对你而言，可以把它当成普通回归模型用：
      model = ScaledMultiOutputRF(...超参数...)
      model.fit(X_train, Y_train)
      Y_pred = model.predict(X_test)  # 这里返回的是已经“反标准化”的真实 UTS / EL
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float | int | None = "auto",
        bootstrap: bool = True,
        random_state: int = SEED,
        n_jobs: int = -1,
        scale_x: bool = True,
        scale_y: bool = True,
    ):
        # RF 的超参数
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        # 是否对 X / Y 做标准化（默认都 True）
        self.scale_x = scale_x
        self.scale_y = scale_y

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """拟合模型：内部会根据配置对 X、Y 做标准化再训练 RF。"""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        # 1）X 标准化
        if self.scale_x:
            self.x_scaler_ = StandardScaler()
            X_scaled = self.x_scaler_.fit_transform(X)
        else:
            self.x_scaler_ = None
            X_scaled = X

        # 2）Y 标准化
        if self.scale_y:
            self.y_scaler_ = StandardScaler()
            Y_scaled = self.y_scaler_.fit_transform(Y)
        else:
            self.y_scaler_ = None
            Y_scaled = Y

        # 3）RF 本体
        self.rf_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.rf_.fit(X_scaled, Y_scaled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测时会先对 X 做同样的标准化，然后把 Y 反标准化回真实单位。"""
        X = np.asarray(X, dtype=float)

        if getattr(self, "x_scaler_", None) is not None:
            X_scaled = self.x_scaler_.transform(X)
        else:
            X_scaled = X

        Y_scaled_pred = self.rf_.predict(X_scaled)

        if getattr(self, "y_scaler_", None) is not None:
            Y_pred = self.y_scaler_.inverse_transform(Y_scaled_pred)
        else:
            Y_pred = Y_scaled_pred

        return Y_pred

    @property
    def feature_importances_(self) -> np.ndarray:
        """暴露内部 RF 的特征重要性，便于后面做解释性分析。"""
        return self.rf_.feature_importances_

# ====================== 6. 单次训练 + 评估（供手动调参和定型共用） ======================

def train_eval_once(
    params: Dict[str, Any],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    target_names: List[str],
) -> Tuple[Dict[str, Any], ScaledMultiOutputRF, np.ndarray]:
    """
    用给定 params 训练一次多输出 RF，并返回：
      - metrics：包含 UTS/EL 的训练/测试指标 + CV R²
      - model  ：已经 fit 好的 ScaledMultiOutputRF 实例
      - Y_test_pred：测试集的预测结果（真实单位）

    params 中写的是 RF 的超参数（不含 scale_*，默认都标准化）。
    """
    # ------- 1）构造模型并训练 -------
    model = ScaledMultiOutputRF(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features", "auto"),
        bootstrap=params.get("bootstrap", True),
        random_state=params.get("random_state", SEED),
        n_jobs=params.get("n_jobs", -1),
        scale_x=True,
        scale_y=True,
    )
    model.fit(X_train, Y_train)

    # ------- 2）交叉验证（对两个目标一起算平均 R²） -------
    cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)
    cv_scores = cross_val_score(model, X_train, Y_train, cv=cv, scoring="r2", n_jobs=-1)
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    # ------- 3）训练集 / 测试集预测 -------
    Y_tr_pred = model.predict(X_train)
    Y_te_pred = model.predict(X_test)

    # ------- 4）分别打印 UTS / EL 的指标 -------
    idx_uts = target_names.index("UTS")
    idx_el = target_names.index("EL")

    print("\n>>> 当前参数：", params)
    print(f"CV R²（两个目标平均）: {cv_mean:.4f} ± {cv_std:.4f}")

    uts_metrics = print_target_metrics(
        "UTS",
        Y_train[:, idx_uts], Y_tr_pred[:, idx_uts],
        Y_test[:, idx_uts], Y_te_pred[:, idx_uts],
    )
    el_metrics = print_target_metrics(
        "EL",
        Y_train[:, idx_el], Y_tr_pred[:, idx_el],
        Y_test[:, idx_el], Y_te_pred[:, idx_el],
    )

    # ------- 5）整体打包 metrics -------
    metrics = {
        "params": params,
        "cv_r2_mean": cv_mean,
        "cv_r2_std": cv_std,
        "UTS": uts_metrics,
        "EL": el_metrics,
        "n_train": int(len(Y_train)),
        "n_test": int(len(Y_test)),
    }
    return metrics, model, Y_te_pred

# ====================== 7. 定型模式下的落盘函数 ======================

def save_bundle(
    outdir: Path,
    metrics: Dict[str, Any],
    model: ScaledMultiOutputRF,
    Y_test: np.ndarray,
    Y_test_pred: np.ndarray,
    feature_names: List[str],
    target_names: List[str],
) -> None:
    """
    把 metrics / 预测结果 / 特征重要性 / 模型打包保存到磁盘。
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 1）指标 json
    import json
    metrics_path = outdir / "MultiRF_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 2）预测结果：每个目标的真值/预测值/APE%
    cols = {}
    for j, name in enumerate(target_names):
        y_true = Y_test[:, j]
        y_pred = Y_test_pred[:, j]
        ape = np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true)) * 100.0
        cols[f"{name}_true"] = y_true
        cols[f"{name}_pred"] = y_pred
        cols[f"{name}_APE_percent"] = ape
    pred_df = pd.DataFrame(cols)
    pred_df.to_csv(outdir / "MultiRF_predictions.csv", index=False)

    # 3）特征重要性
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df.sort_values("importance", ascending=False).to_csv(
        outdir / "MultiRF_importances.csv", index=False
    )

    # 4）模型打包：包含 X/Y 标准化器 + RF 本体 + 特征/目标名
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "target_names": target_names,
    }
    dump(bundle, outdir / "MultiRF_best_model.joblib")

    print("\n✅ 已保存文件：")
    print("  -", metrics_path)
    print("  - MultiRF_predictions.csv")
    print("  - MultiRF_importances.csv")
    print("  - MultiRF_best_model.joblib")

# ====================== 8. 主程序：手动调参 or 定型 ======================

def main():
    # ---------- 8.1 读取数据 ----------
    X_train, Y_train, X_test, Y_test, feat_names, tgt_names = read_xy_files(
        X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH,
        target_cols=["UTS", "EL"],
    )

    print("=== 数据概况 ===")
    print(f"n_train / n_test : {len(Y_train)} / {len(Y_test)}")
    print(f"n_features       : {X_train.shape[1]}")
    print(f"targets          : {tgt_names}")

    # ---------- 8.2 手动调参模式 ----------
    if MANUAL_TUNE:
        print("\n================ 多输出 RF 手动调参 ================\n")
        rows = []
        for i, p in enumerate(MANUAL_TRIALS, 1):
            print(f"\n------ [{i}/{len(MANUAL_TRIALS)}] ------")
            metrics, model, Y_te_pred = train_eval_once(
                params=p,
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                target_names=tgt_names,
            )
            row = {
                "idx": i,
                "test_r2_UTS": metrics["UTS"]["test_r2"],
                "test_r2_EL": metrics["EL"]["test_r2"],
                "cv_r2_mean": metrics["cv_r2_mean"],
                "cv_r2_std": metrics["cv_r2_std"],
                "n_estimators": p.get("n_estimators"),
                "max_depth": p.get("max_depth"),
                "min_samples_split": p.get("min_samples_split"),
                "min_samples_leaf": p.get("min_samples_leaf"),
                "max_features": p.get("max_features"),
            }
            rows.append(row)

        # 按 UTS 测试 R² 排序展示（你也可以改成按 EL 排）
        table = pd.DataFrame(rows).sort_values("test_r2_UTS", ascending=False)
        print("\n=== 手动调参排名（按 UTS 测试 R²） ===\n")
        print(table.to_string(index=False, justify="center", float_format=lambda x: f"{x:.4f}"))

        print("\n提示")
        print("  • max_features(每次分裂可用特征数):'sqrt', 0.5, 0.7, None")
        print("  • min_samples_leaf(叶子最小样本):1, 2, 4, 8")
        print("  • min_samples_split(分裂最小样本):2, 5, 10, 20")
        print("  • max_depth(最大深度):None, 20, 30, 40, 50")
        print("  • n_estimators(数的数量):600, 900, 1200, 1500, 1800")

        print("\n提示：根据上表选择你最满意的一组参数，复制到 FINAL_PARAMS，"
              "然后把 MANUAL_TUNE=False、MASTER_SAVE_SWITCH=True 再跑一遍即完成定型与保存。")
        return

    # ---------- 8.3 定型模式 ----------
    print("\n================ 多输出 RF 定型模式 ================\n")
    print("将使用 FINAL_PARAMS：", FINAL_PARAMS)
    metrics, model, Y_te_pred = train_eval_once(
        params=FINAL_PARAMS,
        X_train=X_train, Y_train=Y_train,
        X_test=X_test, Y_test=Y_test,
        target_names=tgt_names,
    )

    if not MASTER_SAVE_SWITCH:
        print("\n⚠ 未保存任何文件：MASTER_SAVE_SWITCH=False（总开关关闭）。"
              "如果你对上述指标满意，把 MASTER_SAVE_SWITCH 改为 True 再运行一次即可落盘。")
    else:
        save_bundle(
            outdir=OUTPUT_DIR,
            metrics=metrics,
            model=model,
            Y_test=Y_test,
            Y_test_pred=Y_te_pred,
            feature_names=feat_names,
            target_names=tgt_names,
        )

if __name__ == "__main__":
    main()
