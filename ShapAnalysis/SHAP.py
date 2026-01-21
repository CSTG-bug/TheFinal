import os
import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 强制使用无界面后端，避免保存图片时弹窗或报错

import matplotlib.pyplot as plt
import shap


# =========================
# 路径与参数
# =========================
MODEL_PATH = r"/XGBoost/ElementTreatmentEl-UTS/output-exceptEL/XGB_best_model.joblib"
X_PATH     = r"/ShapAnalysis/all/all.xlsx"
OUT_DIR    = r"/ShapAnalysis/all"

# 可选：建议提供训练时的列顺序文件，防止“列错位但不报错”
FEATURE_ORDER_JSON = ""

# 计算与导出参数
BACKGROUND_SIZE = 200    # interventional 背景集大小
BATCH_SIZE = 0           # 分批计算 SHAP：0 表示不分批
PLOT_SAMPLE = 3000       # 画 dependence 图时抽样点数
INTERACTION = "auto"     # "auto" / "none" / 或指定某个特征名（作为着色交互特征）
SAVE_FORMAT = "parquet"  # "parquet"或 "csv"
SEED = 0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_feature_order(json_path: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "feature_order" in obj:
        obj = obj["feature_order"]
    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        raise ValueError("feature_order 文件格式不正确，期望为 JSON list[str] 或 {'feature_order': list[str]}")
    return obj


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    # 默认按 csv
    return pd.read_csv(path)


def broadcast_base_values(base_values, n: int) -> np.ndarray:
    base = np.atleast_1d(base_values)
    if base.size == 1:
        return np.full(n, base.item())
    if base.size != n:
        raise ValueError(f"base_values 长度异常：{base.size} vs n_samples={n}")
    return base


def compute_shap_in_batches(explainer, X: pd.DataFrame, batch_size: int):
    """返回 shap.Explanation；必要时分批计算并拼接。"""
    if batch_size <= 0 or batch_size >= len(X):
        return explainer(X)

    chunks = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        chunks.append(explainer(X.iloc[start:end]))

    # 优先使用官方 concatenate
    try:
        return shap.Explanation.concatenate(*chunks)
    except Exception:
        # 兜底：手动拼接
        values = np.vstack([c.values for c in chunks])
        base_values = np.concatenate([broadcast_base_values(c.base_values, len(c.values)) for c in chunks])
        data = np.vstack([c.data for c in chunks]) if chunks[0].data is not None else None
        feature_names = chunks[0].feature_names
        return shap.Explanation(
            values=values,
            base_values=base_values,
            data=data,
            feature_names=feature_names,
        )


def save_summary_plots(shap_values, out_dir: str, max_display: int = 30):
    ensure_dir(out_dir)

    # bar
    plt.figure()
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # beeswarm
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_beeswarm.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_dependence_plots_for_all_features(
    shap_values,
    X: pd.DataFrame,
    out_dir: str,
    interaction: str,
    plot_sample: int,
    seed: int = 0,
):
    """
    为每个特征保存 dependence plot（边际效应图）：
    - interaction='auto'：自动挑交互特征着色
    - interaction='none'：不着色（更快、更清爽）
    - interaction='<feature_name>'：固定用某个特征着色
    """
    ensure_dir(out_dir)

    # 为了画图更快，可抽样；数值导出仍是全量（在另一个函数里完成）
    if plot_sample > 0 and len(X) > plot_sample:
        Xp = X.sample(plot_sample, random_state=seed)
        idx = X.index.get_indexer(Xp.index)

        base_all = broadcast_base_values(shap_values.base_values, len(X))
        sv_values = shap_values.values[idx, :]
        sv_base = base_all[idx]
    else:
        Xp = X
        base_all = broadcast_base_values(shap_values.base_values, len(X))
        sv_values = shap_values.values
        sv_base = base_all

    for feat in X.columns:
        if interaction.lower() == "none":
            interaction_index = None
        elif interaction.lower() == "auto":
            interaction_index = "auto"
        else:
            interaction_index = interaction  # 指定特征名

        # 关键点：不要自己 plt.figure()，让 shap 自己建图
        shap.dependence_plot(
            feat,
            sv_values,
            Xp,
            interaction_index=interaction_index,
            show=False,
        )

        # 保存 shap 创建的当前 figure
        fig = plt.gcf()
        safe_name = str(feat).replace("/", "_").replace("\\", "_").replace(" ", "_")
        fig.savefig(
            os.path.join(out_dir, f"dependence_{safe_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )

        # 关键点：关掉 shap 创建的 figure（避免累计到 20+）
        plt.close(fig)



def export_shap_values(
    shap_values,
    X: pd.DataFrame,
    y_pred: np.ndarray,
    out_dir: str,
    save_format: str = "parquet",
):
    """
    导出：
    - shap_values 数值矩阵（n_samples x n_features）
    - base_values（每个样本的基线）
    - prediction（模型预测）
    同时保存 npz 便于快速加载复现。
    """
    ensure_dir(out_dir)

    base_vals = broadcast_base_values(shap_values.base_values, len(X))

    shap_df = pd.DataFrame(shap_values.values, columns=X.columns, index=X.index)
    shap_df.insert(0, "prediction", y_pred)
    shap_df.insert(1, "base_value", base_vals)

    if save_format.lower() == "csv":
        shap_df.to_csv(os.path.join(out_dir, "shap_values_full.csv"), index=True, encoding="utf-8-sig")
    else:
        shap_df.to_parquet(os.path.join(out_dir, "shap_values_full.parquet"), index=True)

    np.savez_compressed(
        os.path.join(out_dir, "shap_arrays.npz"),
        shap_values=shap_values.values,
        base_values=base_vals,
        predictions=y_pred,
        X=X.values,
        feature_names=np.array(list(X.columns), dtype=object),
    )


def main():
    ensure_dir(OUT_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_root = os.path.join(OUT_DIR, f"shap_export_{ts}")
    data_dir = os.path.join(export_root, "data")
    plots_dir = os.path.join(export_root, "plots")
    dep_dir = os.path.join(plots_dir, "dependence")

    ensure_dir(export_root)
    ensure_dir(data_dir)
    ensure_dir(plots_dir)
    ensure_dir(dep_dir)

    # 1) load model
    model = joblib.load(MODEL_PATH)
    print("[Info] Model type:", type(model))
    print("[Info] model.n_features_in_ =", getattr(model, "n_features_in_", None))
    print("[Info] has feature_names_in_ =", hasattr(model, "feature_names_in_"))

    # 2) load X
    X = read_table(X_PATH)
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X 必须是 DataFrame（含列名）。")

    # 3) enforce feature order
    if FEATURE_ORDER_JSON:
        order = load_feature_order(FEATURE_ORDER_JSON)
        missing = [c for c in order if c not in X.columns]
        extra = [c for c in X.columns if c not in order]
        if missing:
            raise ValueError(f"X 缺少以下特征列：{missing}")
        if extra:
            print(f"[Warn] X 存在训练未定义的额外列，将被忽略：{extra}")
        X = X[order]

    # 4) sanity check: feature count
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and X.shape[1] != n_in:
        raise ValueError(f"特征数不一致：X={X.shape[1]} vs model.n_features_in_={n_in}")

    # 5) background
    bg_n = min(BACKGROUND_SIZE, len(X))
    X_bg = X.sample(bg_n, random_state=SEED)

    # 6) explainer + shap
    explainer = shap.TreeExplainer(
        model,
        data=X_bg,
        feature_perturbation="interventional",
        model_output="raw",  # 回归：解释预测值本身
    )

    shap_values = compute_shap_in_batches(explainer, X, BATCH_SIZE)

    # 7) predictions
    y_pred = model.predict(X)

    # 8) export numeric results
    export_shap_values(shap_values, X, y_pred, data_dir, save_format=SAVE_FORMAT)

    # 9) summary plots
    save_summary_plots(shap_values, plots_dir, max_display=min(30, X.shape[1]))

    # 10) dependence plots for all features
    save_dependence_plots_for_all_features(
        shap_values=shap_values,
        X=X,
        out_dir=dep_dir,
        interaction=INTERACTION,
        plot_sample=PLOT_SAMPLE,
        seed=SEED,
    )

    # 11) metadata
    meta = {
        "timestamp": ts,
        "model_path": MODEL_PATH,
        "x_path": X_PATH,
        "out_dir": OUT_DIR,
        "model_type": str(type(model)),
        "n_features": int(X.shape[1]),
        "n_samples": int(len(X)),
        "background_size": int(bg_n),
        "batch_size": int(BATCH_SIZE),
        "plot_sample": int(PLOT_SAMPLE),
        "interaction": INTERACTION,
        "save_format": SAVE_FORMAT,
        "feature_order_json": FEATURE_ORDER_JSON,
        "explainer_expected_value": getattr(explainer, "expected_value", None),
        "notes": "务必保证 X 的列顺序与训练一致；建议提供 FEATURE_ORDER_JSON 锁定顺序。",
    }
    with open(os.path.join(export_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[Done] Exported to:", export_root)


if __name__ == "__main__":
    main()
