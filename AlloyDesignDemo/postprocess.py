from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# 当前体系的“成分列”
ELEMENT_COLS: List[str] = [
    "Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Zn", "V", "Ti", "Zr",
    "Li", "Ni", "Be", "Sc", "Ag", "Bi", "Pb", "Al"
]

# 元素密度（g/cm3），用于密度计算（质量分数混合法的常用近似）
ELEMENT_DENSITY_G_CM3: Dict[str, float] = {
    "Al": 2.70,
    "Si": 2.33,
    "Fe": 7.87,
    "Cu": 8.96,
    "Mn": 7.21,
    "Mg": 1.74,
    "Cr": 7.19,
    "Zn": 7.14,
    "V": 6.11,
    "Ti": 4.51,
    "Zr": 6.52,
    "Li": 0.534,
    "Ni": 8.90,
    "Be": 1.85,
    "Sc": 2.99,
    "Ag": 10.49,
    "Bi": 9.78,
    "Pb": 11.34,
}


def _ensure_cols(df: pd.DataFrame, cols: List[str], fill_value: float = 0.0) -> pd.DataFrame:
    """确保 df 至少包含 cols；缺失列用 fill_value 补齐。"""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = fill_value
    return out


def add_el_pred(
    df_candidates: pd.DataFrame,
    el_model,
    feature_names_el: List[str],
    uts_pred_col: str = "UTS_pred",
    el_uts_feature_name: str = "UTS",
    out_col: str = "EL_pred",
) -> pd.DataFrame:
    """
    给候选表批量增加 EL_pred。
    要求：
      - df_candidates 至少包含 feature_names_el 中除 UTS 外的所有列
      - 并包含 uts_pred_col（例如 UTS_pred）
    """
    if uts_pred_col not in df_candidates.columns:
        raise ValueError(f"df_candidates 缺少列 {uts_pred_col}，无法给EL模型提供UTS输入")

    df = df_candidates.copy()
    X_el = pd.DataFrame(index=df.index)

    for f in feature_names_el:
        if f == el_uts_feature_name:
            X_el[f] = df[uts_pred_col].astype(float)
        else:
            if f not in df.columns:
                raise ValueError(f"df_candidates 缺少特征列 {f}，无法预测EL")
            X_el[f] = df[f]

    df[out_col] = el_model.predict(X_el)
    return df


def calc_density_g_cm3(
    df_candidates: pd.DataFrame,
    element_cols: List[str] = ELEMENT_COLS,
    density_table: Dict[str, float] = ELEMENT_DENSITY_G_CM3,
    out_col: str = "density_g_cm3",
) -> pd.DataFrame:
    """
    合金密度近似：rho = 1 / Σ(w_i / rho_i)，其中 w_i 为质量分数（wt%/100）。
    """
    df = _ensure_cols(df_candidates, element_cols, 0.0).copy()

    # 质量分数
    w = df[element_cols].astype(float) / 100.0

    denom = np.zeros(len(df), dtype=float)
    missing = []
    for e in element_cols:
        rho = density_table.get(e)
        if rho is None or rho <= 0:
            missing.append(e)
            continue
        denom += (w[e].to_numpy() / rho)

    if missing:
        # 缺密度数据的元素当作忽略
        pass

    df[out_col] = np.where(denom > 0, 1.0 / denom, np.nan)
    return df


def calc_cost_per_kg(
    df_candidates: pd.DataFrame,
    price_table: Dict[str, float],
    element_cols: List[str] = ELEMENT_COLS,
    out_col: str = "cost_per_kg",
) -> pd.DataFrame:
    """
    合金成本（单位由 price_table 决定，例如 元/kg）：
      cost = Σ(w_i * price_i)，w_i = wt%/100
    price_table 由你/用户在界面里填，不在这里硬编码“真实价格”（避免时效性问题）。
    """
    df = _ensure_cols(df_candidates, element_cols, 0.0).copy()
    w = df[element_cols].astype(float) / 100.0

    cost = np.zeros(len(df), dtype=float)
    for e in element_cols:
        p = float(price_table.get(e, 0.0))
        cost += w[e].to_numpy() * p

    df[out_col] = cost
    return df


@dataclass
class ConstraintOptions:
    use_el: bool = False
    target_el: Optional[float] = None

    use_density: bool = False
    max_density: Optional[float] = None

    use_cost: bool = False
    max_cost: Optional[float] = None


def apply_constraints(
    df_candidates: pd.DataFrame,
    opts: ConstraintOptions,
    el_col: str = "EL_pred",
    density_col: str = "density_g_cm3",
    cost_col: str = "cost_per_kg",
) -> pd.DataFrame:
    """按可选约束过滤候选表。"""
    df = df_candidates.copy()

    if opts.use_el and opts.target_el is not None and el_col in df.columns:
        df = df[df[el_col] >= float(opts.target_el)]

    if opts.use_density and opts.max_density is not None and density_col in df.columns:
        df = df[df[density_col] <= float(opts.max_density)]

    if opts.use_cost and opts.max_cost is not None and cost_col in df.columns:
        df = df[df[cost_col] <= float(opts.max_cost)]

    return df.copy()
