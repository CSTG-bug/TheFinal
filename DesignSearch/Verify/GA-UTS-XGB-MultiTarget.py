#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出多目标强度的GA设计方案（基于已训练好的 XGBoost 模型）
=====================================================================
新增“定点强度搜索（target UTS）”模式：
  - 给定多个目标强度 TARGET_UTS_LIST = [600, 500, 400, 300, 200]
  - 脚本会在一次运行中，依次对每个 target 做一轮 GA（适应度 = -|UTS_pred - target|）
  - 每个 target 输出：最接近该目标的 Top-N以及最佳方案
  - 最终汇总 5 条最佳方案到一个总表 CSV
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from joblib import load

# ====================== 配置区 ======================

# 1) 原始 X_train（未标准化版本）的路径
RAW_X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\ElementTreatmentEl-UTS\output-exceptEL\exceptEL-X_train_raw.csv"

# 2) 训练好的 XGBoost 模型路径
MODEL_PATH = r"D:\MLDesignAl\TheFinal\XGBoost\ElementTreatmentEl-UTS\output-exceptEL\XGB_best_model.joblib"

# 3) 结果输出目录
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\DesignSearch\Verify")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 4) 保存开关
MASTER_SAVE_SWITCH = False

# 5) 多目标强度
TARGET_UTS_LIST: List[float] = [600.0, 500.0, 400.0, 300.0, 200.0]
TOP_N_PER_TARGET = 100                 # 每个目标保存“最接近目标”的前 N 条

# 6) 设计空间的手动覆盖
#    - 不写的特征将自动使用 raw X_train 的 min/max
FEATURE_BOUNDS_OVERRIDE: Dict[str, Tuple[float, float]] = {
    # —— 工艺边界
    "Ageing Time": (0.0, 24.0),
    # —— 把非必要元素固定为 0
    "Si": (0.0, 0.0),
    "Fe": (0.0, 0.0),
    "Mn": (0.0, 0.0),
    "Cr": (0.0, 0.0),
    "V":  (0.0, 0.0),
    "Ti": (0.0, 0.1),
    "Zr": (0.0, 0.2),
    "Li": (0.0, 0.0),
    "Ni": (0.0, 0.0),
    "Be": (0.0, 0.0),
    "Sc": (0.0, 0.25),
    "Ag": (0.0, 0.0),
    "Bi": (0.0, 0.0),
    "Pb": (0.0, 0.0),
}

# 7) GA 参数
POP_SIZE = 200
N_GENERATIONS_PER_TARGET = 1000
ELITE_FRAC = 0.1
TOURNAMENT_SIZE = 3
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.2
MUTATION_RATE = 0.1
RANDOM_SEED = 42

# 8) 约束相关设置
COMPOSITION_COLS: list[str] = [
    "Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Zn", "V", "Ti", "Zr", "Li", "Ni", "Be", "Sc", "Ag", "Bi", "Pb", "Al"
]
AL_COL = "Al"
TARGET_SUM = 100.0

AGEING_TIME_COL = "Ageing Time"
AGEING_TIME_MAX = 24.0
AGEING_TIME_STEP: Optional[float] = None

# 在“误差几乎一样”的情况下，轻微偏好更短的 Ageing Time
PREFER_SHORT_AGEING_TIME = True
PREFER_SHORT_AGEING_EPS = 1e-4

# 9) 早停
TARGET_TOL = 5.0                 # 当 |UTS_pred - target| <= 5 MPa 认为已命中
EARLY_STOP_PATIENCE = 50         # 连续命中多少代后提前停止


def load_raw_x_and_bounds(path: str) -> tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    raw_df = pd.read_csv(path)
    feature_names = raw_df.columns.tolist()

    bounds: Dict[str, Tuple[float, float]] = {}
    for col in feature_names:
        bounds[col] = (float(raw_df[col].min()), float(raw_df[col].max()))

    for col, (lo, hi) in FEATURE_BOUNDS_OVERRIDE.items():
        if col not in bounds:
            raise KeyError(f"你在 FEATURE_BOUNDS_OVERRIDE 中指定了列 '{col}'，但在 RAW_X_TRAIN 中未找到该列。")
        bounds[col] = (float(lo), float(hi))

    print("\n=== 原始 X_train 信息 ===")
    print(f"路径     : {path}")
    print(f"样本数   : {len(raw_df)}")
    print(f"特征列数 : {len(feature_names)}")
    print(f"列名     : {feature_names}")

    print("\n=== 设计空间特征范围（最终采用） ===")
    for col in feature_names:
        lo, hi = bounds[col]
        print(f"{col:>12s} : [{lo}, {hi}]")

    return raw_df, bounds


def init_population(bounds: Dict[str, Tuple[float, float]], feature_order: list[str], pop_size: int) -> np.ndarray:
    n_features = len(feature_order)
    pop = np.empty((pop_size, n_features), dtype=float)
    for j, name in enumerate(feature_order):
        lo, hi = bounds[name]
        pop[:, j] = np.random.uniform(lo, hi, size=pop_size)
    return pop


def predict_uts(pop: np.ndarray, feature_order: list[str], model) -> np.ndarray:
    df = pd.DataFrame(pop, columns=feature_order)
    uts = model.predict(df)
    return np.asarray(uts, dtype=float).reshape(-1)


def evaluate_population_target(pop: np.ndarray, feature_order: list[str], model, target_uts: float) -> np.ndarray:
    """适应度：越接近 target_uts 越好（最大化 fitness）。"""
    uts = predict_uts(pop, feature_order, model)
    err = np.abs(uts - float(target_uts))
    fitness = -err

    # 同分择优：更短时效略占优（系数很小，不改变主排序）
    if PREFER_SHORT_AGEING_TIME and (AGEING_TIME_COL in feature_order):
        df = pd.DataFrame(pop, columns=feature_order)
        fitness = fitness + PREFER_SHORT_AGEING_EPS * (AGEING_TIME_MAX - df[AGEING_TIME_COL].values)
    return fitness


def tournament_select(fitness: np.ndarray, k: int, t_size: int) -> int:
    indices = np.random.choice(k, size=t_size, replace=False)
    return int(indices[np.argmax(fitness[indices])])


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.random.rand()
    child1 = alpha * parent1 + (1.0 - alpha) * parent2
    child2 = alpha * parent2 + (1.0 - alpha) * parent1
    return child1, child2


def mutate(ind: np.ndarray, bounds: Dict[str, Tuple[float, float]], feature_order: list[str]) -> np.ndarray:
    new_ind = ind.copy()
    for j, name in enumerate(feature_order):
        if np.random.rand() < MUTATION_PROB:
            lo, hi = bounds[name]
            span = hi - lo
            sigma = span * MUTATION_RATE
            new_ind[j] += np.random.normal(loc=0.0, scale=sigma)
            new_ind[j] = float(np.clip(new_ind[j], lo, hi))
    return new_ind


def repair_individual(ind: np.ndarray,
                      bounds: Dict[str, Tuple[float, float]],
                      feature_order: list[str]) -> np.ndarray:
    idx_map = {name: i for i, name in enumerate(feature_order)}
    x = ind.copy()

    # (A) Ageing Time
    if AGEING_TIME_COL in idx_map:
        i = idx_map[AGEING_TIME_COL]
        lo, hi = bounds[AGEING_TIME_COL]
        hi = min(hi, AGEING_TIME_MAX)
        x[i] = float(np.clip(x[i], lo, hi))
        if AGEING_TIME_STEP is not None and AGEING_TIME_STEP > 0:
            x[i] = round(x[i] / AGEING_TIME_STEP) * AGEING_TIME_STEP
            x[i] = float(np.clip(x[i], lo, hi))

    # (B) composition clip
    comp_cols = [c for c in COMPOSITION_COLS if c in idx_map]
    for col in comp_cols:
        j = idx_map[col]
        lo, hi = bounds[col]
        x[j] = float(np.clip(x[j], lo, hi))

    # (C) sum=100 with Al as remainder
    if (AL_COL in idx_map) and (len(comp_cols) > 0):
        al_j = idx_map[AL_COL]
        other_cols = [c for c in comp_cols if c != AL_COL]
        other_js = [idx_map[c] for c in other_cols]
        sum_other = float(np.sum(x[other_js])) if other_js else 0.0

        al_min = bounds[AL_COL][0]
        max_other = TARGET_SUM - al_min
        if sum_other > max_other + 1e-12 and other_js:
            ratio = max_other / (sum_other + 1e-12)
            x[other_js] *= ratio
            sum_other = float(np.sum(x[other_js]))

        x[al_j] = TARGET_SUM - sum_other
        lo, hi = bounds[AL_COL]
        x[al_j] = float(np.clip(x[al_j], lo, hi))

    return x


def run_ga_search_target(raw_df: pd.DataFrame,
                         bounds: Dict[str, Tuple[float, float]],
                         model,
                         target_uts: float,
                         pop_size: int = POP_SIZE,
                         n_generations: int = N_GENERATIONS_PER_TARGET) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_order = raw_df.columns.tolist()

    pop = init_population(bounds, feature_order, pop_size)
    pop = np.vstack([repair_individual(ind, bounds, feature_order) for ind in pop])
    fitness = evaluate_population_target(pop, feature_order, model, target_uts)

    print("\n=== 开始 GA 定点搜索 ===")
    print(f"目标 UTS = {target_uts:.1f} MPa | 种群={pop_size} | 迭代={n_generations}")

    elite_size = max(1, int(pop_size * ELITE_FRAC))
    hit_streak = 0

    for gen in range(1, n_generations + 1):
        order = np.argsort(-fitness)
        pop = pop[order]
        fitness = fitness[order]

        # 计算当前最优个体的真实误差（不要用 fitness 反推，避免受 eps 影响）
        best_uts = float(predict_uts(pop[:1], feature_order, model)[0])
        best_err = abs(best_uts - target_uts)

        print(f"Gen {gen:03d} | best UTS={best_uts:.3f} | abs_err={best_err:.3f} | mean_fit={fitness.mean():.3f}")

        # 早停
        if best_err <= TARGET_TOL:
            hit_streak += 1
            if hit_streak >= EARLY_STOP_PATIENCE:
                print(f"[EarlyStop] 连续 {EARLY_STOP_PATIENCE} 代命中 ±{TARGET_TOL} MPa，提前停止。")
                break
        else:
            hit_streak = 0

        new_pop = pop[:elite_size].copy()
        while new_pop.shape[0] < pop_size:
            p1_idx = tournament_select(fitness, pop_size, TOURNAMENT_SIZE)
            p2_idx = tournament_select(fitness, pop_size, TOURNAMENT_SIZE)
            p1 = pop[p1_idx]
            p2 = pop[p2_idx]

            if np.random.rand() < CROSSOVER_PROB:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = repair_individual(mutate(c1, bounds, feature_order), bounds, feature_order)
            c2 = repair_individual(mutate(c2, bounds, feature_order), bounds, feature_order)

            new_pop = np.vstack([new_pop, c1[None, :], c2[None, :]])

        if new_pop.shape[0] > pop_size:
            new_pop = new_pop[:pop_size]

        pop = new_pop
        fitness = evaluate_population_target(pop, feature_order, model, target_uts)

    # 最终按 fitness 排序
    order = np.argsort(-fitness)
    pop = pop[order]
    fitness = fitness[order]
    return pop, fitness, feature_order


def save_target_topN(pop: np.ndarray,
                     feature_order: list[str],
                     model,
                     target_uts: float,
                     outdir: Path,
                     top_n: int = 20) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(pop, columns=feature_order)
    df["UTS_pred"] = predict_uts(pop, feature_order, model)
    df["Target_UTS"] = float(target_uts)
    df["AbsError"] = (df["UTS_pred"] - df["Target_UTS"]).abs()

    df = df.sort_values("AbsError", ascending=True).head(top_n).copy()
    out_path = outdir / f"GA_TargetUTS_{int(target_uts)}_top{top_n}.csv"
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"已保存 target={target_uts:.0f} 的 Top-{top_n} 到：{out_path}")
    return out_path


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    raw_df, bounds = load_raw_x_and_bounds(RAW_X_TRAIN_PATH)

    print("\n=== 加载 XGBoost 模型 ===")
    print(f"模型路径: {MODEL_PATH}")
    model = load(MODEL_PATH)
    print(f"模型类型: {type(model)}")

    best_rows = []

    for target in TARGET_UTS_LIST:
        pop, fitness, feature_order = run_ga_search_target(
            raw_df=raw_df,
            bounds=bounds,
            model=model,
            target_uts=float(target),
            pop_size=POP_SIZE,
            n_generations=N_GENERATIONS_PER_TARGET,
        )

        # 保存每个 target 的 Top-N
        if MASTER_SAVE_SWITCH:
            save_target_topN(pop, feature_order, model, float(target), OUTPUT_DIR, top_n=TOP_N_PER_TARGET)

        # 取“最接近 target”的最佳一条
        df_all = pd.DataFrame(pop, columns=feature_order)
        df_all["UTS_pred"] = predict_uts(pop, feature_order, model)
        df_all["Target_UTS"] = float(target)
        df_all["AbsError"] = (df_all["UTS_pred"] - df_all["Target_UTS"]).abs()
        best = df_all.sort_values("AbsError", ascending=True).iloc[0].to_dict()
        best_rows.append(best)

        print("\n[Best] target={:.0f} | UTS_pred={:.3f} | AbsError={:.3f}".format(
            target, best["UTS_pred"], best["AbsError"]
        ))

    # 汇总 5 条最佳
    df_best = pd.DataFrame(best_rows)
    # 按目标强度从高到低排序
    df_best = df_best.sort_values("Target_UTS", ascending=False)
    out_best = OUTPUT_DIR / "GA_TargetUTS_best5.csv"
    df_best.to_csv(out_best, index=False, float_format="%.6f")
    print(f"\n=== 已汇总 5 个目标强度的最佳方案到：{out_best} ===")
