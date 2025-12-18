#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用遗传算法(GA)在设计空间中搜索高 UTS 配方 —— 基于已训练好的 XGBoost 模型
=====================================================================
功能概览：
  1) 读取你已经保存好的 X_train_raw，用于：
       - 确认特征列名顺序
       - 估计每个特征的默认范围（min/max）
  2) 允许你在代码顶部按“特征名: [min, max]”的形式手动覆盖设计空间
  3) 加载已训练好的 XGB 模型 (XGB_best_model.joblib)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from joblib import load

# ====================== 配置区（需要根据你的项目修改） ======================

# 1) 原始 X_train（未标准化版本）的路径 —— 用于估计范围 & 特征名
RAW_X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\ElementTreatmentEl-UTS\output-exceptEL\exceptEL-X_train_raw.csv"

# 2) 训练好的 XGBoost 模型路径（你之前保存的 XGB_best_model.joblib）
MODEL_PATH = r"D:\MLDesignAl\TheFinal\XGBoost\ElementTreatmentEl-UTS\output-exceptEL\XGB_best_model.joblib"

# 3) 结果输出目录
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\DesignSearch\GA-UTS-XGB")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 4) 是否保存结果，以及保存多少条
MASTER_SAVE_SWITCH = True      # 调参/测试阶段建议 False；满意后改为 True
TOP_K_TO_SAVE = 200             # 保存 UTS_pred 最高的前 K 条方案

# 5) 设计空间的手动覆盖（可选）
#    - 不写的特征将自动使用 raw X_train 的 min/max
FEATURE_BOUNDS_OVERRIDE: Dict[str, Tuple[float, float]] = {
    "Ageing Time": (0.0, 48.0),
    # "Si": (0.0, 2.0),
    # "Mg": (0.0, 2.0),
    # "Al": (80.0, 100.0),
    # "SS Temp": (450.0, 560.0),
    # "Ageing Temp": (100.0, 250.0),
    # "Ageing Time": (60.0, 1200.0),
}

# 6) GA 参数
POP_SIZE = 200             # 每一代的个体数量
N_GENERATIONS = 1250         # 迭代次数
ELITE_FRAC = 0.1           # 精英保留比例（前 10% 直接保留到下一代）
TOURNAMENT_SIZE = 3        # 锦标赛选择的个体数量
CROSSOVER_PROB = 0.9       # 发生交叉的概率
MUTATION_PROB = 0.2        # 每个基因发生变异的概率
MUTATION_RATE = 0.1        # 变异幅度（相对于特征范围的百分比）

RANDOM_SEED = 42           # 全局随机种子，保证可复现

# 7) 约束相关设置（实验可行性）
#    - 成分约束：COMPOSITION_COLS 中所有元素质量分数之和 = 100（通过把 Al 作为余量强制满足）
#    - 时效时间约束：Ageing Time ≤ AGEING_TIME_MAX（并可选离散到 AGEING_TIME_STEP 便于实验）
COMPOSITION_COLS: list[str] = [
    "Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Zn", "V", "Ti", "Zr", "Li", "Ni", "Be", "Sc", "Ag", "Bi", "Pb", "Al"
]
AL_COL = "Al"
TARGET_SUM = 100.0

AGEING_TIME_COL = "Ageing Time"
AGEING_TIME_MAX = 48.0
AGEING_TIME_STEP: Optional[float] = 0.5   # 例如 1.0=按 1h 步长；若不想离散化，改为 None

# 可选：在 UTS 基本相同的情况下，轻微偏好更短的 Ageing Time（不改变“主目标是 UTS 最大化”）
PREFER_SHORT_AGEING_TIME = True
PREFER_SHORT_AGEING_EPS = 1e-4   # 越小越“只做同分择优”

# ======================================================================


def load_raw_x_and_bounds(path: str) -> tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    读取原始 X_train，并基于数据计算默认的特征范围（min/max），然后套用用户覆盖。
    返回：
      - raw_df: 原始 X_train 数据（DataFrame）
      - bounds: dict[str, (min, max)]，最终的设计空间
    """
    raw_df = pd.read_csv(path)
    feature_names = raw_df.columns.tolist()

    # 默认范围：用原始数据的 min/max
    bounds: Dict[str, Tuple[float, float]] = {}
    for col in feature_names:
        col_min = float(raw_df[col].min())
        col_max = float(raw_df[col].max())
        bounds[col] = (col_min, col_max)

    # 应用用户覆盖（如果有）
    for col, (lo, hi) in FEATURE_BOUNDS_OVERRIDE.items():
        if col not in bounds:
            raise KeyError(f"你在 FEATURE_BOUNDS_OVERRIDE 中指定了列 '{col}'，"
                           f"但在 RAW_X_TRAIN 中未找到该列。")
        bounds[col] = (float(lo), float(hi))

    # 控制台打印一下用于确认
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
    """
    初始化种群：
      - 输入：特征范围 dict + 特征名顺序 + 种群大小
      - 输出：pop (pop_size, n_features) 的 numpy 数组
    """
    n_features = len(feature_order)
    pop = np.empty((pop_size, n_features), dtype=float)

    for j, name in enumerate(feature_order):
        lo, hi = bounds[name]
        pop[:, j] = np.random.uniform(lo, hi, size=pop_size)

    return pop


def evaluate_population(pop: np.ndarray, feature_order: list[str], model) -> np.ndarray:
    """
    对种群中每个个体计算适应度（这里就是预测 UTS）。
    注意：
      - 我们假设 XGB 模型是用“原始特征”训练的，因此这里直接用原始数值。
      - 用 DataFrame 保持列名和训练时一致，避免特征顺序问题。
    """
    df = pd.DataFrame(pop, columns=feature_order)
    uts_pred = model.predict(df)   # 这里直接调用 XGBRegressor 的 predict
    uts_pred = np.asarray(uts_pred, dtype=float).reshape(-1)

    # 可选：只在 UTS 接近时偏好更短时效（同分择优，基本不影响 UTS）
    if PREFER_SHORT_AGEING_TIME and (AGEING_TIME_COL in df.columns):
        # 让 Ageing Time 越短，适应度略微越高（系数极小）
        uts_pred = uts_pred + PREFER_SHORT_AGEING_EPS * (AGEING_TIME_MAX - df[AGEING_TIME_COL].values)
    return uts_pred


def tournament_select(fitness: np.ndarray, k: int, t_size: int) -> int:
    """
    锦标赛选择：
      - 在 [0, k) 范围内随机抽取 t_size 个个体
      - 返回其中适应度最高者的索引
    """
    indices = np.random.choice(k, size=t_size, replace=False)
    best_idx = indices[np.argmax(fitness[indices])]
    return best_idx


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    简单的算术交叉 (arithmetic crossover)：
      - child1 = alpha * p1 + (1 - alpha) * p2
      - child2 = alpha * p2 + (1 - alpha) * p1
      - alpha 在 [0, 1] 随机
    """
    alpha = np.random.rand()
    child1 = alpha * parent1 + (1.0 - alpha) * parent2
    child2 = alpha * parent2 + (1.0 - alpha) * parent1
    return child1, child2


def mutate(ind: np.ndarray, bounds: Dict[str, Tuple[float, float]], feature_order: list[str]) -> np.ndarray:
    """
    变异操作：
      - 对每个基因，以 MUTATION_PROB 的概率做一次高斯扰动
      - 扰动幅度 ~ N(0, sigma)，sigma = 特征范围 * MUTATION_RATE
      - 最后把值裁剪回 [min, max] 范围内
    """
    new_ind = ind.copy()
    for j, name in enumerate(feature_order):
        if np.random.rand() < MUTATION_PROB:
            lo, hi = bounds[name]
            span = hi - lo
            sigma = span * MUTATION_RATE
            new_ind[j] += np.random.normal(loc=0.0, scale=sigma)
            # 裁剪到合法范围
            if new_ind[j] < lo:
                new_ind[j] = lo
            elif new_ind[j] > hi:
                new_ind[j] = hi
    return new_ind


def repair_individual(ind: np.ndarray,
                      bounds: Dict[str, Tuple[float, float]],
                      feature_order: list[str]) -> np.ndarray:
    """硬约束修复：
    1) Ageing Time ≤ AGEING_TIME_MAX（可选离散化到 AGEING_TIME_STEP）
    2) 成分总和=TARGET_SUM（把 Al 设为余量；并限制 sum(others) ≤ TARGET_SUM - Al_min 以避免 Al 越界）
    """
    idx_map = {name: i for i, name in enumerate(feature_order)}
    x = ind.copy()

    # ---- (A) Ageing Time 硬裁剪 + 可选离散化 ----
    if AGEING_TIME_COL in idx_map:
        i = idx_map[AGEING_TIME_COL]
        lo, hi = bounds[AGEING_TIME_COL]
        hi = min(hi, AGEING_TIME_MAX)
        x[i] = float(np.clip(x[i], lo, hi))
        if AGEING_TIME_STEP is not None and AGEING_TIME_STEP > 0:
            x[i] = round(x[i] / AGEING_TIME_STEP) * AGEING_TIME_STEP
            x[i] = float(np.clip(x[i], lo, hi))

    # ---- (B) 成分列先裁剪到各自上下限 ----
    comp_cols = [c for c in COMPOSITION_COLS if c in idx_map]
    for col in comp_cols:
        j = idx_map[col]
        lo, hi = bounds[col]
        x[j] = float(np.clip(x[j], lo, hi))

    # ---- (C) 质量守恒：把 Al 作为余量，保证总和=TARGET_SUM ----
    if (AL_COL in idx_map) and (len(comp_cols) > 0):
        al_j = idx_map[AL_COL]
        other_cols = [c for c in comp_cols if c != AL_COL]
        other_js = [idx_map[c] for c in other_cols]

        sum_other = float(np.sum(x[other_js])) if other_js else 0.0

        # 受 Al 下限约束，others 的总和不能超过 TARGET_SUM - Al_min
        al_min = bounds[AL_COL][0]
        max_other = TARGET_SUM - al_min

        if sum_other > max_other + 1e-12 and other_js:
            ratio = max_other / (sum_other + 1e-12)
            x[other_js] *= ratio
            sum_other = float(np.sum(x[other_js]))

        x[al_j] = TARGET_SUM - sum_other

        # 最后再裁剪一次 Al（理论上不会越界）
        lo, hi = bounds[AL_COL]
        x[al_j] = float(np.clip(x[al_j], lo, hi))

    return x


def run_ga_search(raw_df: pd.DataFrame,
                  bounds: Dict[str, Tuple[float, float]],
                  model,
                  pop_size: int = POP_SIZE,
                  n_generations: int = N_GENERATIONS) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    运行主 GA 循环：
      - 初始化种群
      - 多代进化
      - 返回最终种群及其适应度，以及特征名顺序
    """
    feature_order = raw_df.columns.tolist()
    n_features = len(feature_order)

    # 初始化种群
    pop = init_population(bounds, feature_order, pop_size)

    # 约束修复：保证初始种群满足（Ageing Time 上限 + 元素总和=100）
    pop = np.vstack([repair_individual(ind, bounds, feature_order) for ind in pop])

    # 评估初始适应度
    fitness = evaluate_population(pop, feature_order, model)

    print("\n=== 开始 GA 进化搜索 ===")
    print(f"初始种群大小: {pop_size}, 特征数: {n_features}, 迭代代数: {n_generations}")

    elite_size = max(1, int(pop_size * ELITE_FRAC))

    for gen in range(1, n_generations + 1):
        # 1. 按适应度排序（从高到低）
        order = np.argsort(-fitness)
        pop = pop[order]
        fitness = fitness[order]

        best = fitness[0]
        mean = fitness.mean()
        std = fitness.std()

        print(f"Gen {gen:03d} | best UTS={best:.3f}, mean={mean:.3f}, std={std:.3f}")

        # 2. 精英保留
        new_pop = pop[:elite_size].copy()

        # 3. 生成剩余个体（选择 + 交叉 + 变异）
        while new_pop.shape[0] < pop_size:
            # 锦标赛从前 k=pop_size 中选择父母
            p1_idx = tournament_select(fitness, pop_size, TOURNAMENT_SIZE)
            p2_idx = tournament_select(fitness, pop_size, TOURNAMENT_SIZE)
            parent1 = pop[p1_idx]
            parent2 = pop[p2_idx]

            # 交叉
            if np.random.rand() < CROSSOVER_PROB:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # 变异
            child1 = repair_individual(mutate(child1, bounds, feature_order), bounds, feature_order)
            child2 = repair_individual(mutate(child2, bounds, feature_order), bounds, feature_order)

            new_pop = np.vstack([new_pop, child1[None, :], child2[None, :]])

        # 截断到指定大小
        if new_pop.shape[0] > pop_size:
            new_pop = new_pop[:pop_size]

        # 更新种群与适应度
        pop = new_pop
        fitness = evaluate_population(pop, feature_order, model)

    # 最终排序
    order = np.argsort(-fitness)
    pop = pop[order]
    fitness = fitness[order]

    print("\n=== GA 搜索结束 ===")
    print(f"最终 best UTS={fitness[0]:.3f}, mean={fitness.mean():.3f}")

    return pop, fitness, feature_order


def save_top_candidates(pop: np.ndarray,
                        fitness: np.ndarray,
                        feature_order: list[str],
                        outdir: Path,
                        top_k: int = 200) -> Path:
    """
    保存 Top-K 个候选到 CSV：
      - 每行包含：所有成分/工艺特征 + 预测 UTS_pred
    """
    outdir.mkdir(parents=True, exist_ok=True)
    k = min(top_k, pop.shape[0])

    df = pd.DataFrame(pop, columns=feature_order)
    df["UTS_pred"] = fitness

    top_df = df.iloc[:k].copy()
    out_path = outdir / f"GA_UTS_top{str(k)}.csv"
    top_df.to_csv(out_path, index=False, float_format="%.6f")

    print(f"\n已保存 Top-{k} 候选方案到：{out_path}")
    return out_path


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    # 1) 读取原始 X_train + 设计空间
    raw_df, bounds = load_raw_x_and_bounds(RAW_X_TRAIN_PATH)

    # 2) 加载 XGB 模型
    print("\n=== 加载 XGBoost 模型 ===")
    print(f"模型路径: {MODEL_PATH}")
    model = load(MODEL_PATH)
    print(f"模型类型: {type(model)}")

    # 3) 运行 GA 搜索
    pop, fitness, feature_order = run_ga_search(
        raw_df=raw_df,
        bounds=bounds,
        model=model,
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS
    )

    # 4) 控制台预览前 10 个方案
    print("\n=== Top-10 方案预览（按 UTS_pred 从高到低） ===")
    preview_k = min(10, pop.shape[0])
    df_preview = pd.DataFrame(pop[:preview_k], columns=feature_order)
    df_preview["UTS_pred"] = fitness[:preview_k]
    print(df_preview.to_string(index=False, float_format="%.6f"))

    # 5) 是否保存结果
    if MASTER_SAVE_SWITCH:
        save_top_candidates(pop, fitness, feature_order, OUTPUT_DIR, top_k=TOP_K_TO_SAVE)
    else:
        print("\nMASTER_SAVE_SWITCH=False，因此本次不保存 CSV 文件。")
