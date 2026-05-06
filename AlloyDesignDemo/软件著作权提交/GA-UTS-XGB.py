from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from joblib import load

# ====================== 配置区 ======================

# 1) 原始 X_train路径
RAW_X_TRAIN_PATH = r"D:\MLDesignAl\TheFinal\Data\ElementTreatmentEl-UTS\output-exceptEL\exceptEL-X_train_raw.csv"

# 2) 训练好的 XGBoost 模型路径
MODEL_PATH = r"D:\MLDesignAl\TheFinal\XGBoost\ElementTreatmentEl-UTS\output-exceptEL\XGB_best_model.joblib"

# 3) 结果输出目录
OUTPUT_DIR = Path(r"D:\MLDesignAl\TheFinal\DesignSearch\GA-UTS-XGB")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 4) 是否保存结果，以及保存多少条
MASTER_SAVE_SWITCH = False
TOP_K_TO_SAVE = 200

# 5) 设计空间的手动覆盖
FEATURE_BOUNDS_OVERRIDE: Dict[str, Tuple[float, float]] = {
    "Si": (0.0, 0.15),
    "Fe": (0.0, 0.15),
    "Cr": (0.0, 0.3),
    "V" : (0.0, 0.1),
    "Zr": (0.0, 0.2),
    "Li": (0.0, 0.0),
    "Ni": (0.0, 0.2),
    "Sc": (0.0, 0.0),
    "Ag": (0.0, 0.0),
    "Bi": (0.0, 0.0),
    "Pb": (0.0, 0.0),
    "Ageing Time": (0.0, 48.0),
}

# 6) GA 参数
POP_SIZE = 200
N_GENERATIONS = 1000
ELITE_FRAC = 0.1
TOURNAMENT_SIZE = 3
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.2
MUTATION_RATE = 0.1

RANDOM_SEED = 42

# 7) 约束相关设置
COMPOSITION_COLS: list[str] = [
    "Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Zn", "V", "Ti", "Zr", "Li", "Ni", "Be", "Sc", "Ag", "Bi", "Pb", "Al"
]
AL_COL = "Al"
TARGET_SUM = 100.0

AGEING_TIME_COL = "Ageing Time"
AGEING_TIME_MAX = 48.0
AGEING_TIME_STEP: Optional[float] = None

PREFER_SHORT_AGEING_TIME = True
PREFER_SHORT_AGEING_EPS = 1e-4

# ======================================================================


def load_raw_x_and_bounds(path: str) -> tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    raw_df = pd.read_csv(path)
    feature_names = raw_df.columns.tolist()

    bounds: Dict[str, Tuple[float, float]] = {}
    for col in feature_names:
        col_min = float(raw_df[col].min())
        col_max = float(raw_df[col].max())
        bounds[col] = (col_min, col_max)

    for col, (lo, hi) in FEATURE_BOUNDS_OVERRIDE.items():
        if col not in bounds:
            raise KeyError(f"你在 FEATURE_BOUNDS_OVERRIDE 中指定了列 '{col}'，"
                           f"但在 RAW_X_TRAIN 中未找到该列。")
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


def evaluate_population(pop: np.ndarray, feature_order: list[str], model) -> np.ndarray:
    df = pd.DataFrame(pop, columns=feature_order)
    uts_pred = model.predict(df)
    uts_pred = np.asarray(uts_pred, dtype=float).reshape(-1)

    if PREFER_SHORT_AGEING_TIME and (AGEING_TIME_COL in df.columns):
        uts_pred = uts_pred + PREFER_SHORT_AGEING_EPS * (AGEING_TIME_MAX - df[AGEING_TIME_COL].values)
    return uts_pred


def tournament_select(fitness: np.ndarray, k: int, t_size: int) -> int:
    indices = np.random.choice(k, size=t_size, replace=False)
    best_idx = indices[np.argmax(fitness[indices])]
    return best_idx


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
            if new_ind[j] < lo:
                new_ind[j] = lo
            elif new_ind[j] > hi:
                new_ind[j] = hi
    return new_ind


def repair_individual(ind: np.ndarray,
                      bounds: Dict[str, Tuple[float, float]],
                      feature_order: list[str]) -> np.ndarray:
    idx_map = {name: i for i, name in enumerate(feature_order)}
    x = ind.copy()

    if AGEING_TIME_COL in idx_map:
        i = idx_map[AGEING_TIME_COL]
        lo, hi = bounds[AGEING_TIME_COL]
        hi = min(hi, AGEING_TIME_MAX)
        x[i] = float(np.clip(x[i], lo, hi))
        if AGEING_TIME_STEP is not None and AGEING_TIME_STEP > 0:
            x[i] = round(x[i] / AGEING_TIME_STEP) * AGEING_TIME_STEP
            x[i] = float(np.clip(x[i], lo, hi))

    comp_cols = [c for c in COMPOSITION_COLS if c in idx_map]
    for col in comp_cols:
        j = idx_map[col]
        lo, hi = bounds[col]
        x[j] = float(np.clip(x[j], lo, hi))

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


def run_ga_search(raw_df: pd.DataFrame,
                  bounds: Dict[str, Tuple[float, float]],
                  model,
                  pop_size: int = POP_SIZE,
                  n_generations: int = N_GENERATIONS) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_order = raw_df.columns.tolist()
    n_features = len(feature_order)

    pop = init_population(bounds, feature_order, pop_size)

    pop = np.vstack([repair_individual(ind, bounds, feature_order) for ind in pop])

    fitness = evaluate_population(pop, feature_order, model)

    print("\n=== 开始 GA 进化搜索 ===")
    print(f"初始种群大小: {pop_size}, 特征数: {n_features}, 迭代代数: {n_generations}")

    elite_size = max(1, int(pop_size * ELITE_FRAC))

    for gen in range(1, n_generations + 1):
        order = np.argsort(-fitness)
        pop = pop[order]
        fitness = fitness[order]

        best = fitness[0]
        mean = fitness.mean()
        std = fitness.std()

        print(f"Gen {gen:03d} | best UTS={best:.3f}, mean={mean:.3f}, std={std:.3f}")

        new_pop = pop[:elite_size].copy()

        while new_pop.shape[0] < pop_size:
            p1_idx = tournament_select(fitness, pop_size, TOURNAMENT_SIZE)
            p2_idx = tournament_select(fitness, pop_size, TOURNAMENT_SIZE)
            parent1 = pop[p1_idx]
            parent2 = pop[p2_idx]

            if np.random.rand() < CROSSOVER_PROB:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = repair_individual(mutate(child1, bounds, feature_order), bounds, feature_order)
            child2 = repair_individual(mutate(child2, bounds, feature_order), bounds, feature_order)

            new_pop = np.vstack([new_pop, child1[None, :], child2[None, :]])

        if new_pop.shape[0] > pop_size:
            new_pop = new_pop[:pop_size]

        pop = new_pop
        fitness = evaluate_population(pop, feature_order, model)

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

    raw_df, bounds = load_raw_x_and_bounds(RAW_X_TRAIN_PATH)

    print("\n=== 加载 XGBoost 模型 ===")
    print(f"模型路径: {MODEL_PATH}")
    model = load(MODEL_PATH)
    print(f"模型类型: {type(model)}")

    pop, fitness, feature_order = run_ga_search(
        raw_df=raw_df,
        bounds=bounds,
        model=model,
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS
    )

    print("\n=== Top-10 方案预览（按 UTS_pred 从高到低） ===")
    preview_k = min(10, pop.shape[0])
    df_preview = pd.DataFrame(pop[:preview_k], columns=feature_order)
    df_preview["UTS_pred"] = fitness[:preview_k]
    print(df_preview.to_string(index=False, float_format="%.6f"))

    if MASTER_SAVE_SWITCH:
        save_top_candidates(pop, fitness, feature_order, OUTPUT_DIR, top_k=TOP_K_TO_SAVE)
    else:
        print("\nMASTER_SAVE_SWITCH=False，因此本次不保存 CSV 文件。")
