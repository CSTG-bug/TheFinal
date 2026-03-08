from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import importlib.util
import numpy as np
import pandas as pd
from functools import lru_cache


def _bounds_to_tuple(bounds: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for k, v in bounds.items():
        if isinstance(v, dict):
            out[k] = (float(v["min"]), float(v["max"]))
        else:
            out[k] = (float(v[0]), float(v[1]))
    return out


@lru_cache(maxsize=8)
def _load_ga_module(ga_script_path_str: str):
    ga_script_path = Path(ga_script_path_str)
    if not ga_script_path.exists():
        raise FileNotFoundError(f"找不到 GA 脚本：{ga_script_path}")

    spec = importlib.util.spec_from_file_location("ga_uts_xgb_module", str(ga_script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 GA 模块（spec/loader 为空）")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feasibility_check_sum100(
    feature_order: List[str],
    bounds_t: Dict[str, Tuple[float, float]],
    composition_cols: List[str],
    al_col: str = "Al",
    target_sum: float = 100.0,
):
    if al_col not in composition_cols or al_col not in feature_order:
        return

    other_cols = [c for c in composition_cols if c != al_col and c in feature_order]
    if not other_cols:
        return

    al_min, al_max = bounds_t[al_col]
    required_other_min = target_sum - al_max
    required_other_max = target_sum - al_min

    sum_min_other = sum(bounds_t[c][0] for c in other_cols)
    sum_max_other = sum(bounds_t[c][1] for c in other_cols)

    if sum_min_other > required_other_max + 1e-9 or sum_max_other < required_other_min - 1e-9:
        raise ValueError(
            "成分上下限不可行。\n"
            f"需要 other_sum ∈ [{required_other_min:.4f}, {required_other_max:.4f}]\n"
            f"但 other_sum 最小可能={sum_min_other:.4f}, 最大可能={sum_max_other:.4f}\n"
            "请放宽某些元素上限/下限。"
        )


def _repair_sum100_strict(
    x: np.ndarray,
    feature_order: List[str],
    bounds_t: Dict[str, Tuple[float, float]],
    composition_cols: List[str],
    al_col: str = "Al",
    target_sum: float = 100.0,
) -> np.ndarray:
    idx = {n: i for i, n in enumerate(feature_order)}
    comp_cols = [c for c in composition_cols if c in idx]
    if (al_col not in idx) or (al_col not in comp_cols):
        return x

    al_j = idx[al_col]
    other_cols = [c for c in comp_cols if c != al_col]
    other_js = [idx[c] for c in other_cols]

    for c in comp_cols:
        j = idx[c]
        lo, hi = bounds_t[c]
        x[j] = float(np.clip(x[j], lo, hi))

    al_min, al_max = bounds_t[al_col]
    max_other = target_sum - al_min
    min_other = target_sum - al_max

    sum_other = float(np.sum(x[other_js])) if other_js else 0.0

    if sum_other > max_other + 1e-12 and other_js:
        ratio = max_other / (sum_other + 1e-12)
        x[other_js] *= ratio
        for c in other_cols:
            j = idx[c]
            lo, hi = bounds_t[c]
            x[j] = float(np.clip(x[j], lo, hi))
        sum_other = float(np.sum(x[other_js]))

    if sum_other < min_other - 1e-12 and other_js:
        deficit = min_other - sum_other
        headrooms = []
        for c in other_cols:
            j = idx[c]
            _, hi = bounds_t[c]
            headrooms.append(max(0.0, hi - float(x[j])))
        headrooms = np.asarray(headrooms, dtype=float)
        total_headroom = float(headrooms.sum())

        if total_headroom > 1e-12:
            add = deficit * (headrooms / total_headroom)
            for k, c in enumerate(other_cols):
                j = idx[c]
                x[j] = float(x[j] + add[k])
                lo, hi = bounds_t[c]
                x[j] = float(np.clip(x[j], lo, hi))

        sum_other = float(np.sum(x[other_js]))

    x[al_j] = target_sum - sum_other
    x[al_j] = float(np.clip(x[al_j], al_min, al_max))

    sum_other = float(np.sum(x[other_js]))
    total = sum_other + float(x[al_j])
    err = target_sum - total
    if abs(err) > 1e-8 and other_js:
        if err > 0:
            caps = np.array([max(0.0, bounds_t[c][1] - float(x[idx[c]])) for c in other_cols], dtype=float)
        else:
            caps = np.array([max(0.0, float(x[idx[c]]) - bounds_t[c][0]) for c in other_cols], dtype=float)

        cap_sum = float(caps.sum())
        if cap_sum > 1e-12:
            delta = err * (caps / cap_sum)
            for k, c in enumerate(other_cols):
                j = idx[c]
                x[j] = float(x[j] + delta[k])
                lo, hi = bounds_t[c]
                x[j] = float(np.clip(x[j], lo, hi))
            sum_other = float(np.sum(x[other_js]))
            x[al_j] = float(np.clip(target_sum - sum_other, al_min, al_max))

    return x


def run_ga_elite_archive(
    *,
    ga_script_path: Path,
    model,
    feature_order: List[str],
    bounds: Dict[str, Any],
    objective: str = 'max',
    uts_low: Optional[float] = None,
    uts_high: Optional[float] = None,
    pop_size: int = 200,
    max_generations: int = 2000,
    patience: int = 300,
    min_delta: float = 0.01,
    elite_frac: float = 0.1,
    tournament_size: int = 3,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.2,
    mutation_rate: float = 0.1,
    seed: int = 42,
    archive_keep_per_gen: Optional[int] = None,
    archive_max_rows: int = 5000,
    prefer_short_ageing: bool = False,
    prefer_eps: float = 1e-4,
    ageing_time_col: str = "Ageing Time",
    on_progress: Optional[Callable[[int, float, float], None]] = None,
) -> pd.DataFrame:
    ga = _load_ga_module(str(Path(ga_script_path)))
    np.random.seed(seed)

    bounds_t = _bounds_to_tuple(bounds)

    ga.ELITE_FRAC = float(elite_frac)
    ga.TOURNAMENT_SIZE = int(tournament_size)
    ga.CROSSOVER_PROB = float(crossover_prob)
    ga.MUTATION_PROB = float(mutation_prob)
    ga.MUTATION_RATE = float(mutation_rate)

    if ageing_time_col in bounds_t:
        lo_t, hi_t = bounds_t[ageing_time_col]
        hi_t = float(min(hi_t, 48.0))
        bounds_t[ageing_time_col] = (float(lo_t), hi_t)
        ga.AGEING_TIME_MAX = hi_t

    if 'Al' in bounds_t:
        lo_a, hi_a = bounds_t['Al']
        bounds_t['Al'] = (float(lo_a), float(min(hi_a, 99.1)))

    composition_cols = getattr(ga, "COMPOSITION_COLS", [])
    if not composition_cols:
        composition_cols = ["Si","Fe","Cu","Mn","Mg","Cr","Zn","V","Ti","Zr","Li","Ni","Be","Sc","Ag","Bi","Pb","Al"]
    al_col = getattr(ga, "AL_COL", "Al")
    target_sum = float(getattr(ga, "TARGET_SUM", 100.0))

    _feasibility_check_sum100(feature_order, bounds_t, composition_cols, al_col=al_col, target_sum=target_sum)

    elite_size = max(1, int(pop_size * elite_frac))
    if archive_keep_per_gen is None:
        archive_keep_per_gen = elite_size
    archive_keep_per_gen = max(1, int(archive_keep_per_gen))

    def predict_uts(pop_arr: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(pop_arr, columns=feature_order)
        y = model.predict(df)
        return np.asarray(y, dtype=float).reshape(-1)

    
    def compute_fitness(pop_arr: np.ndarray, uts_arr: np.ndarray) -> np.ndarray:
        obj = (objective or "max").lower().strip()
        fit = uts_arr.astype(float).copy()

        if uts_low is not None and uts_high is not None and obj in {"range", "range_mid", "range-mid", "range_max", "range-max"}:
            low = float(min(uts_low, uts_high))
            high = float(max(uts_low, uts_high))

            in_range = (fit >= low) & (fit <= high)
            dist_to_range = np.minimum(np.abs(fit - low), np.abs(fit - high))

            if obj in {"range", "range_mid", "range-mid"}:
                mid = 0.5 * (low + high)
                half = max(1e-6, 0.5 * (high - low))
                dist = np.abs(fit - mid) / half  # 0 ~ 1
                fit = np.where(in_range, 1.0 - dist, -1e6 - dist_to_range)
            else:
                fit = np.where(in_range, fit, -1e6 - dist_to_range)

        if prefer_short_ageing and ageing_time_col in feature_order:
            idx_t = feature_order.index(ageing_time_col)
            fit = fit + float(prefer_eps) * (float(ga.AGEING_TIME_MAX) - pop_arr[:, idx_t])

        return fit
# 初始化
    pop = ga.init_population(bounds_t, feature_order, pop_size)
    pop = np.vstack([ga.repair_individual(ind, bounds_t, feature_order) for ind in pop])
    pop = np.vstack([_repair_sum100_strict(ind, feature_order, bounds_t, composition_cols, al_col=al_col, target_sum=target_sum) for ind in pop])

    uts_pred = predict_uts(pop)
    fitness = compute_fitness(pop, uts_pred)

    best_so_far = -np.inf
    no_improve = 0

    archive_parts: List[pd.DataFrame] = []

    for gen in range(1, int(max_generations) + 1):
        order = np.argsort(-fitness)
        pop = pop[order]
        fitness = fitness[order]
        uts_pred = uts_pred[order]

        best = float(uts_pred[0])
        mean = float(uts_pred.mean())
        best_obj = float(fitness[0])

        if on_progress is not None:
            on_progress(gen, best, mean)

        if best_obj > best_so_far + float(min_delta):
            best_so_far = best_obj
            no_improve = 0
        else:
            no_improve += 1

        # 归档本代精英
        k = min(int(archive_keep_per_gen), pop.shape[0])
        df_elite = pd.DataFrame(pop[:k], columns=feature_order)
        df_elite["UTS_pred"] = uts_pred[:k]
        archive_parts.append(df_elite)

        if no_improve >= int(patience):
            break

        # 产生下一代
        new_pop = pop[:elite_size].copy()

        while new_pop.shape[0] < pop_size:
            p1_idx = ga.tournament_select(fitness, pop_size, ga.TOURNAMENT_SIZE)
            p2_idx = ga.tournament_select(fitness, pop_size, ga.TOURNAMENT_SIZE)
            parent1 = pop[p1_idx]
            parent2 = pop[p2_idx]

            if np.random.rand() < ga.CROSSOVER_PROB:
                child1, child2 = ga.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = ga.mutate(child1, bounds_t, feature_order)
            child2 = ga.mutate(child2, bounds_t, feature_order)

            child1 = ga.repair_individual(child1, bounds_t, feature_order)
            child2 = ga.repair_individual(child2, bounds_t, feature_order)

            child1 = _repair_sum100_strict(child1, feature_order, bounds_t, composition_cols, al_col=al_col, target_sum=target_sum)
            child2 = _repair_sum100_strict(child2, feature_order, bounds_t, composition_cols, al_col=al_col, target_sum=target_sum)

            new_pop = np.vstack([new_pop, child1[None, :], child2[None, :]])

        if new_pop.shape[0] > pop_size:
            new_pop = new_pop[:pop_size]

        pop = new_pop
        uts_pred = predict_uts(pop)
        fitness = compute_fitness(pop, uts_pred)

    df_arch = pd.concat(archive_parts, ignore_index=True)

    # 粗去重
    key = (df_arch[feature_order].astype(float).round(6)).astype(str).agg("|".join, axis=1)
    df_arch = df_arch.loc[~key.duplicated()].copy()

    obj = (objective or "max").lower().strip()
    if obj == "range" and uts_low is not None and uts_high is not None:
        low = float(min(uts_low, uts_high))
        high = float(max(uts_low, uts_high))
        mid = 0.5 * (low + high)

        df_in = df_arch[(df_arch["UTS_pred"] >= low) & (df_arch["UTS_pred"] <= high)].copy()

        if len(df_in) > 0:
            k = min(int(archive_max_rows), len(df_in))

            if len(df_in) > k:
                bins = int(min(20, max(1, k)))
                edges = np.linspace(low, high, bins + 1)

                picks: List[pd.DataFrame] = []
                per_bin = max(1, int(np.ceil(k / bins)))

                for i in range(bins):
                    a = edges[i]
                    b = edges[i + 1]
                    sub = df_in[(df_in["UTS_pred"] >= a) & (df_in["UTS_pred"] < (b if i < bins - 1 else b + 1e-12))]
                    if len(sub) == 0:
                        continue

                    sub = sub.sort_values("UTS_pred")
                    take = min(per_bin, len(sub))
                    idxs = np.linspace(0, len(sub) - 1, num=take).round().astype(int)
                    picks.append(sub.iloc[idxs])

                df_sel = pd.concat(picks, ignore_index=True) if picks else df_in
                df_sel["_dist_mid"] = (df_sel["UTS_pred"] - mid).abs()
                df_sel = df_sel.sort_values(["_dist_mid", "UTS_pred"], ascending=[True, True]).drop(columns=["_dist_mid"])
                df_arch = df_sel.head(k).reset_index(drop=True)
            else:
                df_arch = df_in.reset_index(drop=True)

        else:
            df_arch["_dist_range"] = np.minimum((df_arch["UTS_pred"] - low).abs(), (df_arch["UTS_pred"] - high).abs())
            df_arch = (
                df_arch.sort_values(["_dist_range", "UTS_pred"], ascending=[True, False])
                .drop(columns=["_dist_range"])
                .head(int(archive_max_rows))
                .reset_index(drop=True)
            )
    else:
        df_arch = df_arch.sort_values("UTS_pred", ascending=False).head(int(archive_max_rows)).reset_index(drop=True)

    return df_arch
