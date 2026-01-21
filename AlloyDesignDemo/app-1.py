import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import streamlit as st
from postprocess import (
    add_el_pred,
    calc_density_g_cm3,
    calc_cost_per_kg,
    apply_constraints,
    ConstraintOptions,
    ELEMENT_COLS,
)
from GA_wrapper import run_ga_elite_archive
import subprocess


# -----------------------------
# 基础配置
# -----------------------------
POP_SIZE = 200
MAX_GENERATIONS = 2000
PATIENCE = 300
MIN_DELTA = 0.01
SEED = 42

ELITE_FRAC = 0.10
TOURNAMENT_SIZE = 3
CROSSOVER_PROB = 0.90
MUTATION_PROB = 0.20
MUTATION_RATE = 0.10

# 同分择优：默认启用（偏好更短时效时间）
PREFER_SHORT_AGEING = True
PREFER_SHORT_AGEING_EPS = 1e-4


# -----------------------------
# 路径
# -----------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR_DEFAULT = Path.home() / "Documents" / "AlloyDesignOutputs"
OUTPUTS_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)

UTS_MODEL_PATH = MODELS_DIR / "XGB_best_model.joblib"
EL_MODEL_PATH = MODELS_DIR / "EL_XGB_best_model.joblib"

FEATURE_ORDER_UTS_PATH = MODELS_DIR / "feature_order.json"
FEATURE_BOUNDS_UTS_PATH = MODELS_DIR / "feature_bounds.json"

FEATURE_ORDER_EL_PATH = MODELS_DIR / "feature_order_el.json"
PRICE_TABLE_PATH = MODELS_DIR / "price_table.json"

GA_SCRIPT_PATH = BASE_DIR / "GA-UTS-XGB.py"


# -----------------------------
# 工具函数
# -----------------------------
@st.cache_resource
def _load_model(path: Path):
    return joblib.load(path)


@st.cache_data
def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _init_state():
    ss = st.session_state
    ss.setdefault("page", "main")

    # 主界面输入
    ss.setdefault("target_uts", 700.0)
    ss.setdefault("save_dir", str(OUTPUTS_DIR_DEFAULT))
    ss.setdefault("archive_max_rows", 5000)

    # 高级设置：启用/约束
    ss.setdefault("use_el", False)
    ss.setdefault("constrain_el", False)
    ss.setdefault("target_el", 8.0)

    ss.setdefault("use_density", False)
    ss.setdefault("constrain_density", False)
    ss.setdefault("max_density", 2.90)

    ss.setdefault("use_cost", False)
    ss.setdefault("constrain_cost", False)
    ss.setdefault("max_cost", 80.0)

    # 上下限覆盖：只存用户填写的 override（None 表示不覆盖）
    ss.setdefault("bounds_override", {})  # feature -> {"min": Optional[float], "max": Optional[float]}

    # 单价表
    if "price_table" not in ss:
        default_price = {e: 0.0 for e in ELEMENT_COLS}
        if PRICE_TABLE_PATH.exists():
            try:
                data = _load_json(PRICE_TABLE_PATH)
                for e in ELEMENT_COLS:
                    default_price[e] = float(data.get(e, 0.0))
            except Exception:
                pass
        ss["price_table"] = default_price

    ss.setdefault("show_price_editor", False)


def _choose_directory_windows() -> Optional[str]:
    """
    打包版最稳的目录选择方式：
    用 PowerShell 调用 .NET FolderBrowserDialog（独立进程弹窗），
    避免 Streamlit 线程环境导致的对话框失效。
    """
    try:
        ps_script = r'''
Add-Type -AssemblyName System.Windows.Forms
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = "请选择结果保存目录"
$dlg.ShowNewFolderButton = $true
if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    $dlg.SelectedPath
}
'''
        cp = subprocess.run(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        path = (cp.stdout or "").strip()
        return path if path else None
    except Exception:
        return None


def _build_bounds_use(bounds_default: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    规则：
      - 用户没填：用训练集默认 bounds
      - 用户填了：先 clip 到训练集 bounds，再使用（不提示）
      - 若 clip 后 min>max：忽略该特征的覆盖（回退默认）
    """
    out = {k: {"min": float(v["min"]), "max": float(v["max"])} for k, v in bounds_default.items()}

    for feat, ov in overrides.items():
        if feat not in out:
            continue
        dmin = float(out[feat]["min"])
        dmax = float(out[feat]["max"])

        omin = ov.get("min", None)
        omax = ov.get("max", None)

        nmin = None if omin is None else float(max(dmin, min(float(omin), dmax)))
        nmax = None if omax is None else float(max(dmin, min(float(omax), dmax)))

        if nmin is None and nmax is None:
            continue

        # 只有一边填写：另一边用默认
        if nmin is None:
            nmin = dmin
        if nmax is None:
            nmax = dmax

        # min>max：忽略覆盖（回退默认，不提示）
        if nmin > nmax:
            continue

        out[feat]["min"] = nmin
        out[feat]["max"] = nmax

    if "Al" in out:
        out["Al"]["max"] = 99.1

    return out


def _round_rebalance_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    输出格式：
      - 元素：2位并平衡Al使总和=100
      - SS/Ageing Temp：整数
      - Ageing Time：1位
      - UTS/EL：2位；密度4位；价格2位
    去重：按输出后的元素+工艺 key 去重，避免重复/相似结果。
    """
    out = df.copy()

    elem_cols = ELEMENT_COLS[:]
    non_al = [c for c in elem_cols if c != "Al"]

    for c in non_al:
        if c in out.columns:
            out[c] = out[c].astype(float).round(2)

    out["Al"] = (100.0 - out[non_al].sum(axis=1)).round(2)

    if "SS Temp" in out.columns:
        out["SS Temp"] = out["SS Temp"].astype(float).round(0).astype(int)
    if "Ageing Temp" in out.columns:
        out["Ageing Temp"] = out["Ageing Temp"].astype(float).round(0).astype(int)
    if "Ageing Time" in out.columns:
        out["Ageing Time"] = out["Ageing Time"].astype(float).round(1)

    if "UTS_pred" in out.columns:
        out["UTS_pred"] = out["UTS_pred"].astype(float).round(2)
    if "EL_pred" in out.columns:
        out["EL_pred"] = out["EL_pred"].astype(float).round(2)
    if "density_g_cm3" in out.columns:
        out["density_g_cm3"] = out["density_g_cm3"].astype(float).round(4)
    if "cost_per_kg" in out.columns:
        out["cost_per_kg"] = out["cost_per_kg"].astype(float).round(2)

    key_cols = elem_cols + ["SS Temp", "Ageing Temp", "Ageing Time"]
    exist_key_cols = [c for c in key_cols if c in out.columns]
    out["_dedup_key"] = out[exist_key_cols].astype(str).agg("|".join, axis=1)

    out = out.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"]).reset_index(drop=True)
    return out


def _export_excel_bytes(df: pd.DataFrame, cols: list[str]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df[cols].to_excel(writer, index=False, sheet_name="candidates")
    return buf.getvalue()


# -----------------------------
# 页面渲染：高级约束页
# -----------------------------
def render_advanced_page(feature_names_uts, process_cols):
    ss = st.session_state

    st.title("高级约束")

    if st.button("返回主界面", type="primary"):
        ss.page = "main"
        st.rerun()

    st.divider()

    tabs = st.tabs(["EL", "密度", "价格", "元素上下限", "工艺上下限", "GA 参数"])

    # --- EL ---
    with tabs[0]:
        ss.use_el = st.checkbox("计算并展示 EL", value=ss.use_el)
        if ss.use_el:
            ss.constrain_el = st.checkbox("添加约束：EL ≥ 目标值", value=ss.constrain_el)
            if ss.constrain_el:
                ss.target_el = st.number_input("目标 EL（%）", min_value=0.0, value=float(ss.target_el), step=0.5)

    # --- 密度 ---
    with tabs[1]:
        ss.use_density = st.checkbox("计算并展示密度", value=ss.use_density)
        if ss.use_density:
            ss.constrain_density = st.checkbox("添加约束：密度 ≤ 上限", value=ss.constrain_density)
            if ss.constrain_density:
                ss.max_density = st.number_input("密度上限（g/cm³）", min_value=0.0, value=float(ss.max_density), step=0.01)

    # --- 价格 ---
    with tabs[2]:
        ss.use_cost = st.checkbox("计算并展示价格（元/kg）", value=ss.use_cost)
        if ss.use_cost:
            ss.constrain_cost = st.checkbox("添加约束：价格 ≤ 上限", value=ss.constrain_cost)
            if ss.constrain_cost:
                ss.max_cost = st.number_input("价格上限（元/kg）", min_value=0.0, value=float(ss.max_cost), step=1.0)

        st.divider()
        if st.button("设置元素单价（元/kg）"):
            ss.show_price_editor = True

        if ss.show_price_editor:
            st.info("单价用于计算 cost_per_kg。未设置的元素默认单价为 0。")
            price_df = pd.DataFrame([{"element": e, "price": float(ss.price_table.get(e, 0.0))} for e in ELEMENT_COLS])
            edited = st.data_editor(price_df, use_container_width=True, num_rows="fixed", hide_index=True)

            c1, c2, c3 = st.columns(3)
            if c1.button("保存", use_container_width=True):
                ss.price_table = {r["element"]: float(r["price"]) for _, r in edited.iterrows()}
                _save_json(PRICE_TABLE_PATH, ss.price_table)
                st.success("已保存到 models/price_table.json")

            if c2.button("重置全 0", use_container_width=True):
                ss.price_table = {e: 0.0 for e in ELEMENT_COLS}
                _save_json(PRICE_TABLE_PATH, ss.price_table)
                st.success("已重置并保存。")

            if c3.button("关闭单价设置", use_container_width=True):
                ss.show_price_editor = False
                st.rerun()

    # --- 元素上下限（覆盖输入，不展示默认） ---
    with tabs[3]:
        st.caption("留空表示使用默认范围。")
        elem_rows = [{"feature": f, "min_override": None, "max_override": None} for f in ELEMENT_COLS if f in feature_names_uts]
        df = pd.DataFrame(elem_rows)

        for i, r in df.iterrows():
            feat = r["feature"]
            ov = ss.bounds_override.get(feat, {})
            df.at[i, "min_override"] = ov.get("min", None)
            df.at[i, "max_override"] = ov.get("max", None)

        edited = st.data_editor(df, use_container_width=True, num_rows="fixed", hide_index=True)

        c1, c2 = st.columns(2)
        if c1.button("保存元素覆盖", use_container_width=True):
            for _, r in edited.iterrows():
                feat = str(r["feature"])
                omin = None if pd.isna(r["min_override"]) else float(r["min_override"])
                omax = None if pd.isna(r["max_override"]) else float(r["max_override"])
                if omin is None and omax is None:
                    ss.bounds_override.pop(feat, None)
                else:
                    ss.bounds_override[feat] = {"min": omin, "max": omax}
            st.success("已保存。")

        if c2.button("清空元素覆盖", use_container_width=True):
            for f in ELEMENT_COLS:
                ss.bounds_override.pop(f, None)
            st.success("已清空。")

    # --- 工艺上下限 ---
    with tabs[4]:
        st.caption("留空表示使用默认范围")
        proc_rows = [{"feature": f, "min_override": None, "max_override": None} for f in process_cols]
        df = pd.DataFrame(proc_rows)

        for i, r in df.iterrows():
            feat = r["feature"]
            ov = ss.bounds_override.get(feat, {})
            df.at[i, "min_override"] = ov.get("min", None)
            df.at[i, "max_override"] = ov.get("max", None)

        edited = st.data_editor(df, use_container_width=True, num_rows="fixed", hide_index=True)

        c1, c2 = st.columns(2)
        if c1.button("保存工艺覆盖", use_container_width=True):
            for _, r in edited.iterrows():
                feat = str(r["feature"])
                omin = None if pd.isna(r["min_override"]) else float(r["min_override"])
                omax = None if pd.isna(r["max_override"]) else float(r["max_override"])
                if omin is None and omax is None:
                    ss.bounds_override.pop(feat, None)
                else:
                    ss.bounds_override[feat] = {"min": omin, "max": omax}
            st.success("已保存。")

        if c2.button("清空工艺覆盖", use_container_width=True):
            for f in process_cols:
                ss.bounds_override.pop(f, None)
            st.success("已清空。")

    # --- GA 参数 ---
    with tabs[5]:
        st.caption("GA 参数候选保留数量。")
        ss.archive_max_rows = st.number_input("候选保留数量", min_value=200, value=int(ss.archive_max_rows), step=200)


# -----------------------------
# 主程序
# -----------------------------
st.set_page_config(page_title="铝合金智能设计软件", layout="wide")

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

_init_state()

# 启动检查
missing = []
for pth, msg in [
    (UTS_MODEL_PATH, "缺少 UTS 模型：models/XGB_best_model.joblib"),
    (FEATURE_ORDER_UTS_PATH, "缺少 UTS 特征顺序：models/feature_order.json"),
    (FEATURE_BOUNDS_UTS_PATH, "缺少默认上下限：models/feature_bounds.json"),
    (GA_SCRIPT_PATH, "缺少 GA 脚本：AlloyDesignDemo/GA-UTS-XGB.py"),
]:
    if not pth.exists():
        missing.append(msg)
if missing:
    st.error("启动检查失败：\n- " + "\n- ".join(missing))
    st.stop()

uts_model = _load_model(UTS_MODEL_PATH)
feature_names_uts = _load_json(FEATURE_ORDER_UTS_PATH)
bounds_default = _load_json(FEATURE_BOUNDS_UTS_PATH)
if "Al" in bounds_default:
    bounds_default["Al"]["max"] = 99.1

process_cols = [f for f in feature_names_uts if f not in ELEMENT_COLS]

# EL 模型
el_model = None
feature_names_el = None
if EL_MODEL_PATH.exists() and FEATURE_ORDER_EL_PATH.exists():
    el_model = _load_model(EL_MODEL_PATH)
    feature_names_el = _load_json(FEATURE_ORDER_EL_PATH)

# 高级约束页
if st.session_state.page == "adv":
    render_advanced_page(feature_names_uts, process_cols)
    st.stop()

# -----------------------------
# 主界面（最简）
# -----------------------------
st.title("铝合金智能设计软件")

st.session_state.target_uts = st.number_input(
    "目标 UTS（MPa）",
    min_value=0.0,
    value=float(st.session_state.target_uts),
    step=10.0,
)

st.session_state.save_dir = st.text_input("结果保存目录", value=str(st.session_state.save_dir))

col_a, col_b = st.columns([1, 1])
if col_a.button("选择目录"):
    chosen = _choose_directory_windows()
    if chosen:
        st.session_state.save_dir = chosen
        st.rerun()
    else:
        st.warning("未能打开系统目录选择窗口。请在输入框中手动填写路径。")

if col_b.button("打开输出目录"):
    st.info(f"输出目录：{st.session_state.save_dir}")

b1, b2 = st.columns([3, 1])
with b1:
    start = st.button("开始搜索", type="primary", use_container_width=True)
with b2:
    if st.button("高级约束", use_container_width=True):
        st.session_state.page = "adv"
        st.rerun()

# -----------------------------
# 运行 GA + 后处理
# -----------------------------
if start:
    bounds_use = _build_bounds_use(bounds_default, st.session_state.bounds_override)

    status = st.empty()
    status.write("GA 进程启动中...")

    def on_prog(gen: int, best: float, mean: float):
        if gen == 1 or gen % 5 == 0:
            status.write(f"GA 迭代中：第 {gen} 代 | 当前最优 UTS_pred = {best:.2f} MPa")

    with st.spinner("正在运行 GA 搜索..."):
        df_candidates = run_ga_elite_archive(
            ga_script_path=GA_SCRIPT_PATH,
            model=uts_model,
            feature_order=feature_names_uts,
            bounds=bounds_use,
            pop_size=POP_SIZE,
            max_generations=MAX_GENERATIONS,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            elite_frac=ELITE_FRAC,
            tournament_size=TOURNAMENT_SIZE,
            crossover_prob=CROSSOVER_PROB,
            mutation_prob=MUTATION_PROB,
            mutation_rate=MUTATION_RATE,
            seed=SEED,
            archive_keep_per_gen=None,
            archive_max_rows=int(st.session_state.archive_max_rows),
            prefer_short_ageing=PREFER_SHORT_AGEING,
            prefer_eps=PREFER_SHORT_AGEING_EPS,
            on_progress=on_prog,
        )

    status.write("GA 已完成，正在进行二阶段预测/约束过滤/去重与导出...")

    df = df_candidates.copy()

    # EL
    if st.session_state.use_el:
        if el_model is None or feature_names_el is None:
            st.error("已启用 EL，但未检测到 EL 模型或 feature_order_el.json。请检查 models/ 目录。")
            st.stop()
        df = add_el_pred(
            df_candidates=df,
            el_model=el_model,
            feature_names_el=feature_names_el,
            uts_pred_col="UTS_pred",
            el_uts_feature_name="UTS",
            out_col="EL_pred",
        )

    # 密度/价格
    if st.session_state.use_density:
        df = calc_density_g_cm3(df, out_col="density_g_cm3")
    if st.session_state.use_cost:
        df = calc_cost_per_kg(df, price_table=st.session_state.price_table, out_col="cost_per_kg")

    # 约束过滤
    opts = ConstraintOptions(
        use_el=bool(st.session_state.constrain_el and st.session_state.use_el),
        target_el=float(st.session_state.target_el) if (st.session_state.constrain_el and st.session_state.use_el) else None,
        use_density=bool(st.session_state.constrain_density and st.session_state.use_density),
        max_density=float(st.session_state.max_density) if (st.session_state.constrain_density and st.session_state.use_density) else None,
        use_cost=bool(st.session_state.constrain_cost and st.session_state.use_cost),
        max_cost=float(st.session_state.max_cost) if (st.session_state.constrain_cost and st.session_state.use_cost) else None,
    )
    df_f = apply_constraints(df, opts)

    df_out = _round_rebalance_and_dedup(df_f)

    if len(df_out) == 0:
        st.error("当前约束下没有任何候选方案。请放宽约束或调整上下限。")
        st.stop()

    target_uts = float(st.session_state.target_uts)
    df_meet = df_out[df_out["UTS_pred"] >= target_uts].copy() if target_uts > 0 else df_out.copy()

    if len(df_meet) == 0:
        df_show = df_out.sort_values("UTS_pred", ascending=False).reset_index(drop=True)
        export_df = df_out
        export_name = "alloy_candidates_all.xlsx"
        status.write("提示：未能达到目标 UTS，将展示最强方案，并导出全部候选。")
    else:
        df_show = df_meet.sort_values("UTS_pred", ascending=False).reset_index(drop=True)
        export_df = df_meet
        export_name = "alloy_candidates_meet_target.xlsx"
        status.write("已找到满足目标 UTS 的候选集合。")

    best = df_show.iloc[0].to_dict()

    save_dir = Path(st.session_state.save_dir).expanduser()
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        save_dir = OUTPUTS_DIR_DEFAULT
        save_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_path = save_dir / export_name.replace(".xlsx", f"_{ts}.xlsx")

    show_cols = [c for c in (ELEMENT_COLS + ["SS Temp", "Ageing Temp", "Ageing Time", "UTS_pred", "EL_pred", "density_g_cm3", "cost_per_kg"]) if c in export_df.columns]
    excel_bytes = _export_excel_bytes(export_df, show_cols)
    try:
        saved_path.write_bytes(excel_bytes)
        saved_ok = True
    except Exception:
        saved_ok = False

    st.divider()
    st.subheader("最优方案")

    mcols = st.columns(4)
    mcols[0].metric("UTS_pred (MPa)", f'{best["UTS_pred"]:.2f}')
    if "EL_pred" in best:
        mcols[1].metric("EL_pred (%)", f'{best["EL_pred"]:.2f}')
    if "density_g_cm3" in best:
        mcols[2].metric("density (g/cm³)", f'{best["density_g_cm3"]:.4f}')
    if "cost_per_kg" in best:
        mcols[3].metric("cost (元/kg)", f'{best["cost_per_kg"]:.2f}')

    best_df = pd.DataFrame([best])[show_cols]
    best_df.index = [1]
    st.dataframe(best_df, use_container_width=True)

    st.info(f"候选数：{len(df_show)}。")
    if saved_ok:
        st.success(f"Excel 已保存到：{saved_path}")
    else:
        st.warning("未能写入到你指定的目录（可能权限或路径问题）。你仍可以使用下面的下载按钮获取 Excel。")

    st.download_button(
        label="下载候选方案 Excel",
        data=excel_bytes,
        file_name=saved_path.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    status.empty()
