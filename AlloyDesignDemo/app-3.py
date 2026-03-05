import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
import os
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
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
# 字段显示名（中文）
# -----------------------------
PROC_NAME_MAP = {
    "SS Temp": "固溶温度",
    "Ageing Temp": "时效温度",
    "Ageing Time": "时效时间",
}
PROC_LABEL_MAP = {
    "SS Temp": "固溶温度（℃）",
    "Ageing Temp": "时效温度（℃）",
    "Ageing Time": "时效时间（h）",
}
PROC_UNIT_MAP = {
    "SS Temp": "℃",
    "Ageing Temp": "℃",
    "Ageing Time": "h",
}


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
    # 目标 UTS 区间
    ss.setdefault("uts_low", 500.0)
    ss.setdefault("uts_high", 550.0)
    # 搜索模式：range_max（区间内最大化） / max（全局极限）
    ss.setdefault("search_mode", "range_max")
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
    ss.setdefault("bounds_override", {})

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


def _open_directory_windows(path_str: str) -> bool:
    """在 Windows 下打开目录。优先用 os.startfile，其次回退 explorer。"""
    try:
        p = Path(path_str).expanduser()
        p.mkdir(parents=True, exist_ok=True)

        # 1) 最稳：直接调用系统关联（等价于资源管理器打开）
        try:
            os.startfile(str(p))  # type: ignore[attr-defined]
            return True
        except Exception:
            pass

        # 2) 回退：显式调用 explorer
        subprocess.Popen(["explorer", str(p)])
        return True
    except Exception:
        return False



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

    # 固定约束：Al 上限必须为 99.1
    if "Al" in out:
        out["Al"]["max"] = 99.1

    # 固定约束：Ageing Time 上限不超过 48（小时）
    if "Ageing Time" in out:
        out["Ageing Time"]["max"] = min(float(out["Ageing Time"]["max"]), 48.0)

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


def _uniform_select_by_uts(df: pd.DataFrame, low: float, high: float,
                           keep_n: int, bins: int = 10) -> pd.DataFrame:
    """
    在 [low, high] 内做分箱均匀抽样，避免结果全部堆在上限（例如全是 550）。
    """
    if df.empty:
        return df

    df2 = df[(df["UTS_pred"] >= low) & (df["UTS_pred"] <= high)].copy()
    if df2.empty:
        return df2

    keep_n = max(int(keep_n), 1)
    bins = max(int(bins), 1)

    edges = np.linspace(low, high, bins + 1)
    df2["_bin"] = pd.cut(df2["UTS_pred"], bins=edges, include_lowest=True)

    per_bin = max(int((keep_n + bins - 1) // bins), 1)

    out = []
    for b in df2["_bin"].cat.categories:
        chunk = df2[df2["_bin"] == b].copy()
        # 每个 bin 里：优先挑“靠近该 bin 中间值”的，避免都挤在 bin 上沿
        if not chunk.empty:
            bin_mid = (b.left + b.right) / 2
            chunk["_d"] = (chunk["UTS_pred"] - bin_mid).abs()
            chunk = chunk.sort_values(["_d", "Ageing Time", "UTS_pred"], ascending=[True, True, False])
            out.append(chunk.head(per_bin))

    out_df = pd.concat(out, ignore_index=True) if out else df2.copy()
    out_df = out_df.drop(columns=[c for c in ["_bin", "_d"] if c in out_df.columns], errors="ignore")

    # 不足 keep_n 时，用区间内剩余的再补齐（按靠近区间中值优先）
    if len(out_df) < keep_n:
        mid = 0.5 * (low + high)
        remain = df2.copy()
        remain["_mid_d"] = (remain["UTS_pred"] - mid).abs()
        remain = remain.sort_values(["_mid_d", "Ageing Time", "UTS_pred"], ascending=[True, True, False])
        out_df = pd.concat([out_df, remain], ignore_index=True)
        out_df = out_df.drop_duplicates().reset_index(drop=True)

    return out_df.head(keep_n)


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

    if st.button("返回主界面", type="primary", key="btn_back_main_from_adv"):
        ss.page = "main"
        st.rerun()

    st.divider()

    tabs = st.tabs(["延伸率", "密度", "价格", "元素上下限", "工艺上下限", "候选合金数量"])

    # --- 延伸率（EL）---
    with tabs[0]:
        use_el = st.checkbox("计算并展示延伸率", value=bool(ss.use_el), key="w_use_el")
        ss.use_el = bool(use_el)

        if ss.use_el:
            constrain_el = st.checkbox(
                "添加约束：延伸率 ≥ 目标值",
                value=bool(ss.constrain_el),
                key="w_constrain_el",
            )
            ss.constrain_el = bool(constrain_el)

            if ss.constrain_el:
                target_el = st.number_input(
                    "目标 延伸率（%）",
                    min_value=0.0,
                    value=float(ss.target_el),
                    step=0.5,
                    key="w_target_el",
                )
                ss.target_el = float(target_el)

    # --- 密度 ---
    with tabs[1]:
        use_density = st.checkbox("计算并展示密度", value=bool(ss.use_density), key="w_use_density")
        ss.use_density = bool(use_density)

        if ss.use_density:
            constrain_density = st.checkbox(
                "添加约束：密度 ≤ 上限",
                value=bool(ss.constrain_density),
                key="w_constrain_density",
            )
            ss.constrain_density = bool(constrain_density)

            if ss.constrain_density:
                max_density = st.number_input(
                    "密度上限（g/cm³）",
                    min_value=0.0,
                    value=float(ss.max_density),
                    step=0.01,
                    key="w_max_density",
                )
                ss.max_density = float(max_density)

    # --- 价格（成本）---
    with tabs[2]:
        use_cost = st.checkbox("计算并展示价格（元/kg）", value=bool(ss.use_cost), key="w_use_cost")
        ss.use_cost = bool(use_cost)

        if ss.use_cost:
            constrain_cost = st.checkbox(
                "添加约束：价格 ≤ 上限",
                value=bool(ss.constrain_cost),
                key="w_constrain_cost",
            )
            ss.constrain_cost = bool(constrain_cost)

            if ss.constrain_cost:
                max_cost = st.number_input(
                    "价格上限（元/kg）",
                    min_value=0.0,
                    value=float(ss.max_cost),
                    step=1.0,
                    key="w_max_cost",
                )
                ss.max_cost = float(max_cost)

        st.divider()

        if st.button("设置元素单价（元/kg）", key="btn_open_price_editor"):
            ss.show_price_editor = True

        if ss.show_price_editor:
            st.info("单价用于计算合金价格。未设置的元素默认单价为 0。")
            price_df = pd.DataFrame(
                [{"Element": e, "Price": float(ss.price_table.get(e, 0.0))} for e in ELEMENT_COLS]
            )
            edited = st.data_editor(
                price_df,
                use_container_width=True,
                num_rows="fixed",
                hide_index=True,
                column_config={
                    "Element": st.column_config.TextColumn("元素", disabled=True),
                    "Price": st.column_config.NumberColumn("价格（元/kg）", min_value=0.0, step=1.0),
                },
                key="price_editor",
            )

            c1, c2, c3 = st.columns(3)
            if c1.button("保存", use_container_width=True, key="btn_save_price"):
                ss.price_table = {r["Element"]: float(r["Price"]) for _, r in edited.iterrows()}
                _save_json(PRICE_TABLE_PATH, ss.price_table)
                st.success("已保存到 models/price_table.json")

            if c2.button("重置全 0", use_container_width=True, key="btn_reset_price"):
                ss.price_table = {e: 0.0 for e in ELEMENT_COLS}
                _save_json(PRICE_TABLE_PATH, ss.price_table)
                st.success("已重置并保存。")

            if c3.button("关闭单价设置", use_container_width=True, key="btn_close_price_editor"):
                ss.show_price_editor = False
                st.rerun()

    # --- 元素上下限（覆盖输入，不展示默认） ---
    with tabs[3]:
        st.caption("留空表示使用默认范围。")
        elem_rows = [{"元素": f, "最小值": None, "最大值": None} for f in ELEMENT_COLS if f in feature_names_uts]
        df = pd.DataFrame(elem_rows)

        for i, r in df.iterrows():
            feat = r["元素"]
            ov = ss.bounds_override.get(feat, {})
            df.at[i, "最小值"] = ov.get("min", None)
            df.at[i, "最大值"] = ov.get("max", None)

        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "元素": st.column_config.TextColumn("元素", disabled=True),
                "最小值": st.column_config.NumberColumn("最小值"),
                "最大值": st.column_config.NumberColumn("最大值"),
            },
            key="elem_bounds_editor",
        )

        c1, c2 = st.columns(2)
        if c1.button("保存元素覆盖", use_container_width=True, key="btn_save_elem_override"):
            for _, r in edited.iterrows():
                feat = str(r["元素"])
                omin = None if pd.isna(r["最小值"]) else float(r["最小值"])
                omax = None if pd.isna(r["最大值"]) else float(r["最大值"])
                if omin is None and omax is None:
                    ss.bounds_override.pop(feat, None)
                else:
                    ss.bounds_override[feat] = {"min": omin, "max": omax}
            st.success("已保存。")

        if c2.button("清空元素覆盖", use_container_width=True, key="btn_clear_elem_override"):
            for f in ELEMENT_COLS:
                ss.bounds_override.pop(f, None)
            st.success("已清空。")

    # --- 工艺上下限 ---
    with tabs[4]:
        st.caption("留空表示使用默认范围")

        # 显示为中文，但保存/读取仍使用原英文特征名作为 key，避免 bounds_override 对不上
        proc_rows = [
            {"工艺": PROC_NAME_MAP.get(f, f), "__key": f, "最小值": None, "最大值": None}
            for f in process_cols
        ]
        df = pd.DataFrame(proc_rows).set_index("__key")

        # 从持久化覆盖中回填（注意：key 是原英文特征名）
        for feat in df.index:
            ov = ss.bounds_override.get(str(feat), {})
            df.at[feat, "最小值"] = ov.get("min", None)
            df.at[feat, "最大值"] = ov.get("max", None)

        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,  # 隐藏英文 key，只展示中文“工艺”列
            column_config={
                "工艺": st.column_config.TextColumn("工艺", disabled=True),
                "最小值": st.column_config.NumberColumn("最小值"),
                "最大值": st.column_config.NumberColumn("最大值"),
            },
            key="proc_bounds_editor",
        )

        c1, c2 = st.columns(2)
        if c1.button("保存工艺覆盖", use_container_width=True, key="btn_save_proc_override"):
            # 注意：iterrows() 的 index 就是原英文特征名
            for feat, r in edited.iterrows():
                feat = str(feat)
                omin = None if pd.isna(r["最小值"]) else float(r["最小值"])
                omax = None if pd.isna(r["最大值"]) else float(r["最大值"])
                if omin is None and omax is None:
                    ss.bounds_override.pop(feat, None)
                else:
                    ss.bounds_override[feat] = {"min": omin, "max": omax}
            st.success("已保存。")

        if c2.button("清空工艺覆盖", use_container_width=True, key="btn_clear_proc_override"):
            for f in process_cols:
                ss.bounds_override.pop(f, None)
            st.success("已清空。")

    # --- 候选合金数量 ---
    with tabs[5]:
        st.caption("设置候选保留数量。")
        archive_max_rows = st.number_input(
            "候选保留数量",
            min_value=1,
            value=int(ss.archive_max_rows),
            step=200,
            key="w_archive_max_rows",
        )
        ss.archive_max_rows = int(archive_max_rows)
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


def render_predict_page(feature_names_uts, process_cols, uts_model, el_model, feature_names_el, bounds_default):
    ss = st.session_state
    st.title("预测模块")

    if st.button("返回主界面", type="primary"):
        ss.page = "main"
        st.rerun()

    if el_model is None or feature_names_el is None:
        st.error("未检测到 EL 模型或 feature_order_el.json，预测模块需要 EL 才能输出延伸率。")
        st.stop()

    st.divider()

    # --- 输入表单：用 form 避免输入过程频繁 rerun 带来的“第一次不保存”问题 ---
    with st.form("predict_form", clear_on_submit=False):
        st.subheader("输入：成分（wt.%）与工艺参数")

        # 1) 成分：建议用户只填非Al元素，Al自动平衡
        st.caption("建议只填写非Al元素；Al 将自动按 100-其余元素之和 计算。")

        elem_cols = [e for e in ELEMENT_COLS if e in feature_names_uts]
        non_al = [e for e in elem_cols if e != "Al"]

        # 成分输入网格
        cols = st.columns(4)
        elem_vals = {}
        for i, e in enumerate(non_al):
            with cols[i % 4]:
                elem_vals[e] = st.number_input(
                    e, min_value=0.0, step=0.1, key=f"pred_elem_{e}"
                )

        other_sum = float(sum(elem_vals.values()))
        al_val = 100.0 - other_sum
        st.write(f"自动计算：Al = {al_val:.2f} wt.%")

        # 2) 工艺输入
        proc_vals = {}
        st.subheader("工艺参数")
        for p in process_cols:
            # 给个温和默认值：bounds 中值（如果没有 bounds 就用 0）
            if p in bounds_default:
                p0 = 0.5 * (float(bounds_default[p]["min"]) + float(bounds_default[p]["max"]))
            else:
                p0 = 0.0

            # 温度字段常见为整数，这里只做轻量处理（不改变模型输入的 float 本质）
            step = 1.0 if "Temp" in p else 0.1
            proc_vals[p] = st.number_input(
                PROC_LABEL_MAP.get(p, p), value=float(p0), step=float(step), key=f"pred_proc_{p}"
            )

        submitted = st.form_submit_button("开始预测", type="primary")

    if not submitted:
        return

    # --- 校验 ---
    if al_val < 0:
        st.error("成分之和超过 100%，请减少某些元素含量。")
        return

    # 遵循你在 GA 中的硬约束：Al 上限 99.1（这里也提示用户）
    if al_val > 99.1 + 1e-9:
        st.error(f"计算得到的 Al={al_val:.2f} 超过 100，请提高其它元素总量或调整范围。")
        return

    # --- 组织一行输入（严格按 feature_order.json 顺序给模型）---
    row = {}
    for f in feature_names_uts:
        if f in non_al:
            row[f] = float(elem_vals.get(f, 0.0))
        elif f == "Al":
            row[f] = float(al_val)
        elif f in proc_vals:
            row[f] = float(proc_vals[f])
        else:
            row[f] = 0.0

    X_uts = pd.DataFrame([row], columns=feature_names_uts)
    uts_pred = float(uts_model.predict(X_uts)[0])

    # 二段：EL = f(成分+工艺+UTS)
    df_tmp = X_uts.copy()
    df_tmp["UTS_pred"] = uts_pred
    df_tmp = add_el_pred(
        df_candidates=df_tmp,
        el_model=el_model,
        feature_names_el=feature_names_el,
        uts_pred_col="UTS_pred",
        el_uts_feature_name="UTS",
        out_col="EL_pred",
    )
    el_pred = float(df_tmp.loc[0, "EL_pred"])

    # --- 大字体展示（隐藏输入为0的元素）---
    st.divider()
    st.markdown("<div style='font-size:30px;font-weight:800'>预测结果</div>", unsafe_allow_html=True)

    # 计算密度与成本（预测模块默认展示；成本依赖单价表 ss.price_table）
    density = None
    cost = None
    try:
        df_dc = calc_density_g_cm3(df_tmp.copy(), out_col="density_g_cm3")
        df_dc = calc_cost_per_kg(df_dc, price_table=ss.price_table, out_col="cost_per_kg")
        if "density_g_cm3" in df_dc.columns:
            density = float(df_dc.loc[0, "density_g_cm3"])
        if "cost_per_kg" in df_dc.columns:
            cost = float(df_dc.loc[0, "cost_per_kg"])
    except Exception:
        pass

    # 两行指标：第一行 UTS/EL，第二行 密度/成本
    r1c1, r1c2 = st.columns(2)
    r1c1.metric("预测 UTS（MPa）", f"{uts_pred:.2f}")
    r1c2.metric("预测 EL（%）", f"{el_pred:.2f}")

    r2c1, r2c2 = st.columns(2)
    r2c1.metric("密度（g/cm³）", f"{density:.4f}" if density is not None else "—")
    r2c2.metric("成本（元/kg）", f"{cost:.2f}" if cost is not None else "—")

    # 只展示非零元素（你要求：用户输入为0的元素不显示）
    show_elems = []
    for e in non_al:
        v = float(elem_vals.get(e, 0.0))
        if abs(v) > 1e-12:
            show_elems.append(f"{e} {v:.2f}")
    show_elems.append(f"Al {al_val:.2f}")  # Al 一定显示

    show_proc = [f"{PROC_NAME_MAP.get(p, p)}:{float(proc_vals[p]):.4g}{PROC_UNIT_MAP.get(p, '')}" for p in process_cols]

    st.markdown(
        f"""
<div style="font-size:22px;line-height:1.6">
<b>成分：</b>{'，'.join(show_elems)}<br/>
<b>工艺：</b>{'，'.join(show_proc)}
</div>
""",
        unsafe_allow_html=True,
    )

# ---- 路由入口：和 adv 一样加一段 ----
if st.session_state.page == "pred":
    render_predict_page(feature_names_uts, process_cols, uts_model, el_model, feature_names_el, bounds_default)
    st.stop()

# -----------------------------
# 主界面（最简）
# -----------------------------
st.title("铝合金智能设计软件")

# 1) 搜索策略：先定义 mode_label（必须在前）
mode_label = st.radio(
    "搜索策略",
    ["区间寻优", "极限寻优"],
    horizontal=True,
    index=0 if st.session_state.get("search_mode", "range_max") != "max" else 1,
)
st.session_state.search_mode = "max" if mode_label == "极限寻优" else "range_max"

# 2) 再根据 mode_label 决定显示哪些输入框
if mode_label == "极限寻优":
    st.session_state.uts_low = st.number_input(
        "UTS 下限（MPa）",
        min_value=0.0,
        value=float(st.session_state.get("uts_low", 0.0)),
        step=10.0,
    )
    # 极限模式下不展示上限；但保留 uts_high 的历史值，切回区间模式还能用
else:
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.uts_low = st.number_input(
            "UTS 下限（MPa）",
            min_value=0.0,
            value=float(st.session_state.get("uts_low", 0.0)),
            step=10.0,
        )
    with c2:
        st.session_state.uts_high = st.number_input(
            "UTS 上限（MPa）",
            min_value=0.0,
            value=float(st.session_state.get("uts_high", 0.0)),
            step=10.0,
        )

    # 仅区间模式需要纠正 low/high
    if st.session_state.uts_low > st.session_state.uts_high:
        st.session_state.uts_low, st.session_state.uts_high = (
            st.session_state.uts_high,
            st.session_state.uts_low,
        )


# 自动纠正：low > high 时交换（不额外提醒用户）
if st.session_state.uts_low > st.session_state.uts_high:
    st.session_state.uts_low, st.session_state.uts_high = st.session_state.uts_high, st.session_state.uts_low

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
    ok = _open_directory_windows(st.session_state.save_dir)
    if not ok:
        st.warning("未能打开输出目录。你可以复制路径到资源管理器地址栏打开。")

b1, b2, b3 = st.columns([3, 1, 1])
with b1:
    start = st.button("开始搜索", type="primary", use_container_width=True)
with b2:
    if st.button("高级约束", use_container_width=True):
        st.session_state.page = "adv"
        st.rerun()
with b3:
    if st.button("预测模块", use_container_width=True):
        st.session_state.page = "pred"
        st.rerun()


# -----------------------------
# 运行 GA + 后处理
# -----------------------------
if start:
    bounds_use = _build_bounds_use(bounds_default, st.session_state.bounds_override)

    progress_slot = st.empty()
    progress_bar = progress_slot.progress(0.0)



    def on_prog(gen: int, best: float, mean: float):
        # 进度按代数推进；最多到 99%，100% 在全部计算完成后置满
        if gen == 1 or gen % 5 == 0:
            pct = min(gen / float(MAX_GENERATIONS), 0.99)
            progress_bar.progress(pct)

    with st.spinner("正在搜索"):
        df_candidates = run_ga_elite_archive(
            ga_script_path=GA_SCRIPT_PATH,
            model=uts_model,
            feature_order=feature_names_uts,
            bounds=bounds_use,
            objective=str(st.session_state.get('search_mode', 'range_max')),
            uts_low=(float(st.session_state.get('uts_low')) if st.session_state.get('search_mode','range_max')!='max' else None),
            uts_high=(float(st.session_state.get('uts_high')) if st.session_state.get('search_mode','range_max')!='max' else None),
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

        # 读取区间
        low = float(st.session_state.get("uts_low", 0.0))
        high = float(st.session_state.get("uts_high", low))

        # 自动纠正：low > high 时交换
        if low > high:
            low, high = high, low

        mode = str(st.session_state.get("search_mode", "range_max"))
        keep_n = int(st.session_state.archive_max_rows)

        if mode == "max":
            # 极限模式：全局最大化 UTS_pred
            df_show = df_out.sort_values(["UTS_pred", "Ageing Time"], ascending=[False, True]).reset_index(drop=True)
            export_df = df_show.head(keep_n).reset_index(drop=True)
            export_name = "alloy_candidates_global_max.xlsx"
        else:
            # 区间模式：只在 [low, high] 内挑选，并在区间内最大化 UTS_pred
            df_in = df_out[(df_out["UTS_pred"] >= low) & (df_out["UTS_pred"] <= high)].copy()

            if len(df_in) == 0:
                # 区间内没有：给“最接近区间”的方案（界面展示仍取最接近的一条）
                df_tmp = df_out.copy()
                df_tmp["_dist"] = 0.0
                df_tmp.loc[df_tmp["UTS_pred"] < low, "_dist"] = (low - df_tmp["UTS_pred"])
                df_tmp.loc[df_tmp["UTS_pred"] > high, "_dist"] = (df_tmp["UTS_pred"] - high)

                df_tmp = df_tmp.sort_values(["_dist", "Ageing Time", "UTS_pred"], ascending=[True, True, False])
                df_show = df_tmp.drop(columns=["_dist"]).reset_index(drop=True)

                export_df = df_show.head(keep_n).reset_index(drop=True)
                export_name = "alloy_candidates_nearest_range.xlsx"
            else:
                df_show = df_in.sort_values(["UTS_pred", "Ageing Time"], ascending=[False, True]).reset_index(drop=True)
                export_df = df_show.head(keep_n).reset_index(drop=True)
                export_name = "alloy_candidates_in_range_max.xlsx"

        # Excel 里统一按强度由高到低排序
        if "UTS_pred" in export_df.columns:
            export_df = export_df.sort_values(["UTS_pred"], ascending=False).reset_index(drop=True)
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

    progress_bar.progress(1.0)
    st.divider()
    st.subheader("最优方案")

    mcols = st.columns(4)
    mcols[0].metric("UTS (MPa)", f'{best["UTS_pred"]:.2f}')
    if "EL_pred" in best:
        mcols[1].metric("延伸率 (%)", f'{best["EL_pred"]:.2f}')
    if "density_g_cm3" in best:
        mcols[2].metric("密度 (g/cm³)", f'{best["density_g_cm3"]:.4f}')
    if "cost_per_kg" in best:
        mcols[3].metric("价格 (元/kg)", f'{best["cost_per_kg"]:.2f}')

    best_df = pd.DataFrame([best])[show_cols]
    best_df.index = [1]
    # 表格列名中文映射
    rename_map = {
        "SS Temp": "固溶温度（℃）",
        "Ageing Temp": "时效温度（℃）",
        "Ageing Time": "时效时间（h）",
        "UTS_pred": "预测UTS（MPa）",
        "EL_pred": "预测EL（%）",
        "density_g_cm3": "密度（g/cm³）",
        "cost_per_kg": "成本（元/kg）",
    }
    best_df_show = best_df.rename(columns={k: v for k, v in rename_map.items() if k in best_df.columns})
    st.dataframe(best_df_show, use_container_width=True)

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
