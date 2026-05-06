import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
import os
import sys
import base64
import html
from typing import Dict, Any, Optional, List, Tuple

import joblib
import openpyxl
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

PRIORITY_OPTIONS = ["强度", "延伸率", "密度", "价格"]
PRIORITY_META = {
    "强度": ("UTS_pred", False),
    "延伸率": ("EL_pred", False),
    "密度": ("density_g_cm3", True),
    "价格": ("cost_per_kg", True),
}

DEFAULT_PRICE_FALLBACK = {
    "Si": 13.0,
    "Fe": 5.0,
    "Cu": 102.0,
    "Mn": 20.0,
    "Mg": 18.0,
    "Cr": 62.0,
    "Zn": 24.0,
    "V": 330.0,
    "Ti": 56.0,
    "Zr": 290.0,
    "Li": 480.0,
    "Ni": 143.0,
    "Be": 8225.0,
    "Sc": 3000.0,
    "Ag": 20.0,
    "Bi": 145.0,
    "Pb": 16.0,
    "Al": 25.0,
}

DEFAULT_HELP_TEXTS = {
    "go_design": "进入设计模块页面，用于设置目标强度、保存目录、执行候选方案搜索，并查看最优组合与全部候选结果。",
    "go_pred": "进入预测模块页面，用于输入指定合金成分与热处理工艺，快速预测材料性能。",
    "design_back_home": "返回软件首页，不会清除当前会话中已经设置的参数。",
    "design_adv": "打开高级约束页面，可设置延伸率、密度、价格、元素上下限、工艺上下限、候选数量及多级排序规则。",
    "design_save_toggle": "展开或收起结果保存目录区域，可设置搜索结果 Excel 文件的保存位置。",
    "design_choose_dir": "打开系统目录选择窗口，手动选择候选结果文件的保存文件夹。",
    "design_open_dir": "直接打开当前结果保存目录，便于查看历史导出文件。",
    "design_start_search": "根据当前目标性能、约束条件和设计空间设置启动智能搜索。搜索过程中界面将临时锁定，防止误操作。",
    "design_download": "下载当前搜索得到的候选方案 Excel 文件。",
    "adv_back_design": "返回设计模块页面，并保留当前已设置的高级约束参数。",
    "adv_open_price_editor": "打开元素单价编辑表格，用于查看、修改和保存价格设置。",
    "adv_save_price": "保存当前编辑后的元素单价，并在后续价格计算与结果排序中自动生效。",
    "adv_reset_price_default": "将当前元素单价重置为系统默认真实价格，而不是重置为 0。",
    "adv_close_price_editor": "关闭元素单价编辑表格，但不会清除已经保存的价格数据。",
    "adv_save_elem_override": "保存当前元素上下限覆盖设置，并在设计模块搜索时作为有效边界参与计算。",
    "adv_clear_elem_override": "清空全部元素上下限覆盖设置，恢复为系统默认元素范围。",
    "adv_save_proc_override": "保存当前热处理工艺上下限覆盖设置，并在设计模块搜索时作为有效边界参与计算。",
    "adv_clear_proc_override": "清空全部工艺上下限覆盖设置，恢复为系统默认工艺范围。",
    "adv_sort_priority": "设置候选方案的一级、二级、三级和四级排序优先级。若排序中包含延伸率、密度或价格，系统会自动计算对应指标。",
    "pred_back_home": "返回软件首页，不清空当前预测输入内容。",
    "pred_start": "根据当前输入的合金成分和工艺参数调用机器学习模型进行正向预测。"
}

# -----------------------------
# 路径
# -----------------------------


def runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).parent


def resource_path(rel: str) -> Path:
    meipass = Path(getattr(sys, "_MEIPASS", ""))
    if meipass:
        p = meipass / rel
        if p.exists():
            return p
    exe_dir = runtime_base_dir()
    p2 = exe_dir / rel
    if p2.exists():
        return p2
    return Path(__file__).parent / rel


BASE_DIR = runtime_base_dir()
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
CONFIG_DIR = BASE_DIR / "config"

OUTPUTS_DIR_DEFAULT = Path.home() / "Documents" / "AlloyDesignOutputs"
OUTPUTS_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)

UTS_MODEL_PATH = MODELS_DIR / "XGB_best_model.joblib"
EL_MODEL_PATH = MODELS_DIR / "EL_XGB_best_model.joblib"

FEATURE_ORDER_UTS_PATH = MODELS_DIR / "feature_order.json"
FEATURE_BOUNDS_UTS_PATH = MODELS_DIR / "feature_bounds.json"
FEATURE_ORDER_EL_PATH = MODELS_DIR / "feature_order_el.json"

HOME_BG_PATH = resource_path("assets/home_bg.png")
HELP_TEXTS_PATH = resource_path("config/help_texts.json")
DEFAULT_PRICE_XLSX_PATH = resource_path("config/default_price_table.xlsx")
USER_PRICE_JSON_PATH = CONFIG_DIR / "price_table_user.json"

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
RENAME_MAP = {
    "SS Temp": "固溶温度（℃）",
    "Ageing Temp": "时效温度（℃）",
    "Ageing Time": "时效时间（h）",
    "UTS_pred": "预测UTS（MPa）",
    "EL_pred": "预测EL（%）",
    "density_g_cm3": "密度（g/cm³）",
    "cost_per_kg": "成本（元/kg）",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@st.cache_data
def load_help_texts(path: Path):
    data = DEFAULT_HELP_TEXTS.copy()
    if path.exists():
        try:
            user_data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(user_data, dict):
                data.update(user_data)
        except Exception:
            pass
    return data


@st.cache_data
def load_default_price_table_from_xlsx(path: Path) -> dict:
    data = DEFAULT_PRICE_FALLBACK.copy()
    if not path.exists():
        return data

    try:
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row:
                continue
            elem = row[0]
            price = row[1] if len(row) > 1 else None
            if elem is None:
                continue
            elem = str(elem).strip()
            if elem in ELEMENT_COLS and price is not None:
                data[elem] = float(price)
    except Exception:
        pass
    return data


def render_help_icon(text: str):
    text = html.escape(text or "")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:center;height:38px;">
            <span title="{text}" style="
                display:inline-flex;
                align-items:center;
                justify-content:center;
                width:20px;height:20px;
                border-radius:50%;
                background:#e6f0fa;
                color:#1d5fa7;
                font-size:12px;
                font-weight:700;
                cursor:help;
                user-select:none;
            ">i</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def button_with_help(
    label: str,
    key: str,
    help_text: str,
    *,
    type: str = "secondary",
    use_container_width: bool = True,
    disabled: bool = False,
) -> bool:
    bcol, icol = st.columns([12, 1])
    with bcol:
        clicked = st.button(
            label,
            key=key,
            type=type,
            use_container_width=use_container_width,
            disabled=disabled,
        )
    with icol:
        render_help_icon(help_text)
    return clicked


def parse_required_float(text: str, field_name: str) -> float:
    text = str(text).strip()
    if text == "":
        raise ValueError(f"{field_name}不能为空。")
    return float(text)


def parse_optional_float(text: str):
    text = str(text).strip()
    if text == "":
        return None
    return float(text)


def _init_state():
    ss = st.session_state
    ss.setdefault("page", "home")

    # 首页/设计页状态
    ss.setdefault("show_save_panel", False)
    ss.setdefault("pending_search", False)
    ss.setdefault("is_searching", False)

    # 目标 UTS 输入：允许上限留空
    ss.setdefault("uts_low", 500.0)
    ss.setdefault("uts_high", 550.0)
    ss.setdefault("uts_low_text", "500")
    ss.setdefault("uts_high_text", "550")
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

    # 多级排序
    ss.setdefault("priority_order", ["强度", "延伸率", "密度", "价格"])

    # 上下限覆盖
    ss.setdefault("bounds_override", {})

    # 价格表
    if "default_price_table" not in ss:
        ss["default_price_table"] = load_default_price_table_from_xlsx(DEFAULT_PRICE_XLSX_PATH)

    if "price_table" not in ss:
        ss["price_table"] = ss["default_price_table"].copy()
        if USER_PRICE_JSON_PATH.exists():
            try:
                data = _load_json(USER_PRICE_JSON_PATH)
                for e in ELEMENT_COLS:
                    if e in data:
                        ss["price_table"][e] = float(data[e])
            except Exception:
                pass

    ss.setdefault("show_price_editor", False)


def _choose_directory_windows() -> Optional[str]:
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
    try:
        p = Path(path_str).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(p))  # type: ignore[attr-defined]
            return True
        except Exception:
            pass
        subprocess.Popen(["explorer", str(p)])
        return True
    except Exception:
        return False


def _build_bounds_use(bounds_default: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
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
        if nmin is None:
            nmin = dmin
        if nmax is None:
            nmax = dmax
        if nmin > nmax:
            continue

        out[feat]["min"] = nmin
        out[feat]["max"] = nmax

    if "Al" in out:
        out["Al"]["max"] = 99.1
    if "Ageing Time" in out:
        out["Ageing Time"]["max"] = min(float(out["Ageing Time"]["max"]), 48.0)

    return out


def _round_rebalance_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
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


def _export_excel_bytes(df: pd.DataFrame, cols: List[str]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df[cols].to_excel(writer, index=False, sheet_name="candidates")
    return buf.getvalue()


def sort_candidates_by_priority(df: pd.DataFrame, priority_order: List[str]) -> pd.DataFrame:
    by = []
    ascending = []
    for item in priority_order:
        col, asc = PRIORITY_META[item]
        if col in df.columns:
            by.append(col)
            ascending.append(asc)

    if "Ageing Time" in df.columns:
        by.append("Ageing Time")
        ascending.append(True)

    if not by:
        return df.reset_index(drop=True)

    return df.sort_values(by=by, ascending=ascending, na_position="last").reset_index(drop=True)


def render_search_overlay():
    st.markdown(
        """
        <style>
        .search-overlay {
            position: fixed;
            inset: 0;
            background: rgba(255,255,255,0.72);
            z-index: 999999;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            backdrop-filter: blur(2px);
        }
        .search-spinner {
            width: 56px;
            height: 56px;
            border: 6px solid #d8e6f3;
            border-top: 6px solid #2d77c7;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 18px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>

        <div class="search-overlay">
            <div class="search-spinner"></div>
            <div style="font-size:22px;font-weight:700;color:#1f4c7a;">正在搜索，请勿操作界面…</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_home_background(img_path: Path):
    if not img_path.exists():
        return

    ext = img_path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    b64 = base64.b64encode(img_path.read_bytes()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(255,255,255,0.22), rgba(255,255,255,0.22)),
                url("data:{mime};base64,{b64}") center center / cover no-repeat fixed;
        }}


        .home-title {{
            text-align:center;
            font-size: 38px;
            font-weight: 800;
            margin-top: 30px;
            margin-bottom: 20px;
            color: #103b61;
        }}

        .home-subtitle {{
            text-align:center;
            font-size: 16px;
            margin-bottom: 36px;
            color: #27506f;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def top_action_button(col, label: str, key: str, help_text: str, *, type: str = "secondary", use_container_width: bool = True, disabled: bool = False) -> bool:
    with col:
        bcol, icol = st.columns([12, 1])
        with bcol:
            clicked = st.button(label, key=key, type=type, use_container_width=use_container_width, disabled=disabled)
        with icol:
            render_help_icon(help_text)
    return clicked


def rename_result_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})


def render_home_page(help_texts: dict):
    inject_home_background(HOME_BG_PATH)

    st.markdown('<div class="home-title">时效强化型铝合金智能设计软件</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-subtitle">请选择进入设计模块或预测模块</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("### 设计模块")
        st.write("用于目标性能约束下的候选合金搜索、排序、展示与导出。")
        if button_with_help("进入设计模块", "go_design", help_texts.get("go_design", ""), type="primary"):
            st.session_state.page = "design"
            st.rerun()

    with c2:
        st.markdown("### 预测模块")
        st.write("用于输入指定成分和热处理参数后进行性能正向预测。")
        if button_with_help("进入预测模块", "go_pred", help_texts.get("go_pred", ""), type="primary"):
            st.session_state.page = "pred"
            st.rerun()


def render_save_panel(help_texts: dict):
    ss = st.session_state
    st.markdown("#### 结果保存目录")

    # 先处理目录选择按钮，再创建 text_input。
    # 这样不会在 text_input 已实例化后修改同名 key。
    c1, c2 = st.columns(2)
    with c1:
        if button_with_help("选择目录", "design_choose_dir", help_texts.get("design_choose_dir", "")):
            chosen = _choose_directory_windows()
            if chosen:
                ss.save_dir = chosen
                ss.design_save_dir = chosen
                st.rerun()
            else:
                st.warning("未能打开系统目录选择窗口。请在输入框中手动填写路径。")

    with c2:
        if button_with_help("打开输出目录", "design_open_dir", help_texts.get("design_open_dir", "")):
            current_dir = ss.get("design_save_dir", ss.save_dir)
            ok = _open_directory_windows(current_dir)
            if not ok:
                st.warning("未能打开输出目录。你可以复制路径到资源管理器地址栏打开。")

    # 目录选择处理完以后，再渲染输入框
    if "design_save_dir" not in ss:
        ss.design_save_dir = str(ss.save_dir)

    ss.save_dir = st.text_input("结果保存目录", key="design_save_dir")

def render_advanced_page(feature_names_uts, process_cols, help_texts: dict):
    ss = st.session_state

    st.title("高级约束")

    if button_with_help("返回设计模块", "btn_back_design_from_adv", help_texts.get("adv_back_design", ""), type="primary"):
        ss.page = "design"
        st.rerun()

    st.divider()

    tabs = st.tabs(["延伸率", "密度", "价格", "元素上下限", "工艺上下限", "候选合金数量", "优先级排序"])

    with tabs[0]:
        use_el = st.checkbox("计算并展示延伸率", value=bool(ss.use_el), key="w_use_el")
        ss.use_el = bool(use_el)
        if ss.use_el:
            constrain_el = st.checkbox("添加约束：延伸率 ≥ 目标值", value=bool(ss.constrain_el), key="w_constrain_el")
            ss.constrain_el = bool(constrain_el)
            if ss.constrain_el:
                target_el = st.number_input("目标 延伸率（%）", min_value=0.0, value=float(ss.target_el), step=0.5, key="w_target_el")
                ss.target_el = float(target_el)

    with tabs[1]:
        use_density = st.checkbox("计算并展示密度", value=bool(ss.use_density), key="w_use_density")
        ss.use_density = bool(use_density)
        if ss.use_density:
            constrain_density = st.checkbox("添加约束：密度 ≤ 上限", value=bool(ss.constrain_density), key="w_constrain_density")
            ss.constrain_density = bool(constrain_density)
            if ss.constrain_density:
                max_density = st.number_input("密度上限（g/cm³）", min_value=0.0, value=float(ss.max_density), step=0.01, key="w_max_density")
                ss.max_density = float(max_density)

    with tabs[2]:
        use_cost = st.checkbox("计算并展示价格（元/kg）", value=bool(ss.use_cost), key="w_use_cost")
        ss.use_cost = bool(use_cost)
        if ss.use_cost:
            constrain_cost = st.checkbox("添加约束：价格 ≤ 上限", value=bool(ss.constrain_cost), key="w_constrain_cost")
            ss.constrain_cost = bool(constrain_cost)
            if ss.constrain_cost:
                max_cost = st.number_input("价格上限（元/kg）", min_value=0.0, value=float(ss.max_cost), step=1.0, key="w_max_cost")
                ss.max_cost = float(max_cost)

        st.divider()

        if button_with_help("设置元素单价（元/kg）", "btn_open_price_editor", help_texts.get("adv_open_price_editor", "")):
            ss.show_price_editor = True

        if ss.show_price_editor:
            st.info("单价用于计算合金价格。系统默认值为内置真实价格，用户可自行修改并保存。")
            price_df = pd.DataFrame([{"Element": e, "Price": float(ss.price_table.get(e, 0.0))} for e in ELEMENT_COLS])
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
            with c1:
                if button_with_help("保存", "btn_save_price", help_texts.get("adv_save_price", "")):
                    ss.price_table = {r["Element"]: float(r["Price"]) for _, r in edited.iterrows()}
                    _save_json(USER_PRICE_JSON_PATH, ss.price_table)
                    st.success("已保存当前元素单价设置。")
            with c2:
                if button_with_help("重置为默认值", "btn_reset_price_default", help_texts.get("adv_reset_price_default", "")):
                    ss.price_table = ss.default_price_table.copy()
                    _save_json(USER_PRICE_JSON_PATH, ss.price_table)
                    st.success("已重置为系统默认价格。")
                    st.rerun()
            with c3:
                if button_with_help("关闭单价设置", "btn_close_price_editor", help_texts.get("adv_close_price_editor", "")):
                    ss.show_price_editor = False
                    st.rerun()

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
        with c1:
            if button_with_help("保存元素覆盖", "btn_save_elem_override", help_texts.get("adv_save_elem_override", "")):
                for _, r in edited.iterrows():
                    feat = str(r["元素"])
                    omin = None if pd.isna(r["最小值"]) else float(r["最小值"])
                    omax = None if pd.isna(r["最大值"]) else float(r["最大值"])
                    if omin is None and omax is None:
                        ss.bounds_override.pop(feat, None)
                    else:
                        ss.bounds_override[feat] = {"min": omin, "max": omax}
                st.success("已保存。")
        with c2:
            if button_with_help("清空元素覆盖", "btn_clear_elem_override", help_texts.get("adv_clear_elem_override", "")):
                for f in ELEMENT_COLS:
                    ss.bounds_override.pop(f, None)
                st.success("已清空。")

    with tabs[4]:
        st.caption("留空表示使用默认范围。")
        proc_rows = [{"工艺": PROC_NAME_MAP.get(f, f), "__key": f, "最小值": None, "最大值": None} for f in process_cols]
        df = pd.DataFrame(proc_rows).set_index("__key")

        for feat in df.index:
            ov = ss.bounds_override.get(str(feat), {})
            df.at[feat, "最小值"] = ov.get("min", None)
            df.at[feat, "最大值"] = ov.get("max", None)

        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "工艺": st.column_config.TextColumn("工艺", disabled=True),
                "最小值": st.column_config.NumberColumn("最小值"),
                "最大值": st.column_config.NumberColumn("最大值"),
            },
            key="proc_bounds_editor",
        )

        c1, c2 = st.columns(2)
        with c1:
            if button_with_help("保存工艺覆盖", "btn_save_proc_override", help_texts.get("adv_save_proc_override", "")):
                for feat, r in edited.iterrows():
                    feat = str(feat)
                    omin = None if pd.isna(r["最小值"]) else float(r["最小值"])
                    omax = None if pd.isna(r["最大值"]) else float(r["最大值"])
                    if omin is None and omax is None:
                        ss.bounds_override.pop(feat, None)
                    else:
                        ss.bounds_override[feat] = {"min": omin, "max": omax}
                st.success("已保存。")
        with c2:
            if button_with_help("清空工艺覆盖", "btn_clear_proc_override", help_texts.get("adv_clear_proc_override", "")):
                for f in process_cols:
                    ss.bounds_override.pop(f, None)
                st.success("已清空。")

    with tabs[5]:
        st.caption("设置候选保留数量。")
        archive_max_rows = st.number_input("候选保留数量", min_value=1, value=int(ss.archive_max_rows), step=200, key="w_archive_max_rows")
        ss.archive_max_rows = int(archive_max_rows)

    with tabs[6]:
        st.caption("设置候选方案的一级、二级、三级和四级排序优先级。若排序中包含延伸率、密度或价格，系统将自动计算对应指标。")
        st.info(help_texts.get("adv_sort_priority", ""))

        p1 = st.selectbox("一级优先级", PRIORITY_OPTIONS, index=PRIORITY_OPTIONS.index(ss.priority_order[0]), key="priority_1")
        p2 = st.selectbox("二级优先级", PRIORITY_OPTIONS, index=PRIORITY_OPTIONS.index(ss.priority_order[1]), key="priority_2")
        p3 = st.selectbox("三级优先级", PRIORITY_OPTIONS, index=PRIORITY_OPTIONS.index(ss.priority_order[2]), key="priority_3")
        p4 = st.selectbox("四级优先级", PRIORITY_OPTIONS, index=PRIORITY_OPTIONS.index(ss.priority_order[3]), key="priority_4")

        selected = [p1, p2, p3, p4]
        if len(set(selected)) < 4:
            st.error("四级优先级不能重复，请重新选择。")
        else:
            ss.priority_order = selected
            st.success("当前排序顺序：" + " → ".join(ss.priority_order))


def render_predict_page(feature_names_uts, process_cols, uts_model, el_model, feature_names_el, bounds_default, help_texts: dict):
    ss = st.session_state
    st.title("预测模块")

    if button_with_help("返回首页", "btn_pred_back_home", help_texts.get("pred_back_home", ""), type="primary"):
        ss.page = "home"
        st.rerun()

    if el_model is None or feature_names_el is None:
        st.error("未检测到 EL 模型或 feature_order_el.json，预测模块需要 EL 才能输出延伸率。")
        st.stop()

    st.divider()

    with st.form("predict_form", clear_on_submit=False):
        st.subheader("输入：成分（wt.%）与工艺参数")
        st.caption("建议只填写非Al元素；Al 将自动按 100-其余元素之和 计算。")

        elem_cols = [e for e in ELEMENT_COLS if e in feature_names_uts]
        non_al = [e for e in elem_cols if e != "Al"]

        cols = st.columns(4)
        elem_vals = {}
        for i, e in enumerate(non_al):
            with cols[i % 4]:
                elem_vals[e] = st.number_input(e, min_value=0.0, step=0.1, key=f"pred_elem_{e}")

        other_sum = float(sum(elem_vals.values()))
        al_val = 100.0 - other_sum
        st.write(f"自动计算：Al = {al_val:.2f} wt.%")

        proc_vals = {}
        st.subheader("工艺参数")
        for p in process_cols:
            if p in bounds_default:
                p0 = 0.5 * (float(bounds_default[p]["min"]) + float(bounds_default[p]["max"]))
            else:
                p0 = 0.0
            step = 1.0 if "Temp" in p else 0.1
            proc_vals[p] = st.number_input(PROC_LABEL_MAP.get(p, p), value=float(p0), step=float(step), key=f"pred_proc_{p}")

        sb1, sb2 = st.columns([12, 1])
        with sb1:
            submitted = st.form_submit_button("开始预测", type="primary", use_container_width=True)
        with sb2:
            render_help_icon(help_texts.get("pred_start", ""))

    if not submitted:
        return

    if al_val < 0:
        st.error("成分之和超过 100%，请减少某些元素含量。")
        return

    if al_val > 99.1 + 1e-9:
        st.error(f"计算得到的 Al={al_val:.2f} 超过 99.1，请提高其它元素总量或调整范围。")
        return

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

    result_row = X_uts.copy()
    result_row["UTS_pred"] = uts_pred
    result_row = add_el_pred(
        df_candidates=result_row,
        el_model=el_model,
        feature_names_el=feature_names_el,
        uts_pred_col="UTS_pred",
        el_uts_feature_name="UTS",
        out_col="EL_pred",
    )

    try:
        result_row = calc_density_g_cm3(result_row, out_col="density_g_cm3")
        result_row = calc_cost_per_kg(result_row, price_table=ss.price_table, out_col="cost_per_kg")
    except Exception:
        pass

    st.divider()
    st.subheader("预测结果")

    uts_value = float(result_row.loc[result_row.index[0], "UTS_pred"]) if "UTS_pred" in result_row.columns else None
    el_value = float(result_row.loc[result_row.index[0], "EL_pred"]) if "EL_pred" in result_row.columns else None
    density_value = float(result_row.loc[result_row.index[0], "density_g_cm3"]) if "density_g_cm3" in result_row.columns else None
    cost_value = float(result_row.loc[result_row.index[0], "cost_per_kg"]) if "cost_per_kg" in result_row.columns else None

    mcols = st.columns(4)
    mcols[0].metric("预测UTS（MPa）", f"{uts_value:.2f}" if uts_value is not None else "—")
    mcols[1].metric("预测EL（%）", f"{el_value:.2f}" if el_value is not None else "—")
    mcols[2].metric("密度（g/cm³）", f"{density_value:.4f}" if density_value is not None else "—")
    mcols[3].metric("成本（元/kg）", f"{cost_value:.2f}" if cost_value is not None else "—")

    table_exclude_cols = {"UTS_pred", "EL_pred", "density_g_cm3", "cost_per_kg"}
    show_cols = [c for c in (ELEMENT_COLS + ["SS Temp", "Ageing Temp", "Ageing Time"]) if c in result_row.columns and c not in table_exclude_cols]
    result_show = rename_result_table(result_row[show_cols].copy())
    result_show.index = [1]

    st.dataframe(result_show, use_container_width=True)


def run_design_search(uts_model, feature_names_uts, bounds_default, el_model, feature_names_el) -> Tuple[pd.DataFrame, pd.DataFrame, dict, List[str], bytes, Path, bool]:
    ss = st.session_state

    try:
        low = parse_required_float(ss.uts_low_text, "UTS下限")
        high = parse_optional_float(ss.uts_high_text)
    except ValueError as e:
        raise ValueError(str(e)) from e

    if high is not None and low > high:
        low, high = high, low

    ss.uts_low = low
    ss.uts_high = high if high is not None else low
    ss.search_mode = "range_max" if high is not None else "max"

    bounds_use = _build_bounds_use(bounds_default, ss.bounds_override)

    progress_slot = st.empty()
    progress_bar = progress_slot.progress(0.0)

    def on_prog(gen: int, best: float, mean: float):
        if gen == 1 or gen % 5 == 0:
            pct = min(gen / float(MAX_GENERATIONS), 0.99)
            progress_bar.progress(pct)

    df_candidates = run_ga_elite_archive(
        ga_script_path=GA_SCRIPT_PATH,
        model=uts_model,
        feature_order=feature_names_uts,
        bounds=bounds_use,
        objective=str(ss.get("search_mode", "range_max")),
        uts_low=(float(ss.get("uts_low")) if ss.get("search_mode", "range_max") != "max" else None),
        uts_high=(float(ss.get("uts_high")) if ss.get("search_mode", "range_max") != "max" else None),
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
        archive_max_rows=int(ss.archive_max_rows),
        prefer_short_ageing=PREFER_SHORT_AGEING,
        prefer_eps=PREFER_SHORT_AGEING_EPS,
        on_progress=on_prog,
    )

    df = df_candidates.copy()

    need_el = bool(ss.use_el or ss.constrain_el or ("延伸率" in ss.priority_order))
    need_density = bool(ss.use_density or ss.constrain_density or ("密度" in ss.priority_order))
    need_cost = bool(ss.use_cost or ss.constrain_cost or ("价格" in ss.priority_order))

    if need_el:
        if el_model is None or feature_names_el is None:
            raise RuntimeError("排序或约束需要延伸率，但未检测到 EL 模型或 feature_order_el.json。")
        df = add_el_pred(
            df_candidates=df,
            el_model=el_model,
            feature_names_el=feature_names_el,
            uts_pred_col="UTS_pred",
            el_uts_feature_name="UTS",
            out_col="EL_pred",
        )

    if need_density:
        df = calc_density_g_cm3(df, out_col="density_g_cm3")
    if need_cost:
        df = calc_cost_per_kg(df, price_table=ss.price_table, out_col="cost_per_kg")

    opts = ConstraintOptions(
        use_el=bool(ss.constrain_el),
        target_el=float(ss.target_el) if ss.constrain_el else None,
        use_density=bool(ss.constrain_density),
        max_density=float(ss.max_density) if ss.constrain_density else None,
        use_cost=bool(ss.constrain_cost),
        max_cost=float(ss.max_cost) if ss.constrain_cost else None,
    )
    df_f = apply_constraints(df, opts)
    df_out = _round_rebalance_and_dedup(df_f)

    if len(df_out) == 0:
        raise RuntimeError("当前约束下没有任何候选方案。请放宽约束或调整上下限。")

    low = float(ss.get("uts_low", 0.0))
    high = float(ss.get("uts_high", low))
    if low > high:
        low, high = high, low

    mode = str(ss.get("search_mode", "range_max"))
    keep_n = int(ss.archive_max_rows)

    if mode == "max":
        df_show = df_out.copy()
        export_df = df_show.head(keep_n).reset_index(drop=True)
        export_name = "alloy_candidates_global_max.xlsx"
    else:
        df_in = df_out[(df_out["UTS_pred"] >= low) & (df_out["UTS_pred"] <= high)].copy()
        if len(df_in) == 0:
            df_tmp = df_out.copy()
            df_tmp["_dist"] = 0.0
            df_tmp.loc[df_tmp["UTS_pred"] < low, "_dist"] = (low - df_tmp["UTS_pred"])
            df_tmp.loc[df_tmp["UTS_pred"] > high, "_dist"] = (df_tmp["UTS_pred"] - high)
            df_show = df_tmp.sort_values(["_dist", "Ageing Time", "UTS_pred"], ascending=[True, True, False]).drop(columns=["_dist"]).reset_index(drop=True)
            export_df = df_show.head(keep_n).reset_index(drop=True)
            export_name = "alloy_candidates_nearest_range.xlsx"
        else:
            df_show = df_in.reset_index(drop=True)
            export_df = df_show.head(keep_n).reset_index(drop=True)
            export_name = "alloy_candidates_in_range_max.xlsx"

    df_show = sort_candidates_by_priority(df_show, ss.priority_order)
    export_df = sort_candidates_by_priority(export_df, ss.priority_order)

    best = export_df.iloc[0].to_dict()

    save_dir = Path(ss.save_dir).expanduser()
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
    return export_df, df_show, best, show_cols, excel_bytes, saved_path, saved_ok


def render_design_page(feature_names_uts, process_cols, uts_model, el_model, feature_names_el, bounds_default, help_texts: dict):
    ss = st.session_state
    st.title("设计模块")

    top1, top2, top3, top4 = st.columns([1.2, 1, 1, 5])
    if top_action_button(top1, "返回首页", "btn_design_back_home", help_texts.get("design_back_home", ""), type="primary"):
        ss.page = "home"
        st.rerun()
    if top_action_button(top2, "高级约束", "btn_adv_small", help_texts.get("design_adv", "")):
        ss.page = "adv"
        st.rerun()
    if top_action_button(top3, "保存目录", "btn_save_small", help_texts.get("design_save_toggle", "")):
        ss.show_save_panel = not ss.show_save_panel
        st.rerun()

    st.divider()

    if ss.show_save_panel:
        render_save_panel(help_texts)
        st.divider()

    st.subheader("目标性能输入")
    c1, c2 = st.columns(2)
    with c1:
        ss.uts_low_text = st.text_input("UTS 下限（MPa）", value=ss.uts_low_text, key="uts_low_text_input")
    with c2:
        ss.uts_high_text = st.text_input("UTS 上限（MPa）", value=ss.uts_high_text, key="uts_high_text_input")

    st.divider()

    if button_with_help("开始搜索", "btn_design_start_search", help_texts.get("design_start_search", ""), type="primary", disabled=ss.is_searching):
        ss.pending_search = True
        st.rerun()

    if ss.pending_search:
        ss.is_searching = True
        overlay_slot = st.empty()
        with overlay_slot:
            render_search_overlay()

        try:
            export_df, df_show, best, show_cols, excel_bytes, saved_path, saved_ok = run_design_search(
                uts_model=uts_model,
                feature_names_uts=feature_names_uts,
                bounds_default=bounds_default,
                el_model=el_model,
                feature_names_el=feature_names_el,
            )
            overlay_slot.empty()

            st.divider()
            st.subheader("第1名综合结果")
            mcols = st.columns(4)
            mcols[0].metric("UTS (MPa)", f'{best["UTS_pred"]:.2f}')
            if "EL_pred" in best:
                mcols[1].metric("延伸率 (%)", f'{best["EL_pred"]:.2f}')
            if "density_g_cm3" in best:
                mcols[2].metric("密度 (g/cm³)", f'{best["density_g_cm3"]:.4f}')
            if "cost_per_kg" in best:
                mcols[3].metric("价格 (元/kg)", f'{best["cost_per_kg"]:.2f}')

            top5_df = export_df.head(5).copy()
            top5_show = rename_result_table(top5_df[show_cols].copy())
            top5_show.index = range(1, len(top5_show) + 1)

            st.subheader("最佳五种组合")
            st.dataframe(top5_show, use_container_width=True)

            st.info(f"候选数：{len(df_show)}。当前展示前 5 种最优组合。")
            if saved_ok:
                st.success(f"Excel 已保存到：{saved_path}")
            else:
                st.warning("未能写入到你指定的目录（可能权限或路径问题）。你仍可以使用下面的下载按钮获取 Excel。")

            d1, d2 = st.columns([12, 1])
            with d1:
                st.download_button(
                    label="下载候选方案 Excel",
                    data=excel_bytes,
                    file_name=saved_path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            with d2:
                render_help_icon(help_texts.get("design_download", ""))
        except ValueError as e:
            st.error(str(e))
        except RuntimeError as e:
            st.error(str(e))
        finally:
            try:
                overlay_slot.empty()
            except Exception:
                pass
            ss.is_searching = False
            ss.pending_search = False


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
HELP_TEXTS = load_help_texts(HELP_TEXTS_PATH)

missing = []
for pth, msg in [
    (UTS_MODEL_PATH, "缺少 UTS 模型：models/XGB_best_model.joblib"),
    (FEATURE_ORDER_UTS_PATH, "缺少 UTS 特征顺序：models/feature_order.json"),
    (FEATURE_BOUNDS_UTS_PATH, "缺少默认上下限：models/feature_bounds.json"),
    (GA_SCRIPT_PATH, "缺少 GA 脚本：GA-UTS-XGB.py"),
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

el_model = None
feature_names_el = None
if EL_MODEL_PATH.exists() and FEATURE_ORDER_EL_PATH.exists():
    el_model = _load_model(EL_MODEL_PATH)
    feature_names_el = _load_json(FEATURE_ORDER_EL_PATH)

if st.session_state.page == "home":
    render_home_page(HELP_TEXTS)
    st.stop()

if st.session_state.page == "adv":
    render_advanced_page(feature_names_uts, process_cols, HELP_TEXTS)
    st.stop()

if st.session_state.page == "pred":
    render_predict_page(feature_names_uts, process_cols, uts_model, el_model, feature_names_el, bounds_default, HELP_TEXTS)
    st.stop()

if st.session_state.page == "design":
    render_design_page(feature_names_uts, process_cols, uts_model, el_model, feature_names_el, bounds_default, HELP_TEXTS)
    st.stop()
