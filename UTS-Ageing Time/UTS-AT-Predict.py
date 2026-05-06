import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ===================== 1) 配置区 =====================
MODEL_PATH = r"D:\MLDesignAl\TheFinal\XGBoost\ElementTreatmentEl-UTS\output-exceptEL\XGB_best_model.joblib"

# 读取特征列
TRAIN_X_PATH = r"D:\MLDesignAl\TheFinal\Data\ElementTreatmentEl-UTS\output-exceptEL\exceptEL-X_train_raw.csv"

# 如果 TRAIN_X_PATH 里包含目标列（例如UTS），就在这里写上列名以便剔除；否则留 None
TARGET_COL = None
DROP_COLS = []             # 如有不参与训练的列（ID/编号等），写在这里，如 ["AlloyID"]

# 保存位置
SAVE_TARGET = r"D:\MLDesignAl\TheFinal\UTS-Ageing Time"

# 固定输入参数
fixed_inputs = {
    "Si": 0.00,
    "Fe": 0.00,
    "Cu": 2.36,
    "Mn": 0.00,
    "Mg": 2.46,
    "Cr": 0.00,
    "Zn": 6.28,
    "V" : 0.00,
    "Ti": 0.058,
    "Zr": 0.14,
    "Li": 0.00,
    "Ni": 0.00,
    "Be": 0.00,
    "Sc": 0.00,
    "Ag": 0.00,
    "Bi": 0.00,
    "Pb": 0.00,
    "Al": 88.702,
    "SS Temp"    : 465,
    "Ageing Temp": 120,
}
AGING_TIME_COL = "Ageing Time"

# 时效时间网格
t_grid = np.linspace(0, 48, 481)


# ===================== 2) 工具函数：解析保存路径 =====================
def resolve_save_paths(target: str, base_name: str):
    p = Path(str(target).strip().strip('"').strip("'")).expanduser()
    is_dir_like = (p.exists() and p.is_dir()) or str(p).endswith(("/", "\\")) or (p.suffix == "")
    if is_dir_like:
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{base_name}.png", p / f"{base_name}.csv"

    p.parent.mkdir(parents=True, exist_ok=True)
    suf = p.suffix.lower()
    if suf == ".png":
        return p, p.with_suffix(".csv")
    if suf == ".csv":
        return p.with_suffix(".png"), p
    return p.with_suffix(".png"), p.with_suffix(".csv")


# ===================== 3) 标注边界处理：自动选点位并夹紧 =====================
def get_safe_annotate_xytext(x, y, x_min, x_max, y_min, y_max):
    """
    根据点 (x,y) 与坐标边界，自动选择文本放置方向（左/右、上/下），并防止越界。
    返回：xytext, ha, va
    """
    x_range = (x_max - x_min) if (x_max > x_min) else 1.0
    y_range = (y_max - y_min) if (y_max > y_min) else 1.0

    # 偏移量（随坐标范围自适应）
    dx = 0.06 * x_range
    dy = 0.06 * y_range

    # 右边太近则放左边；顶部太近则放下方
    if x > x_max - 0.18 * x_range:
        x_text = x - dx
        ha = "right"
    else:
        x_text = x + dx
        ha = "left"

    if y > y_max - 0.18 * y_range:
        y_text = y - dy
        va = "top"
    else:
        y_text = y + dy
        va = "bottom"

    # 夹紧，确保文本锚点仍在边界内（留一点边距）
    x_text = min(max(x_text, x_min + 0.02 * x_range), x_max - 0.02 * x_range)
    y_text = min(max(y_text, y_min + 0.02 * y_range), y_max - 0.02 * y_range)

    return (x_text, y_text), ha, va


# ===================== 4) 载入模型（joblib） =====================
model = joblib.load(MODEL_PATH)


# ===================== 5) 从原始训练数据读取特征列 =====================
# 只读取表头（nrows=0）
cols = pd.read_csv(TRAIN_X_PATH, nrows=0).columns.tolist()

# 去掉目标列与不需要的列
if TARGET_COL is not None and TARGET_COL in cols:
    cols.remove(TARGET_COL)
for c in DROP_COLS:
    if c in cols:
        cols.remove(c)

feature_cols = cols


# ===================== 6) 构造输入（固定 + 变化时效时间） =====================
rows = []
for t in t_grid:
    row = fixed_inputs.copy()
    row[AGING_TIME_COL] = float(t)
    rows.append(row)

X_curve = pd.DataFrame(rows)

# 补齐缺失列（缺失成分补0通常合理；若缺的是关键工艺列，说明你 fixed_inputs 写漏了）
missing = [c for c in feature_cols if c not in X_curve.columns]
for c in missing:
    X_curve[c] = 0.0

# 严格按训练特征列顺序排列
X_curve = X_curve[feature_cols]


# ===================== 7) 预测 =====================
y_pred = model.predict(X_curve)

# 如果你训练时对UTS做过变换（例如log），在这里做反变换：
# y_pred = np.exp(y_pred)


# ===================== 8) 峰值点 =====================
idx_peak = int(np.argmax(y_pred))
t_peak = float(t_grid[idx_peak])
uts_peak = float(y_pred[idx_peak])


# ===================== 9) 绘图 + 峰值标注（含边界处理） =====================
plt.figure(figsize=(8, 5))
plt.plot(t_grid, y_pred, linewidth=2)
plt.scatter([t_peak], [uts_peak], zorder=3)
plt.axvline(t_peak, linestyle="--", linewidth=1, alpha=0.6)

# 给y轴留一点上下边距，让标注更不容易挤出图
y_min, y_max = float(np.min(y_pred)), float(np.max(y_pred))
y_pad = 0.08 * (y_max - y_min) if (y_max > y_min) else 1.0
plot_y_min, plot_y_max = y_min - y_pad, y_max + y_pad
plt.ylim(plot_y_min, plot_y_max)

label_text = f"Peak: {uts_peak:.1f} MPa @ {t_peak:.2f} h"
(x_text, y_text), ha, va = get_safe_annotate_xytext(
    t_peak, uts_peak,
    x_min=float(np.min(t_grid)), x_max=float(np.max(t_grid)),
    y_min=plot_y_min, y_max=plot_y_max
)

plt.annotate(
    label_text,
    xy=(t_peak, uts_peak),
    xytext=(x_text, y_text),
    ha=ha, va=va,
    arrowprops=dict(arrowstyle="->", lw=1),
)

plt.xlabel("Aging time (h)")
plt.ylabel("Predicted UTS (MPa)")
plt.title("UTS-Ageing Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()


# ===================== 10) 保存 PNG + CSV =====================
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = f"UTS_vs_AgingTime_{stamp}"
png_path, csv_path = resolve_save_paths(SAVE_TARGET, base_name)

plt.savefig(png_path, dpi=300, bbox_inches="tight")

out_df = pd.DataFrame({
    "Aging_time_h": t_grid,
    "Predicted_UTS_MPa": y_pred
})
out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"[OK] 曲线图已保存：{png_path}")
print(f"[OK] 数据已保存：{csv_path}")
print(f"[INFO] 峰值点：UTS={uts_peak:.3f} MPa, t={t_peak:.3f} h")

plt.show()
