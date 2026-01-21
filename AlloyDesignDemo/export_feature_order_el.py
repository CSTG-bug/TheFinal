from pathlib import Path
import json
import pandas as pd

# ====== 换成训练“EL模型”用的 X 数据文件 ======
DATA_PATH = r"D:\MLDesignAl\TheFinal\Data\ElementTreatmentUTS-EL\output\ElementTreatmentUTS-EL-X_train_raw.csv"
# ============================================================================

DROP_COLS = []  # 如果你的文件里有ID/牌号等非特征列，填在这里删掉

OUT_DIR = Path(__file__).parent / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FEATURE_ORDER = OUT_DIR / "feature_order_el.json"
OUT_BOUNDS = OUT_DIR / "feature_bounds_el.json"


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到数据文件：{p}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    else:
        raise ValueError("只支持 .xlsx/.xls/.csv")


def main():
    df = read_table(DATA_PATH)

    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    feature_cols = list(df.columns)

    print("========== 导出信息（EL模型输入特征） ==========")
    print("数据文件：", DATA_PATH)
    print("特征数：", len(feature_cols))
    print("前10个特征：", feature_cols[:10])
    print("后10个特征：", feature_cols[-10:])

    # 强校验
    if len(feature_cols) != 22:
        raise ValueError(
            f"特征数不是 22（当前 {len(feature_cols)}）。\n"
            f"说明：你选的不是 EL 模型训练用的 X 文件，或包含了多余列。\n"
            f"当前列名：{feature_cols}"
        )

    # 自动提示哪个列可能是UTS（用于后续把UTS_pred填进去）
    uts_like = [c for c in feature_cols if "uts" in c.lower()]
    print("可能的UTS列名（包含'UTS'字样）：", uts_like if uts_like else "未发现（需要你人工确认是哪一列代表UTS）")

    OUT_FEATURE_ORDER.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入：", OUT_FEATURE_ORDER)

    bounds = {}
    for c in feature_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        bounds[c] = {"min": float(col.min()), "max": float(col.max())}
    OUT_BOUNDS.write_text(json.dumps(bounds, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入：", OUT_BOUNDS)

    print("完成。")


if __name__ == "__main__":
    main()
