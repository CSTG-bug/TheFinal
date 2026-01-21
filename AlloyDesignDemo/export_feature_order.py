from pathlib import Path
import json
import pandas as pd

# ====== 需要改这两项 ======
DATA_PATH = r"D:\MLDesignAl\TheFinal\Data\ElementTreatmentEl-UTS\output-exceptEL\exceptEL-X_train_raw.csv"
TARGET_COL = ""
# =================================

# 如果数据里有一些列不属于特征（比如牌号/文献来源/ID等），写在这里删掉
DROP_COLS = []

OUT_DIR = Path(__file__).parent / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FEATURE_ORDER = OUT_DIR / "feature_order.json"
OUT_BOUNDS = OUT_DIR / "feature_bounds.json"


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

    # 特征列：两种情况
    # 1) 若 TARGET_COL 为空或不存在：全部列都当作特征
    # 2) 若 TARGET_COL 存在：除目标列外都当作特征
    if (not TARGET_COL) or (TARGET_COL not in df.columns):
        feature_cols = list(df.columns)
    else:
        feature_cols = [c for c in df.columns if c != TARGET_COL]

    # 特征列：除目标列外的所有列（保持原顺序！）
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    print("========== 导出信息 ==========")
    print("数据文件：", DATA_PATH)
    print("目标列：", TARGET_COL)
    print("特征数：", len(feature_cols))
    print("前10个特征：", feature_cols[:10])
    print("后10个特征：", feature_cols[-10:])

    # 强校验
    if len(feature_cols) != 21:
        raise ValueError(
            f"特征数不是 21（当前 {len(feature_cols)}）。\n"
            f"说明：你的数据表里可能多了无关列/少了特征列，或 TARGET_COL 填错。\n"
            f"当前特征列：{feature_cols}"
        )

    # 保存特征顺序
    OUT_FEATURE_ORDER.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入：", OUT_FEATURE_ORDER)

    # 同时把训练集范围也导出
    bounds = {}
    for c in feature_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        bounds[c] = {"min": float(col.min()), "max": float(col.max())}
    OUT_BOUNDS.write_text(json.dumps(bounds, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入：", OUT_BOUNDS)

    print("完成。")


if __name__ == "__main__":
    main()
