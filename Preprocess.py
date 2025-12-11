#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
你可以【手动指定】输入特征列和目标列；不做自动列识别。默认也不做缺失值填补，若存在缺失会给出报错，
方便你在建库阶段自己清洗。

功能要点
--------
1) 仅使用你通过 --features 指定的列作为输入特征；--target 指定目标列。
2) 训练/测试划分（可分层，按目标值分箱；极端情况下自动回退成普通划分）。
3) 可选标准化（--scale），默认开启；关闭就去掉任何缩放。
4) 可选缺失值填补（--impute），默认 none（不填补，遇缺失直接报错）。
5) 产物：
   - 预处理后的训练/测试集（CSV），文件名/路径你可完全自定义；
   - 可选保存一个 .npz（Numpy 格式）便于后续直接加载（--save-npz）。
6) 集成 MAPE 函数（供后续评估脚本复用），本脚本不计算 MAPE/R2（建模阶段再算）。

注意
----
- Excel 默认读取第 1 个 sheet；如需指定 sheet，传 --sheet 0 或 --sheet "Sheet1"。
- 若你的特征名里有空格，请确保正确输入（如 "SS Temp"）。
- 如果你自己完全保证无缺失与无非数值，则建议使用 --impute none（默认）。
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

# -------------------------
# 指标函数（后续建模脚本可导入或复制）
# -------------------------
def mean_absolute_percentage_error(y_true, y_pred) -> float:
    """
    对 y_true 的 0 做极小值保护，避免除零。
    返回百分数（0-100）。
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

# -------------------------
# 工具函数
# -------------------------
def parse_feature_list(s: str) -> List[str]:
    """
    将逗号分隔的字符串解析为列名列表。允许列名包含空格（不会去掉中间空格）。
    会对每个片段做 strip() 去除首尾空白。
    """
    return [x.strip() for x in s.split(",") if x.strip()]

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 DataFrame 中的列尽量转为数值（errors='coerce'），非数值会变为 NaN。
    之所以这样做：有时读入后是字符串形式的数字（如 '0.25'），先转成数值，后续才好检查缺失或做缩放。
    """
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# -------------------------
# 主流程
# -------------------------
def preprocess(
    input_path: str,
    target_col: str,
    feature_cols: List[str],
    test_size: float = 0.3,
    random_state: int = SEED,
    sheet: int | str | None = None,
    do_scale: bool = True,
    impute: str = "none",   # "none" | "median" | "mean" | "zero"
    outdir: str | None = None,
    prefix: str = "",
    save_csv: bool = True,
    save_npz: bool = False,
    xtrain_file: str | None = None,
    xtest_file: str | None = None,
    ytrain_file: str | None = None,
    ytest_file: str | None = None,
    save_raw: bool = True,
    xtrain_raw_file: str | None = None,
    xtest_raw_file: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    返回：X_train, X_test, y_train, y_test
    同时根据参数写出到你指定的位置/文件名。
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 读取文件
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(p, sheet_name=0 if sheet is None else sheet)
    else:
        raise ValueError("仅支持 .csv / .xls(x)")

    # 仅保留用户明确指定的列（不做任何自动剔除/补齐）
    missing = [c for c in [target_col] + feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"以下列在数据中未找到：{missing}\n可用列为：{list(df.columns)}")

    y = df[target_col].copy()
    X = df[feature_cols].copy()

    # 目标处理：极小值保护（为 MAPE 做准备），其余不做额外清洗
    y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    valid_idx = y.index
    y = y.clip(lower=1e-6)

    # 对齐特征
    X = X.loc[valid_idx]

    # 将特征尽量转为数值
    X = coerce_numeric(X)

    # 缺失处理
    if impute == "none":
        # 若有缺失，直接报错，提醒你先在建库阶段清理
        if X.isna().any().any():
            na_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(f"检测到缺失值（impute=none），请先在数据层面清理。含缺失的列：{na_cols}")
    elif impute == "median":
        X = X.fillna(X.median())
    elif impute == "mean":
        X = X.fillna(X.mean())
    elif impute == "zero":
        X = X.fillna(0.0)
    else:
        raise ValueError("impute 参数必须是 none/median/mean/zero 之一")

    # 训练/测试划分（优先分层，失败则回退）
    stratify = None
    try:
        bins = pd.qcut(y, q=min(10, max(3, int(np.sqrt(len(y))))), duplicates="drop")
        stratify = bins
    except Exception:
        stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    X_train_raw = X_train.copy()
    X_test_raw  = X_test.copy()

    # 特征缩放
    if do_scale:
        scaler = StandardScaler()
        X_train_arr = scaler.fit_transform(X_train)
        X_test_arr = scaler.transform(X_test)
        X_train = pd.DataFrame(X_train_arr, index=X_train.index, columns=feature_cols)
        X_test = pd.DataFrame(X_test_arr, index=X_test.index, columns=feature_cols)

    # 输出路径和文件名
    if outdir is None:
        out_dir = p.parent
    else:
        out_dir = Path(outdir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # 默认文件名（可被显式参数覆盖）
    xtrain_path = out_dir / (xtrain_file if xtrain_file else f"{prefix}X_train.csv")
    xtest_path  = out_dir / (xtest_file  if xtest_file  else f"{prefix}X_test.csv")
    ytrain_path = out_dir / (ytrain_file if ytrain_file else f"{prefix}y_train.csv")
    ytest_path  = out_dir / (ytest_file  if ytest_file  else f"{prefix}y_test.csv")

    # 写出
    if save_csv:
        X_train.to_csv(xtrain_path, index=False)
        X_test.to_csv(xtest_path, index=False)
        # 显式写出目标列名，便于后续识别
        y_train.to_csv(ytrain_path, index=False, header=[target_col])
        y_test.to_csv(ytest_path, index=False, header=[target_col])
        if save_raw:
            xtrain_raw_path = out_dir / (xtrain_raw_file if xtrain_raw_file else f"{prefix}X_train_raw.csv")
            xtest_raw_path  = out_dir / (xtest_raw_file  if xtest_raw_file  else f"{prefix}X_test_raw.csv")
            X_train_raw.to_csv(xtrain_raw_path, index=False)
            X_test_raw.to_csv(xtest_raw_path, index=False)

    if save_npz:
        # 说明：npz 是 Numpy 的压缩打包格式，适合在 Python/NumPy/Sklearn 里快速加载；
        # 对你来说不是“必须”，只是便捷选项。
        np.savez(
            out_dir / f"{prefix}data.npz",
            X_train=X_train.values,
            X_test=X_test.values,
            y_train=y_train.values,
            y_test=y_test.values,
            feature_names=np.array(feature_cols, dtype=object),
            target_name=np.array([target_col], dtype=object),
            scaled=np.array([do_scale], dtype=bool),
            # ★ 可选：追加原始
            X_train_raw=X_train_raw.values,
            X_test_raw=X_test_raw.values,
        )

    # 控制台报告
    print("===== 预处理完成（手动可控版）=====")
    print(f"样本总数(有效): {len(X)} | 训练: {len(X_train)} | 测试: {len(X_test)}")
    print(f"目标列: {target_col}")
    print(f"特征列（{len(feature_cols)} 个）: {feature_cols}")
    print(f"训练集目标最小/最大: {float(y_train.min()):.3f} / {float(y_train.max()):.3f}")
    print(f"测试集目标最小/最大: {float(y_test.min()):.3f} / {float(y_test.max()):.3f}")
    print(f"已保存至: {out_dir.resolve()}")
    if save_npz:
        print(f"(已生成 {prefix}data.npz)")

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入数据路径（CSV或XLSX）")
    ap.add_argument("--sheet", default=None, help="Excel 的 sheet 索引或名称（可选）")
    ap.add_argument("--target", default="UTS", help="目标列名（默认 UTS）")
    ap.add_argument("--features", required=True, help="逗号分隔的特征列名列表，如：Si,Fe,Cu,Mn,Mg")
    ap.add_argument("--test-size", type=float, default=0.3, help="测试集比例（默认 0.3）")
    ap.add_argument("--random-state", type=int, default=SEED, help="随机种子（默认 42）")
    ap.add_argument("--save-raw", action="store_true", help="同时保存未标准化的 X_train_raw/X_test_raw")
    ap.add_argument("--xtrain-raw-file", default=None, help="原始 X_train 文件名（含扩展名）")
    ap.add_argument("--xtest-raw-file", default=None, help="原始 X_test  文件名（含扩展名）")

    # 缩放与填补
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--scale", dest="scale", action="store_true", help="启用标准化（默认启用）")
    grp.add_argument("--no-scale", dest="scale", action="store_false", help="禁用标准化")
    ap.set_defaults(scale=True)

    ap.add_argument("--impute", choices=["none","median","mean","zero"], default="none",
                    help="缺失值填补策略（默认 none 不填补，遇到缺失报错）")

    # 输出控制
    ap.add_argument("--outdir", default=None, help="输出目录（默认与输入同目录）")
    ap.add_argument("--prefix", default="", help="输出文件名前缀（如 exp1_）")
    ap.add_argument("--no-csv", dest="save_csv", action="store_false", help="不保存 CSV（默认保存）")
    ap.add_argument("--save-npz", action="store_true", help="额外保存一个 npz（可选）")

    # 显式文件名（可覆盖 prefix 规则）
    ap.add_argument("--xtrain-file", default=None, help="X_train 文件名（含扩展名）")
    ap.add_argument("--xtest-file",  default=None, help="X_test 文件名（含扩展名）")
    ap.add_argument("--ytrain-file", default=None, help="y_train 文件名（含扩展名）")
    ap.add_argument("--ytest-file",  default=None, help="y_test 文件名（含扩展名）")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    feature_cols = parse_feature_list(args.features)

    preprocess(
        input_path=args.input,
        target_col=args.target,
        feature_cols=feature_cols,
        test_size=args.test_size,
        random_state=args.random_state,
        sheet=args.sheet,
        do_scale=args.scale,
        impute=args.impute,
        outdir=args.outdir,
        prefix=args.prefix,
        save_csv=args.save_csv,
        save_npz=args.save_npz,
        xtrain_file=args.xtrain_file,
        xtest_file=args.xtest_file,
        ytrain_file=args.ytrain_file,
        ytest_file=args.ytest_file,
        save_raw=args.save_raw,
        xtrain_raw_file=args.xtrain_raw_file,
        xtest_raw_file=args.xtest_raw_file,
    )


if __name__ == "__main__":
    # 👇 这里你可以直接写入参数（写死参数模式）
    import sys
    sys.argv = [
        "Preprocess.py",
        "--input", r"D:\MLDesignAl\TheFinal\Data\ElementTreatment-UTS\ElementTreatment-UTS-Processed.csv",
        "--target", "UTS",
        "--features", "Si,Fe,Cu,Mn,Mg,Cr,Zn,V,Ti,Zr,Li,Ni,Be,Sc,Ag,Bi,Pb,Al,SS Temp,Ageing Temp,Ageing Time",
        "--test-size", "0.3",
        "--scale",
        "--save-raw",
        "--outdir", r"D:\MLDesignAl\TheFinal\Data\ElementTreatment-UTS\output-Processed",
        "--xtrain-raw-file", "ElementTreatment-UTS-Processed-X_train_raw.csv",
        "--xtest-raw-file",  "ElementTreatment-UTS-Processed-X_test_raw.csv",
        "--prefix", "ElementTreatment-UTS-Processed-",
        "--save-npz"
    ]
    main()

'''
| 参数                     | 功能                  | 示例                                                 |
| ---------------------- | ------------------- | -------------------------------------------------- |
| `--input`              | 数据文件路径（CSV 或 Excel） | `"C:/data/Origin-1293.xlsx"`                       |
| `--target`             | 目标列名（输出）            | `"UTS"` 或 `"EL"`                                   |
| `--features`           | 逗号分隔的输入特征列名         | `"Si,Fe,Cu,Mn,Mg,SS Temp,Ageing Temp,Ageing Time"` |
| `--test-size`          | 测试集比例               | `0.3`                                              |
| `--scale / --no-scale` | 是否标准化特征             | 默认标准化；不想就加 `--no-scale`                            |
| `--impute`             | 缺失值填补策略             | `none`（默认）、`median`、`mean`、`zero`                  |
| `--outdir`             | 输出文件目录              | `"C:/Users/Ricardo/Desktop/output"`                |
| `--prefix`             | 输出文件名前缀             | `"exp1_"`（结果文件名会变成 `exp1_X_train.csv`）             |
| `--save-npz`           | 是否生成 `.npz` 文件      | 可选，加上就生成                                           |
'''