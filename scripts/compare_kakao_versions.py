#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'  # macOS 기본 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지


import os, argparse, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ORDER = ["강한부정","부정","중립","긍정","강한긍정"]

def read_table(path: str) -> pd.DataFrame:
    return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)

def dist(df: pd.DataFrame, col="sent_5class"):
    counts = df[col].value_counts()
    cnt = {k:int(counts.get(k,0)) for k in ORDER}
    total = sum(cnt.values())
    ratio = {k:(cnt[k]/total*100 if total else 0.0) for k in ORDER}
    return cnt, ratio, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", default="data/processed/kakao_5class_auto_applied.xlsx")
    ap.add_argument("--aggr", default="data/processed/kakao_5class_aggr.xlsx")
    ap.add_argument("--label-col", default="sent_5class")
    ap.add_argument("--outdir", default="outputs_compare_kakao")
    args = ap.parse_args()

    a = read_table(args.auto)
    g = read_table(args.aggr)

    ca, ra, ta = dist(a, args.label_col)
    cg, rg, tg = dist(g, args.label_col)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df_counts = pd.DataFrame({
        "class": ORDER,
        "auto_count": [ca[k] for k in ORDER],
        "aggr_count": [cg[k] for k in ORDER],
    })
    df_counts["delta_count"] = df_counts["aggr_count"] - df_counts["auto_count"]

    df_ratio = pd.DataFrame({
        "class": ORDER,
        "auto_ratio(%)": [round(ra[k],2) for k in ORDER],
        "aggr_ratio(%)": [round(rg[k],2) for k in ORDER],
    })
    df_ratio["delta_ratio(%)"] = (df_ratio["aggr_ratio(%)"] - df_ratio["auto_ratio(%)"]).round(2)

    # 저장
    df_counts.to_csv(outdir/"kakao_versions_counts.csv", index=False, encoding="utf-8-sig")
    df_ratio.to_csv(outdir/"kakao_versions_ratios.csv", index=False, encoding="utf-8-sig")
    with open(outdir/"README.md","w",encoding="utf-8") as f:
        f.write(f"# Kakao Versions Compare\n- total_auto={ta}\n- total_aggr={tg}\n")

    # 막대그래프 (비율)
    x = np.arange(len(ORDER))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x - w/2, [ra[k] for k in ORDER], width=w, label="auto")
    ax.bar(x + w/2, [rg[k] for k in ORDER], width=w, label="aggr")
    ax.set_xticks(x); ax.set_xticklabels(ORDER)
    ax.set_ylabel("ratio (%)")
    ax.set_title("Kakao 5-class ratio: auto vs aggr")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir/"kakao_versions_ratio_bar.png", dpi=200)

    print("[OK] saved:",
          outdir/"kakao_versions_counts.csv",
          outdir/"kakao_versions_ratios.csv",
          outdir/"kakao_versions_ratio_bar.png")

if __name__ == "__main__":
    main()
