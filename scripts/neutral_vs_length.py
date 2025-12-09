#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 한글 깨짐 방지(맥)
import matplotlib
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

def read_table(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)

def char_len(s: str) -> int:
    if not isinstance(s, str): return 0
    # 공백 제거 후 길이(이모지/기호 포함)
    return len(re.sub(r"\s+", "", s))

def token_len(s: str) -> int:
    if not isinstance(s, str): return 0
    # 공백 기준 토큰 수(간단)
    return len(re.findall(r"\S+", s))

def bucketize(vals, edges, labels=None):
    return pd.cut(vals, bins=edges, labels=labels, include_lowest=True, right=True)

def barplot(ax, xlabels, values, title, ylabel):
    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

def lineplot(ax, xvals, yvals, title, xlabel, ylabel):
    ax.plot(xvals, yvals)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="kakao_5class_final.xlsx 경로")
    ap.add_argument("--text-col", default="본문")
    ap.add_argument("--label-col", default="sent_5class")
    ap.add_argument("--outdir", default="outputs_analysis_kakao")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = read_table(in_path)

    if args.text_col not in df.columns:
        raise SystemExit(f"[ERR] 텍스트 컬럼 없음: {args.text_col}")
    if args.label_col not in df.columns:
        raise SystemExit(f"[ERR] 라벨 컬럼 없음: {args.label_col}")

    # 길이 계산
    df = df.copy()
    df["char_len"] = df[args.text_col].apply(char_len)
    df["tok_len"]  = df[args.text_col].apply(token_len)
    df["is_neutral"] = (df[args.label_col] == "중립").astype(int)

    # ① 문자 길이 버킷별 중립 비율
    char_edges  = [0,3,5,8,12,16,20,30,40,60,100, np.inf]
    char_labels = ["≤3","4–5","6–8","9–12","13–16","17–20","21–30","31–40","41–60","61–100","100+"]
    df["char_bucket"] = bucketize(df["char_len"], char_edges, char_labels)
    g_char = df.groupby("char_bucket")["is_neutral"].agg(["mean","count"]).reset_index()
    g_char["neutral_ratio(%)"] = (g_char["mean"]*100).round(2)
    g_char.rename(columns={"count":"n"}, inplace=True)
    g_char.to_csv(outdir/"neutral_by_char_bucket.csv", index=False, encoding="utf-8-sig")

    # ② 토큰 길이 버킷별 중립 비율
    tok_edges  = [0,1,2,3,4,5,6,8,10,15,20,30, np.inf]
    tok_labels = ["1","2","3","4","5","6","7–8","9–10","11–15","16–20","21–30","30+"]
    df["tok_bucket"] = bucketize(df["tok_len"], tok_edges, tok_labels)
    g_tok = df.groupby("tok_bucket")["is_neutral"].agg(["mean","count"]).reset_index()
    g_tok["neutral_ratio(%)"] = (g_tok["mean"]*100).round(2)
    g_tok.rename(columns={"count":"n"}, inplace=True)
    g_tok.to_csv(outdir/"neutral_by_token_bucket.csv", index=False, encoding="utf-8-sig")

    # ③ 문자 길이별(정수) 중립 비율 라인
    by_len = df.groupby("char_len")["is_neutral"].mean().mul(100).reset_index().sort_values("char_len")
    by_len.to_csv(outdir/"neutral_ratio_by_char_len.csv", index=False, encoding="utf-8-sig")

    # --- 그래프 저장 ---
    # bar: char bucket
    fig1, ax1 = plt.subplots(figsize=(8,4))
    barplot(ax1, g_char["char_bucket"].astype(str).tolist(), g_char["neutral_ratio(%)"].tolist(),
            "문자 길이 버킷별 중립 비율", "중립 비율(%)")
    fig1.tight_layout(); fig1.savefig(outdir/"bar_neutral_by_char_bucket.png", dpi=200)

    # bar: token bucket
    fig2, ax2 = plt.subplots(figsize=(8,4))
    barplot(ax2, g_tok["tok_bucket"].astype(str).tolist(), g_tok["neutral_ratio(%)"].tolist(),
            "토큰 길이 버킷별 중립 비율", "중립 비율(%)")
    fig2.tight_layout(); fig2.savefig(outdir/"bar_neutral_by_token_bucket.png", dpi=200)

    # line: char length
    fig3, ax3 = plt.subplots(figsize=(8,4))
    lineplot(ax3, by_len["char_len"], by_len["is_neutral"], "문자 길이별 중립 비율(정수)", "문자 길이", "중립 비율(%)")
    fig3.tight_layout(); fig3.savefig(outdir/"line_neutral_by_char_len.png", dpi=200)

    print("[OK] saved:",
          outdir/"neutral_by_char_bucket.csv",
          outdir/"neutral_by_token_bucket.csv",
          outdir/"neutral_ratio_by_char_len.csv")
    print("[OK] charts:",
          outdir/"bar_neutral_by_char_bucket.png",
          outdir/"bar_neutral_by_token_bucket.png",
          outdir/"line_neutral_by_char_len.png")

if __name__ == "__main__":
    main()
