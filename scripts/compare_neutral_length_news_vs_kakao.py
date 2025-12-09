#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_neutral_length_news_vs_kakao.py
- 입력: 각 데이터셋의 neutral_vs_length 결과 폴더(뉴스/카카오)
- 출력: 버킷 비교표(csv) + 비교 그래프(png)
- 변경점: 'count' 대신 'n', 'mean' 대신 'neutral_ratio(%)'로 저장된 경우도 자동 대응
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# macOS 한글 폰트 설정
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False


def read_len_tables(dir_path: Path, kind: str):
    """kind in {'char','token'}"""
    if kind == "char":
        f = dir_path / "neutral_by_char_bucket.csv"
        bucket_col = "char_bucket"
    else:
        f = dir_path / "neutral_by_token_bucket.csv"
        bucket_col = "tok_bucket"

    df = pd.read_csv(f)

    # ---- 컬럼 정규화 ----
    # count 대신 n 으로 저장된 경우
    if "count" not in df.columns and "n" in df.columns:
        df = df.rename(columns={"n": "count"})

    # mean 대신 neutral_ratio(%) 만 있는 경우
    if "mean" not in df.columns and "neutral_ratio(%)" in df.columns:
        df["mean"] = df["neutral_ratio(%)"] / 100.0

    need = {bucket_col, "mean", "count"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[ERR] 컬럼 부족: {f} -> {list(df.columns)}")

    # 버킷/평균/카운트 NaN 제거(그래프 주석 오류 방지)
    df = df.dropna(subset=[bucket_col, "mean", "count"]).reset_index(drop=True)
    return df, bucket_col


def merge_compare(news_df, kakao_df, bucket_col: str):
    """뉴스 순서를 기준으로 병합하고 delta 계산"""
    news_df = news_df.copy()
    news_df["__ord__"] = range(len(news_df))

    n = news_df.rename(columns={"mean": "news_neutral", "count": "news_count"})
    k = kakao_df.rename(columns={"mean": "kakao_neutral", "count": "kakao_count"})

    out = pd.merge(
        n[[bucket_col, "news_neutral", "news_count", "__ord__"]],
        k[[bucket_col, "kakao_neutral", "kakao_count"]],
        on=bucket_col,
        how="outer",
    )
    out = out.sort_values("__ord__").drop(columns="__ord__")

    # delta(카카오 - 뉴스)
    out["delta_kakao_minus_news"] = out["kakao_neutral"] - out["news_neutral"]

    # 퍼센트 버전
    out_pct = out.copy()
    for c in ["news_neutral", "kakao_neutral", "delta_kakao_minus_news"]:
        out_pct[c] = out_pct[c] * 100.0

    return out, out_pct


def plot_lines(df_pct: pd.DataFrame, bucket_col: str, outpath: Path, title: str):
    p = df_pct[[bucket_col, "news_neutral", "kakao_neutral"]].dropna()
    # 버킷 레이블을 문자열로 통일
    labels = p[bucket_col].astype(str).tolist()
    x = list(range(len(p)))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, p["news_neutral"], marker="o", label="뉴스")
    ax.plot(x, p["kakao_neutral"], marker="o", label="카카오")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("중립 비율(%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news-dir", default="outputs_analysis_news")
    ap.add_argument("--kakao-dir", default="outputs_analysis_kakao")
    ap.add_argument("--outdir", default="outputs_compare_len")
    args = ap.parse_args()

    news_dir = Path(args.news_dir)
    kakao_dir = Path(args.kakao_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 문자 길이 버킷 비교
    n_char, char_col = read_len_tables(news_dir, "char")
    k_char, _ = read_len_tables(kakao_dir, "char")
    char_cmp, char_cmp_pct = merge_compare(n_char, k_char, char_col)
    char_cmp.to_csv(outdir / "char_bucket_compare.csv", index=False, encoding="utf-8-sig")
    char_cmp_pct.to_csv(outdir / "char_bucket_compare_pct.csv", index=False, encoding="utf-8-sig")
    plot_lines(
        char_cmp_pct,
        char_col,
        outdir / "char_neutral_news_vs_kakao.png",
        "문자 길이 버킷별 중립 비율 비교(뉴스 vs 카카오)",
    )

    # 2) 토큰 길이 버킷 비교
    n_tok, tok_col = read_len_tables(news_dir, "token")
    k_tok, _ = read_len_tables(kakao_dir, "token")
    tok_cmp, tok_cmp_pct = merge_compare(n_tok, k_tok, tok_col)
    tok_cmp.to_csv(outdir / "token_bucket_compare.csv", index=False, encoding="utf-8-sig")
    tok_cmp_pct.to_csv(outdir / "token_bucket_compare_pct.csv", index=False, encoding="utf-8-sig")
    plot_lines(
        tok_cmp_pct,
        tok_col,
        outdir / "token_neutral_news_vs_kakao.png",
        "토큰 길이 버킷별 중립 비율 비교(뉴스 vs 카카오)",
    )

    print("[OK] saved:", outdir / "char_bucket_compare.csv", outdir / "char_bucket_compare_pct.csv",
          outdir / "char_neutral_news_vs_kakao.png")
    print("[OK] saved:", outdir / "token_bucket_compare.csv", outdir / "token_bucket_compare_pct.csv",
          outdir / "token_neutral_news_vs_kakao.png")


if __name__ == "__main__":
    main()
