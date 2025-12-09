#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
top_keywords_by_class.py
- 입력: 5클래스 라벨이 포함된 파일(예: kakao_5class_final.xlsx)
- 출력: 클래스별 Top-N 키워드(unigram/bigram) 표 + 선택적 막대그래프
- 토큰화: 간단 전처리 + 공백 기준 토큰화(커뮤니티 단문에 적합)
- 점수 : log-odds(클래스 vs 나머지) + 라플라스 스무딩
"""
import argparse, re, math, os
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트(맥)
import matplotlib
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

LABELS = ["강한부정","부정","중립","긍정","강한긍정"]

DEFAULT_STOPS = set("""
그리고 그러나 하지만 그래서 그래서요 근데 혹시 그냥 매우 너무 진짜 조금 약간
오늘 내일 지금 여기 저기 뭐지 뭐임 이런 요런 저런거 그런거 이런거 그런 그리고요
ㅋㅋ ㅎㅎ ㅠㅠ ㅡㅡ ;; ?? !! … …? …! ㄱㄱ ㄳ ㅇㅋ ㅇㅇ ㅅㅅ ㄴㄴ
주가 주식 시장 뉴스 기사 링크 사진 이미지 영상 참고 퍼옴
""".split())

def read_table(path: Path) -> pd.DataFrame:
    return pd.read_excel(path) if str(path).lower().endswith(".xlsx") else pd.read_csv(path)

def normalize_txt(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"https?://\S+", " ", s)          # URL 제거
    s = re.sub(r"[^\w\sㄱ-ㅎ가-힣]", " ", s)      # 특수문자 제거(한글/숫자/영문/밑줄/공백만)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    toks = [t for t in s.split() if t and t not in DEFAULT_STOPS and not t.isdigit() and len(t) >= 2]
    return toks

def to_bigrams(toks):
    return [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks)-1)]

def collect_counts(df: pd.DataFrame, text_col: str, label_col: str, use_bigrams=False, min_count=5):
    # 전체/클래스별 카운트
    total_c = Counter()
    by_cls = {lab: Counter() for lab in LABELS}

    for _, row in df.iterrows():
        lab = row.get(label_col, "")
        if lab not in LABELS: continue
        toks = tokenize(normalize_txt(row.get(text_col, "")))
        if use_bigrams:
            toks = to_bigrams(toks)
        total_c.update(toks)
        by_cls[lab].update(toks)

    # 희귀 토큰 제거
    vocab = {w for w, c in total_c.items() if c >= min_count}
    total_c = Counter({w:c for w,c in total_c.items() if w in vocab})
    for lab in LABELS:
        by_cls[lab] = Counter({w:c for w,c in by_cls[lab].items() if w in vocab})

    return total_c, by_cls, len(vocab)

def log_odds_table(total_c: Counter, by_cls: dict, topn=20, prior=0.5):
    """클래스 vs 나머지 log-odds (라플라스 스무딩 prior)"""
    # 총합
    N_total = sum(total_c.values()) + 1e-12
    rows = []
    for lab in LABELS:
        N_lab = sum(by_cls[lab].values())
        N_rest = N_total - N_lab
        # 분모가 0일 수 있으므로 최소치 가드
        if N_lab <= 0 or N_rest <= 0:
            continue
        for w, c_lab in by_cls[lab].items():
            c_rest = total_c[w] - c_lab
            # 라플라스 스무딩
            p_lab  = (c_lab  + prior) / (N_lab  + prior*len(total_c))
            p_rest = (c_rest + prior) / (N_rest + prior*len(total_c))
            score = math.log(p_lab / p_rest)
            rows.append((lab, w, c_lab, total_c[w], p_lab, p_rest, score))
    out = pd.DataFrame(rows, columns=["class","token","count_in_class","count_total","p_in_class","p_in_rest","log_odds"])
    # 클래스별 Top 정렬
    tables = {}
    for lab in LABELS:
        tbl = out[out["class"]==lab].sort_values("log_odds", ascending=False).head(topn)
        tables[lab] = tbl.reset_index(drop=True)
    return tables

def save_tables_as_excel(tables_uni: dict, tables_bi: dict, out_path: Path):
    with pd.ExcelWriter(out_path) as w:
        for lab in LABELS:
            if lab in tables_uni:
                tables_uni[lab].to_excel(w, sheet_name=f"{lab}_uni", index=False)
            if lab in tables_bi:
                tables_bi[lab].to_excel(w, sheet_name=f"{lab}_bi", index=False)

def plot_bar(tables: dict, outdir: Path, title_suffix="uni", topk=10):
    outdir.mkdir(parents=True, exist_ok=True)
    for lab in LABELS:
        if lab not in tables: continue
        df = tables[lab].head(topk)
        if df.empty: continue
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(df["token"], df["log_odds"])
        ax.set_title(f"{lab} Top{topk} ({title_suffix}) - log-odds")
        ax.set_ylabel("log-odds (클래스 vs 나머지)")
        ax.set_xticklabels(df["token"], rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(outdir / f"{lab}_top{topk}_{title_suffix}.png", dpi=200)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="예: data/processed/kakao_5class_final.xlsx")
    ap.add_argument("--text-col", default="본문")
    ap.add_argument("--label-col", default="sent_5class")
    ap.add_argument("--outdir", default="outputs_keywords_kakao")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--min-count", type=int, default=5, help="전체 빈도 하한(희귀어 제거)")
    ap.add_argument("--make-charts", action="store_true")
    args = ap.parse_args()

    df = read_table(Path(args.input))
    total_uni, bycls_uni, V1 = collect_counts(df, args.text_col, args.label_col, use_bigrams=False, min_count=args.min_count)
    total_bi,  bycls_bi,  V2 = collect_counts(df, args.text_col, args.label_col, use_bigrams=True,  min_count=max(3, args.min_count//2))

    tables_uni = log_odds_table(total_uni, bycls_uni, topn=args.topn, prior=0.5)
    tables_bi  = log_odds_table(total_bi,  bycls_bi,  topn=args.topn, prior=0.5)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    # CSV로도 저장
    for lab in LABELS:
        if lab in tables_uni:
            tables_uni[lab].to_csv(outdir/f"{lab}_unigram.csv", index=False, encoding="utf-8-sig")
        if lab in tables_bi:
            tables_bi[lab].to_csv(outdir/f"{lab}_bigram.csv",  index=False, encoding="utf-8-sig")

    # Excel 묶음
    save_tables_as_excel(tables_uni, tables_bi, outdir/"top_keywords.xlsx")

    # 선택: 그래프
    if args.make_charts:
        plot_bar(tables_uni, outdir/"charts_uni", title_suffix="uni", topk=min(10, args.topn))
        plot_bar(tables_bi,  outdir/"charts_bi",  title_suffix="bi",  topk=min(10, args.topn))

    print("[OK] saved →", outdir/"top_keywords.xlsx")
    print("[OK] CSVs  →", outdir/"<class>_unigram.csv, <class>_bigram.csv")
    if args.make_charts:
        print("[OK] charts:", (outdir/"charts_uni"), (outdir/"charts_bi"))

if __name__ == "__main__":
    main()
