#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
curate_extremes.py
- 입력: 5클래스 라벨 포함 파일(예: kakao_5class_final.xlsx)
- 출력: 강한부정/강한긍정 각각 상위 K개(신뢰 높은 순)만 모은 엑셀
- 기준: s = p_pos - p_neg (없으면 계산), 강긍정은 s↑, 강부정은 s↓
"""
import argparse
from pathlib import Path
import pandas as pd

def read_table(path: Path) -> pd.DataFrame:
    return pd.read_excel(path) if str(path).lower().endswith(".xlsx") else pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", default="본문")
    ap.add_argument("--label-col", default="sent_5class")
    ap.add_argument("--out", default="data/curation/kakao_top_extremes.xlsx")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--min-char", type=int, default=2, help="너무 짧은 텍스트 제외(문자 수)")
    args = ap.parse_args()

    inp = Path(args.input)
    df = read_table(inp).copy()

    # s 점수 확보
    if "s" not in df.columns:
        if {"p_pos","p_neg"}.issubset(df.columns):
            df["s"] = df["p_pos"] - df["p_neg"]
        else:
            raise SystemExit("[ERR] s도 없고 p_pos/p_neg도 없어 선별 기준을 만들 수 없습니다.")

    # 너무 짧은 텍스트 제외
    df["char_len"] = df[args.text_col].astype(str).str.replace(r"\s+", "", regex=True).str.len()
    df = df[df["char_len"] >= args.min_char].reset_index(drop=True)

    # 강긍정/강부정 추출
    need = {"강한긍정": "desc", "강한부정": "asc"}
    parts = []
    for cls, order in need.items():
        sub = df[df[args.label_col] == cls].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("s", ascending=(order=="asc")).head(args.per_class)
        sub["선정클래스"] = cls
        parts.append(sub)

    if not parts:
        raise SystemExit("[ERR] 해당 클래스에서 뽑을 샘플이 없습니다.")

    out_df = pd.concat(parts, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_path, index=False)
    print(f"[OK] saved → {out_path} (rows={len(out_df)})")

if __name__ == "__main__":
    main()
