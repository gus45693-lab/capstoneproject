#!/usr/bin/env python
import argparse, pandas as pd, os, re

MAP_5TO3 = {
    "강한부정":"부정", "부정":"부정",
    "중립":"중립",
    "긍정":"긍정", "강한긍정":"긍정"
}

def norm_label(x:str):
    if pd.isna(x): return x
    x = str(x).strip()
    # 안전 처리(영문/공백 변형 등)
    x = re.sub(r"\s+", "", x)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="5클래스 결과 파일(.xlsx)")
    ap.add_argument("--text-col", required=True, help="텍스트 컬럼명(예: 본문)")
    ap.add_argument("--five-col", default="sent_5class", help="5클래스 라벨 컬럼명")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_excel(args.input) if args.input.lower().endswith(".xlsx") else pd.read_csv(args.input)
    if args.text_col not in df.columns or args.five_col not in df.columns:
        raise SystemExit(f"[ERR] 컬럼 확인: text={args.text_col}, five={args.five_col}, 실제={list(df.columns)}")

    df["label_5class_original"] = df[args.five_col].astype(str)
    df["label_3_from5"] = df[args.five_col].map(lambda v: MAP_5TO3.get(norm_label(v), None))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out.lower().endswith(".xlsx"):
        df.to_excel(args.out, index=False)
    else:
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] 5→3 접기 저장 → {args.out}")

if __name__ == "__main__":
    main()
