#!/usr/bin/env python
import argparse, os, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="data/processed/news_5class.xlsx (p_neg/p_neu/p_pos 포함)")
    ap.add_argument("--text-col", required=True, help="텍스트 컬럼명 (예: 본문)")
    ap.add_argument("--out", required=True, help="저장 경로 (예: data/processed/news_baseline_3class.xlsx)")
    args = ap.parse_args()

    df = pd.read_excel(args.input) if args.input.lower().endswith(".xlsx") else pd.read_csv(args.input)
    need = [args.text_col, "p_neg", "p_neu", "p_pos"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERR] 필요한 컬럼 없음: {missing}  → infer_news_to_5class.py로 p_* 생성했는지 확인")

    out = df.copy()
    out["s_value_3"] = out["p_pos"] - out["p_neg"]
    # argmax로 3클래스 라벨
    idx = out[["p_neg", "p_neu", "p_pos"]].values.argmax(axis=1)
    mapping = {0: "부정", 1: "중립", 2: "긍정"}
    out["label_3class"] = [mapping[i] for i in idx]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out.lower().endswith(".xlsx"):
        out.to_excel(args.out, index=False)
    else:
        out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[OK] 3클래스 베이스라인 저장 → {args.out}")

if __name__ == "__main__":
    main()
