import argparse, pandas as pd, sys
ALLOWED = {"강한부정","부정","중립","긍정","강한긍정"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    args = ap.parse_args()

    g = pd.read_excel(args.gold)
    if "gold_5" not in g.columns:
        print("[ERR] gold_5 컬럼이 없습니다."); sys.exit(1)

    bad = g[~g["gold_5"].isin(ALLOWED) | g["gold_5"].isna()]
    print(f"[INFO] 총 {len(g)}건, 공란/오탈자 {len(bad)}건")
    if len(bad):
        bad.to_excel(args.gold.replace(".xlsx","_bad.xlsx"), index=False)
        print("[HINT] 잘못된 행 저장 →", args.gold.replace(".xlsx","_bad.xlsx"))
        sys.exit(2)
    print("[OK] gold_5 검증 통과")

if __name__ == "__main__":
    main()
