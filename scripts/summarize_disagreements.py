#!/usr/bin/env python
import argparse, os, re, pandas as pd

def norm(s):
    s = "" if pd.isna(s) else str(s)
    return re.sub(r"\s+", " ", s).strip()

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True)
ap.add_argument("--converted", required=True)
ap.add_argument("--text-col", required=True)
ap.add_argument("--base-label", default="label_3class")
ap.add_argument("--conv-label", default="label_3_from5")
ap.add_argument("--outdir", required=True)
args = ap.parse_args()

b = pd.read_excel(args.baseline) if args.baseline.endswith(".xlsx") else pd.read_csv(args.baseline)
c = pd.read_excel(args.converted) if args.converted.endswith(".xlsx") else pd.read_csv(args.converted)
for col in [args.text_col, args.base_label]:
    assert col in b.columns, f"baseline에 {col} 없음"
for col in [args.text_col, args.conv_label]:
    assert col in c.columns, f"converted에 {col} 없음"

b["_t"] = b[args.text_col].map(norm)
c["_t"] = c[args.text_col].map(norm)
m = pd.merge(b[["_t", args.base_label]], c[["_t", args.conv_label]], on="_t", how="inner")

total = len(m)
diff = m[m[args.base_label] != m[args.conv_label]].copy()
n_diff = len(diff)
rate = (n_diff / total * 100) if total else 0.0

# 어떤 쌍이 갈렸는지 집계
pairs = diff.groupby([args.base_label, args.conv_label]).size().reset_index(name="count")
pairs = pairs.sort_values("count", ascending=False)

os.makedirs(args.outdir, exist_ok=True)
pd.DataFrame([{"total": total, "disagreements": n_diff, "rate_percent": round(rate, 2)}])\
  .to_csv(os.path.join(args.outdir, "disagreement_summary.csv"), index=False, encoding="utf-8-sig")
pairs.to_csv(os.path.join(args.outdir, "disagreement_pairs.csv"), index=False, encoding="utf-8-sig")

print(f"[OK] 요약 저장 → {args.outdir}/disagreement_summary.csv, disagreement_pairs.csv")
print(f"[INFO] total={total}, diff={n_diff}, rate={rate:.2f}%")
