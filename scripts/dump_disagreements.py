#!/usr/bin/env python
import argparse, pandas as pd, os, re

def normalize(s):
    s = "" if pd.isna(s) else str(s)
    return re.sub(r"\s+", " ", s).strip()

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True)
ap.add_argument("--converted", required=True)
ap.add_argument("--text-col", required=True)
ap.add_argument("--base-label", default="label_3class")
ap.add_argument("--conv-label", default="label_3_from5")
ap.add_argument("--out", required=True)
args = ap.parse_args()

b = pd.read_excel(args.baseline) if args.baseline.endswith(".xlsx") else pd.read_csv(args.baseline)
c = pd.read_excel(args.converted) if args.converted.endswith(".xlsx") else pd.read_csv(args.converted)

b["_t"] = b[args.text_col].map(normalize)
c["_t"] = c[args.text_col].map(normalize)
m = pd.merge(
    b[[args.text_col, "_t", args.base_label]],
    c[[args.text_col, "_t", args.conv_label]],
    on="_t", how="inner", suffixes=("_base","_conv")
)

diff = m[m[args.base_label] != m[args.conv_label]].copy()
diff = diff.rename(columns={
    f"{args.text_col}_base":"text_baseline_src",
    f"{args.text_col}_conv":"text_converted_src"
})
# 보기 좋게 정렬
cols = ["_t", args.base_label, args.conv_label, "text_baseline_src", "text_converted_src"]
diff = diff[cols] if set(cols).issubset(diff.columns) else diff

os.makedirs(os.path.dirname(args.out), exist_ok=True)
diff.head(10).to_excel(args.out, index=False)
print(f"[OK] 불일치 상위 10개 저장 → {args.out} (총 불일치 {len(diff)}건)")
