#!/usr/bin/env python
import argparse, pandas as pd, numpy as np, os, re
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def normalize_text(s):
    if pd.isna(s): return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Step1에서 만든 3클래스 예측 파일")
    ap.add_argument("--converted", required=True, help="Step2에서 만든 5→3 접기 파일")
    ap.add_argument("--text-col", required=True, help="공통 텍스트 컬럼명(예: 본문)")
    ap.add_argument("--base-label", default="label_3class", help="베이스라인 라벨 컬럼")
    ap.add_argument("--conv-label", default="label_3_from5", help="5→3 라벨 컬럼")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    b = pd.read_excel(args.baseline) if args.baseline.lower().endswith(".xlsx") else pd.read_csv(args.baseline)
    c = pd.read_excel(args.converted) if args.converted.lower().endswith(".xlsx") else pd.read_csv(args.converted)

    for col in [args.text_col, args.base_label]:
        if col not in b.columns: raise SystemExit(f"[ERR] baseline에 {col} 없음: {list(b.columns)}")
    for col in [args.text_col, args.conv_label]:
        if col not in c.columns: raise SystemExit(f"[ERR] converted에 {col} 없음: {list(c.columns)}")

    b["_text_norm"] = b[args.text_col].map(normalize_text)
    c["_text_norm"] = c[args.text_col].map(normalize_text)

    merged = pd.merge(b[[args.text_col, "_text_norm", args.base_label]],
                      c[[args.text_col, "_text_norm", args.conv_label]],
                      on="_text_norm", suffixes=("_base","_conv"))

    y_true = merged[f"{args.base_label}"]
    y_pred = merged[f"{args.conv_label}"]

    labels = ["부정","중립","긍정"]
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", labels=labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=False, digits=4)

    os.makedirs(args.outdir, exist_ok=True)
    # 저장
    pd.DataFrame({"metric":["accuracy","macro_f1"],"value":[acc, mf1]}).to_csv(
        os.path.join(args.outdir,"metrics_3class.csv"), index=False, encoding="utf-8-sig"
    )
    with open(os.path.join(args.outdir,"classification_report.txt"),"w",encoding="utf-8") as f:
        f.write(rep)

    # 혼동행렬 그림
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Pred (5→3)"); ax.set_ylabel("True (Baseline 3)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir,"confusion_matrix.png"), dpi=200)
    print(f"[OK] 저장: {args.outdir}/metrics_3class.csv, classification_report.txt, confusion_matrix.png")
    print(f"[INFO] ACC={acc:.4f}, Macro-F1={mf1:.4f}")

if __name__ == "__main__":
    main()
