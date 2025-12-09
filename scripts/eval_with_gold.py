#!/usr/bin/env python
import argparse, os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

CLS = ["강한부정","부정","중립","긍정","강한긍정"]
ALIASES = {
    "강부정":"강한부정","매우부정":"강한부정",
    "강긍정":"강한긍정","매우긍정":"강한긍정",
    "부정":"부정","중립":"중립","긍정":"긍정",
    "강한부정":"강한부정","강한긍정":"강한긍정",
}

def canon(x):
    if pd.isna(x): return None
    s = str(x).strip()
    return ALIASES.get(s, s)

def to_idx(seq):
    return [CLS.index(x) if x in CLS else -1 for x in seq]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)      # data/label/label_pack_200.xlsx
    ap.add_argument("--pred", required=True)      # 예: data/processed/kakao_5class_final.xlsx
    ap.add_argument("--text-col", default="본문")
    ap.add_argument("--pred-col", default="sent_5class")
    ap.add_argument("--outdir", default="outputs_gold_eval")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    g = pd.read_excel(args.gold)
    p = pd.read_excel(args.pred)

    # 정리
    g = g[[args.text_col, "gold_5"]].dropna(subset=[args.text_col]).copy()
    g["gold_5"] = g["gold_5"].map(canon)

    p = p[[args.text_col, args.pred_col]].dropna(subset=[args.text_col]).copy()
    p = p.rename(columns={args.pred_col:"pred_5"})
    p["pred_5"] = p["pred_5"].map(canon)

    g_labeled = g[g["gold_5"].isin(CLS)].copy()
    if g_labeled.empty:
        print("[ERR] gold_5 라벨이 비어있거나 허용 라벨과 불일치합니다:", CLS)
        return

    df = pd.merge(g_labeled, p, on=args.text_col, how="inner")
    if df.empty:
        print("[ERR] 골드와 예측이 본문으로 매칭된 행이 없습니다. 텍스트가 수정되었는지 확인하세요.")
        return

    y_true = to_idx(df["gold_5"])
    y_pred = to_idx(df["pred_5"])
    keep = [(a!=-1 and b!=-1) for a,b in zip(y_true,y_pred)]
    df = df.loc[keep].reset_index(drop=True)
    if df.empty:
        print("[ERR] 유효 라벨이 없습니다. 허용 라벨만 사용하세요:", CLS)
        return

    y_true = np.array([a for a,k in zip(y_true,keep) if k])
    y_pred = np.array([a for a,k in zip(y_pred,keep) if k])

    # 지표/리포트
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLS)))

    df.to_excel(os.path.join(args.outdir, "merge_gold_pred.xlsx"), index=False)
    pd.DataFrame({"metric":["macro_f1"], "value":[macro_f1]}).to_csv(
        os.path.join(args.outdir, "metrics.csv"), index=False)

    with open(os.path.join(args.outdir,"classification_report.txt"),"w",encoding="utf-8") as f:
        f.write(classification_report(
            y_true, y_pred,
            labels=list(range(len(CLS))),
            target_names=CLS, digits=4, zero_division=0))

    # 혼동행렬 그림
    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(CLS)), CLS); plt.yticks(range(len(CLS)), CLS)
    for i in range(len(CLS)):
        for j in range(len(CLS)):
            plt.text(j, i, cm[i,j], ha="center", va="center", fontsize=10)
    plt.title(f"Confusion Matrix (macro-F1={macro_f1:.3f})")
    plt.xlabel("Pred"); plt.ylabel("Gold"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,"confusion_matrix.png"), dpi=200)
    print(f"[OK] saved -> {args.outdir}, macro-F1={macro_f1:.4f}")

if __name__ == "__main__":
    main()
