#!/usr/bin/env python
import argparse, pandas as pd, numpy as np, os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

LABELS = ["negative", "neutral", "positive"]

def softmax(x):
    x = np.array(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--model", default="snunlp/KR-FinBERT-SC")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_excel(args.input) if args.input.lower().endswith(".xlsx") else pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise SystemExit(f"[ERR] text-col '{args.text_col}' 없음. 컬럼들: {list(df.columns)}")

    texts = df[args.text_col].astype(str).fillna("").tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    p_neg, p_neu, p_pos, labels = [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch), desc="Infer"):
            batch = texts[i:i+args.batch]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
            enc = {k:v.to(device) for k,v in enc.items()}
            logits = model(**enc).logits.detach().cpu().numpy()
            probs = np.apply_along_axis(softmax, 1, logits)
            p_neg.extend(probs[:,0]); p_neu.extend(probs[:,1]); p_pos.extend(probs[:,2])
            labels.extend([LABELS[j] for j in probs.argmax(axis=1)])

    out = df.copy()
    out["p_neg"] = p_neg
    out["p_neu"] = p_neu
    out["p_pos"] = p_pos
    out["s_value_3"] = out["p_pos"] - out["p_neg"]
    # 3클래스 라벨(Argmax)
    out["label_3class"] = [ ["부정","중립","긍정"][["negative","neutral","positive"].index(l)] for l in labels ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out.lower().endswith(".xlsx"):
        out.to_excel(args.out, index=False)
    else:
        out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[OK] 3클래스 베이스라인 저장 → {args.out}")

if __name__ == "__main__":
    main()
