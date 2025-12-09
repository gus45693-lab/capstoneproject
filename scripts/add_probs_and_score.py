#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, math
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def read_table(p: Path) -> pd.DataFrame:
    return pd.read_excel(p) if str(p).lower().endswith(".xlsx") else pd.read_csv(p)

def write_table(df: pd.DataFrame, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    if str(p).lower().endswith(".xlsx"):
        df.to_excel(p, index=False)
    else:
        df.to_csv(p, index=False, encoding="utf-8-sig")

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", default="본문")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="snunlp/KR-FinBERT-SC")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.out)
    df = read_table(inp).copy()
    if args.text_col not in df.columns:
        raise SystemExit(f"[ERR] 텍스트 컬럼 없음: {args.text_col}")

    device = get_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device); model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    # 라벨 인덱스 찾기(대소문자 무시)
    def idx_for(substr):
        for i, lab in id2label.items():
            low = lab.lower()
            if substr in low:
                return i
        return None
    i_neg = idx_for("neg")
    i_neu = idx_for("neu")
    i_pos = idx_for("pos")
    if None in (i_neg, i_neu, i_pos):
        raise SystemExit(f"[ERR] 라벨 매핑 실패: id2label={id2label}")

    texts = df[args.text_col].astype(str).tolist()
    p_neg, p_neu, p_pos = [], [], []

    for i in tqdm(range(0, len(texts), args.batch), desc="Scoring"):
        batch = texts[i:i+args.batch]
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            probs = F.softmax(out.logits, dim=-1).detach().cpu()
        p_neg += probs[:, i_neg].tolist()
        p_neu += probs[:, i_neu].tolist()
        p_pos += probs[:, i_pos].tolist()

    df["p_neg"] = p_neg
    df["p_neu"] = p_neu
    df["p_pos"] = p_pos
    df["s"] = df["p_pos"] - df["p_neg"]

    write_table(df, outp)
    print(f"[OK] saved → {outp}  (rows={len(df)})")

if __name__ == "__main__":
    main()
