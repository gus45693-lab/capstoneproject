import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import html

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except Exception:
    HAS_CAPTUM = False

def softmax(x):
    x = np.array(x, dtype=np.float64)
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()

def color_span(token, score):
    s = max(-1.0, min(1.0, float(score)))
    alpha = abs(s)
    if s > 0:
        return f'<span style="background-color: rgba(0, 102, 204, {alpha:.2f}); color: white">{html.escape(token)}</span>'
    elif s < 0:
        return f'<span style="background-color: rgba(204, 0, 0, {alpha:.2f}); color: white">{html.escape(token)}</span>'
    else:
        return html.escape(token)

def tokens_to_words(tokens, attrib):
    words = []
    weights = []
    current = ""
    curr_w = []
    for tok, w in zip(tokens, attrib):
        if tok.startswith("##"):
            piece = tok[2:]
            current += piece
            curr_w.append(w)
        else:
            if current:
                words.append(current)
                weights.append(float(np.mean(curr_w)))
            current = tok
            curr_w = [w]
    if current:
        words.append(current)
        weights.append(float(np.mean(curr_w)))
    filt_words, filt_w = [], []
    for t, w in zip(words, weights):
        if t in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        filt_words.append(t)
        filt_w.append(w)
    return filt_words, filt_w

def explain_with_attention(model, tokenizer, text, device, max_length=256):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=False, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
    attentions = outputs.attentions
    last = attentions[-1].detach().cpu().numpy()[0]
    cls_to_tok = last[:, 0, :]
    scores = cls_to_tok.mean(axis=0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return tokens, scores

def explain_with_captum(model, tokenizer, text, target_idx, device, max_length=256):
    ig = IntegratedGradients(model)
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=False, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    def forward_func(input_ids_, attention_mask_):
        out = model(input_ids=input_ids_, attention_mask=attention_mask_)
        return out.logits

    attributions, _ = ig.attribute(
        inputs=input_ids,
        baselines=torch.zeros_like(input_ids),
        additional_forward_args=(attn_mask,),
        target=target_idx,
        return_convergence_delta=True,
        n_steps=24
    )
    attrib = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return tokens, attrib

def main():
    ap = argparse.ArgumentParser(description="감성 분류 근거(토큰 하이라이트) 생성")
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", default="")
    ap.add_argument("--model", default="snunlp/KR-FinBERT-SC")
    ap.add_argument("--outdir", default="outputs_explain")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(in_path)
    else:
        df = pd.read_csv(in_path)

    text_col = args.text_col
    if not text_col:
        cands = ["본문", "내용", "text", "body", "article", "news", "title", "제목"]
        for c in df.columns:
            lc = str(c).lower()
            if any(k in lc for k in ["text", "body", "article", "news", "title"]) or c in ["본문", "내용", "제목"]:
                text_col = c
                break
    if not text_col:
        raise SystemExit("[ERR] 텍스트 컬럼을 찾지 못했습니다. --text-col 로 지정해주세요.")

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    if "sent_5class" not in df.columns and "s_value" in df.columns:
        df["sent_5class"] = pd.cut(
            df["s_value"],
            bins=[-np.inf, -0.75, -0.55, 0.55, 0.75, np.inf],
            labels=["강한부정", "부정", "중립", "긍정", "강한긍정"]
        )

    classes = ["강한부정", "부정", "중립", "긍정", "강한긍정"]
    samples = []
    for c in classes:
        part = df[df["sent_5class"] == c].head(args.per_class)
        samples.append(part)
    samp_df = pd.concat(samples, axis=0, ignore_index=True)

    rows = []
    html_dir = outdir / "html"
    html_dir.mkdir(exist_ok=True, parents=True)

    for idx, row in tqdm(samp_df.iterrows(), total=len(samp_df), desc="[Explain]"):
        text = str(row[text_col])
        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=args.max_length).to(device)
        with torch.no_grad():
            out = model(**enc)
            probs = softmax(out.logits.detach().cpu().numpy()[0])
        p_neg, p_neu, p_pos = probs[0], (probs[1] if len(probs) > 2 else 1 - (probs[0] + probs[-1])), probs[-1]
        s_value = p_pos - p_neg
        target_idx = 2 if p_pos >= p_neg else 0

        try:
            if HAS_CAPTUM:
                tokens, attrib = explain_with_captum(model, tokenizer, text, target_idx, device, max_length=args.max_length)
            else:
                tokens, attrib = explain_with_attention(model, tokenizer, text, device, max_length=args.max_length)
        except Exception:
            tokens, attrib = explain_with_attention(model, tokenizer, text, device, max_length=args.max_length)

        words, weights = tokens_to_words(tokens, attrib)
        if len(weights) > 0:
            w = np.array(weights)
            if np.max(np.abs(w)) > 0:
                w = w / (np.max(np.abs(w)) + 1e-8)
            weights = w.tolist()

        pairs = list(zip(words, weights))
        pos_top = sorted(pairs, key=lambda x: x[1], reverse=True)[:5]
        neg_top = sorted(pairs, key=lambda x: x[1])[:5]

        highlighted = " ".join([color_span(t, s) for t, s in pairs])
        html_path = html_dir / f"sample_{idx}.html"
        html_path.write_text(f"<meta charset='utf-8'>\n<p>{highlighted}</p>\n", encoding="utf-8")

        rows.append({
            "sent_5class": row.get("sent_5class", ""),
            "s_value_infile": row.get("s_value", np.nan),
            "p_neg": p_neg, "p_neu": p_neu, "p_pos": p_pos,
            "s_value_model": s_value,
            "text": text,
            "top_positive_tokens": ", ".join([f"{t}({s:.2f})" for t, s in pos_top]),
            "top_negative_tokens": ", ".join([f"{t}({s:.2f})" for t, s in neg_top]),
            "html_path": str(html_path)
        })

    out_xlsx = outdir / "explain_samples.xlsx"
    pd.DataFrame(rows).to_excel(out_xlsx, index=False)
    print(f"[OK] 샘플 설명 저장 → {out_xlsx}")
    print(f"[OK] 하이라이트 HTML 폴더 → {html_dir}")

if __name__ == "__main__":
    main()
