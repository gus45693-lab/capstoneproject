import argparse
from pathlib import Path
import math
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# 5단계 라벨링 규칙
def label_5class(s, t1=0.55, t2=0.75):
    if s <= -t2: return "강한부정"
    if s <= -t1: return "부정"
    if -t1 < s < t1: return "중립"
    if s < t2: return "긍정"
    return "강한긍정"

def softmax(x):
    x = np.array(x, dtype=np.float64)
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()

def main():
    ap = argparse.ArgumentParser(description="뉴스 엑셀 감성추론 → 5단계 라벨링")
    ap.add_argument("--input", required=True, help="입력 엑셀 경로(예: NewsResult_20250519-20250523.xlsx)")
    ap.add_argument("--text-col", default="", help="본문/제목 등 텍스트 컬럼명(자동감지 실패시 지정)")
    ap.add_argument("--sheet", default=None, help="시트명(필요시)")
    ap.add_argument("--model", default="snunlp/KR-FinBERT-SC", help="허깅페이스 모델명 또는 로컬 경로")
    ap.add_argument("--out", default="", help="출력 경로(.xlsx). 기본: data/processed/<원본명>_5class.xlsx")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    if not in_path.exists():
        raise SystemExit(f"[ERR] 입력 파일 없음: {in_path}")

    # 출력 경로 기본값
    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        proc_dir = Path.cwd() / "data" / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        out_path = proc_dir / (in_path.stem + "_5class.xlsx")

    print(f"[INFO] input : {in_path}")
    print(f"[INFO] output: {out_path}")
    print(f"[INFO] model : {args.model}")

    # 엑셀 로드
    df = pd.read_excel(in_path, sheet_name=args.sheet) if args.sheet else pd.read_excel(in_path)

    # 텍스트 컬럼 자동감지
    text_col = args.text_col
    if not text_col:
        cands = ["본문", "내용", "text", "body", "article", "news", "title", "제목"]
        for c in df.columns:
            if any(k in str(c).lower() for k in cands) or str(c) in ["본문", "내용", "제목"]:
                text_col = c
                break
    if not text_col:
        raise SystemExit("[ERR] 텍스트 컬럼을 찾지 못했습니다. --text-col 로 지정하세요.")

    # 모델/토크나이저 로드
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    texts = df[text_col].astype(str).fillna("").tolist()
    n = len(texts)

    pos_list, neg_list, neu_list = [], [], []

    # 배치 추론
    bs = args.batch_size
    for i in tqdm(range(0, n, bs), desc="[Infer]"):
        batch = texts[i:i+bs]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            # 가정: 3-class (neg, neu, pos) 순서일 가능성 높음 → softmax로 확률화
            logits = out.logits.detach().cpu().numpy()
            probs = np.apply_along_axis(softmax, 1, logits)

        # 클래스 순서 추정: label mapping이 없는 모델도 있어 neg/neu/pos 추정 필요
        # 가장 일반 케이스: label 0=negative, 1=neutral, 2=positive
        # 안전을 위해 양 끝단(0,2)을 neg/pos로 두고 중앙을 neu로 처리
        neg = probs[:, 0]
        neu = probs[:, 1] if probs.shape[1] > 2 else 1.0 - (probs[:,0] + probs[:, -1])
        pos = probs[:, -1]

        pos_list.extend(pos.tolist())
        neg_list.extend(neg.tolist())
        neu_list.extend(neu.tolist())

    # 스코어/라벨
    s_vals = (np.array(pos_list) - np.array(neg_list)).tolist()
    df["positive"] = pos_list
    df["neutral"] = neu_list
    df["negative"] = neg_list
    df["s_value"] = s_vals
    df["sent_5class"] = df["s_value"].apply(label_5class)

    # 저장
    df.to_excel(out_path, index=False)
    print(f"[OK] 저장 완료 → {out_path}")

if __name__ == "__main__":
    main()
