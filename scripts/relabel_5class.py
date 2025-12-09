#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys, re
from pathlib import Path
import pandas as pd

try:
    import yaml
except ImportError:
    print("[ERR] PyYAML이 필요합니다: pip install pyyaml", file=sys.stderr); sys.exit(1)

LABELS_5 = ["강한부정","부정","중립","긍정","강한긍정"]

def read_table(path: str) -> pd.DataFrame:
    return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)

def detect_s_value(df: pd.DataFrame) -> pd.Series:
    for cand in ["s_value_model","s_value","s_value_3"]:
        if cand in df.columns:
            return df[cand].astype(float)
    if {"p_pos","p_neg"}.issubset(df.columns):
        return (df["p_pos"].astype(float) - df["p_neg"].astype(float))
    raise SystemExit("[ERR] s_value(또는 p_pos/p_neg) 컬럼이 없습니다.")

def apply_5class(s, t1, t2):
    import numpy as np
    s = s.to_numpy() if hasattr(s, "to_numpy") else s
    out = np.empty(len(s), dtype=object)
    out[s <= -t2] = "강한부정"
    mask = (s > -t2) & (s <= -t1); out[mask] = "부정"
    mask = (s > -t1) & (s <  t1); out[mask] = "중립"
    mask = (s >=  t1) & (s <  t2); out[mask] = "긍정"
    out[s >=  t2] = "강한긍정"
    return out

def load_thresholds(cfg_path: Path, source: str):
    if not cfg_path.exists():
        raise SystemExit(f"[ERR] YAML 없음: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if source not in cfg or "t1" not in cfg[source] or "t2" not in cfg[source]:
        raise SystemExit(f"[ERR] YAML에 {source}.t1/t2가 없습니다. 파일:{cfg_path}")
    t1 = float(cfg[source]["t1"]); t2 = float(cfg[source]["t2"])
    if not (0.0 <= t1 < t2 <= 1.0):
        raise SystemExit(f"[ERR] 임계값 범위 오류: t1={t1}, t2={t2}")
    print(f"[CFG] from={cfg_path}  source={source}  t1={t1:.2f}  t2={t2:.2f}")
    return t1, t2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source", required=True, choices=["news","community"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--text-col", default=None)  # 로그용
    ap.add_argument("--five-col", default="sent_5class")
    args = ap.parse_args()

    df = read_table(args.input)
    s = detect_s_value(df)
    t1, t2 = load_thresholds(Path(args.config), args.source)

    out = df.copy()
    out[args.five_col] = apply_5class(s, t1, t2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower()==".xlsx":
        out.to_excel(out_path, index=False)
    else:
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 저장: {out_path} (source={args.source}, t1={t1:.2f}, t2={t2:.2f})")

if __name__ == "__main__":
    main()
