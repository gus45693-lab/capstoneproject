#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
auto_tune_thresholds.py
- s = P(pos) - P(neg) 연속 점수에 대해 t1/t2를 격자 탐색
- 균형(엔트로피↑), 경계 샘플↓, 중립 비율 목표에 가깝게 하는 점수 최대화
- 선택한 (t1,t2)를 configs/thresholds.yaml에 source별로 반영 + 재라벨 파일 저장
"""
import argparse, os, sys, math, shutil
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[ERR] PyYAML이 필요합니다: pip install pyyaml", file=sys.stderr); sys.exit(1)

LABELS_5 = ["강한부정","부정","중립","긍정","강한긍정"]

def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)

def detect_s_value(df: pd.DataFrame) -> pd.Series:
    # 가장 흔한 컬럼 우선
    for cand in ["s_value_model", "s_value", "s_value_3"]:
        if cand in df.columns:
            return df[cand].astype(float)
    # 없으면 p_pos/p_neg로 계산
    if {"p_pos","p_neg"}.issubset(df.columns):
        return (df["p_pos"].astype(float) - df["p_neg"].astype(float))
    raise SystemExit("[ERR] s_value(또는 p_pos/p_neg) 컬럼을 찾을 수 없습니다.")

def apply_5class(s: np.ndarray, t1: float, t2: float) -> np.ndarray:
    out = np.empty(len(s), dtype=object)
    out[s <= -t2] = "강한부정"
    mask = (s > -t2) & (s <= -t1)
    out[mask] = "부정"
    mask = (s > -t1) & (s < t1)
    out[mask] = "중립"
    mask = (s >= t1) & (s < t2)
    out[mask] = "긍정"
    out[s >= t2] = "강한긍정"
    return out

def entropy_norm(counts):
    probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    eps = 1e-12
    ent = -np.sum([p*math.log(p+eps) for p in probs if p > 0])
    # 5클래스 최대 엔트로피는 ln(5)
    return ent / math.log(5)

def boundary_ratio(s: np.ndarray, t1: float, t2: float, eps: float = 0.02) -> float:
    # | |s|-t1 | < eps  또는  | |s|-t2 | < eps 인 비율 (경계 근처 샘플)
    a = np.abs(np.abs(s) - t1) < eps
    b = np.abs(np.abs(s) - t2) < eps
    return float(np.mean(a | b))

def score_combo(counts, s, t1, t2, neutral_target=0.6):
    ent = entropy_norm(counts)
    bnd = boundary_ratio(s, t1, t2, eps=0.02)
    total = counts.sum()
    neu_ratio = counts[2] / total if total else 0.0
    # 가중합 점수 (필요시 조정 가능)
    #  - 균형(엔트로피) 높을수록 +
    #  - 경계 근처 샘플 비율 낮을수록 +
    #  - 중립 비율이 목표(neutral_target)에 가까울수록 +
    score = (0.6 * ent) - (0.2 * bnd) - (0.2 * abs(neu_ratio - neutral_target))
    return float(score), ent, bnd, neu_ratio

def load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--source", required=True, choices=["news","community"])
    ap.add_argument("--config", required=True, help="configs/thresholds.yaml")
    ap.add_argument("--out", required=True, help="재라벨 결과 저장 파일(.xlsx 권장)")
    args = ap.parse_args()

    df = read_table(args.input)
    s = detect_s_value(df).to_numpy()

    # 소스별 중립 목표(경향만 반영): 뉴스는 중립 ↑, 커뮤니티는 약간 낮게
    neutral_target = 0.6 if args.source == "news" else 0.45

    # 탐색 격자 (필요하면 범위/간격 조절)
    grid_t1 = np.round(np.arange(0.25, 0.51, 0.01), 2)
    grid_t2 = np.round(np.arange(0.70, 0.96, 0.02), 2)

    best = {"score": -1e9}
    for t1 in grid_t1:
        for t2 in grid_t2:
            if t2 <= t1:  # t2는 항상 t1보다 커야 함
                continue
            labs = apply_5class(s, t1, t2)
            # 클래스 카운트
            counts = np.array([(labs == lab).sum() for lab in LABELS_5], dtype=float)
            sc, ent, bnd, neu = score_combo(counts, s, t1, t2, neutral_target)
            if sc > best["score"]:
                best.update(dict(score=sc, t1=float(t1), t2=float(t2),
                                 entropy=float(ent), boundary=float(bnd), neutral=float(neu),
                                 counts=counts))

    # 결과 출력
    ratios = (best["counts"] / best["counts"].sum()).tolist() if best["counts"].sum() > 0 else [0]*5
    print(f"[BEST] score={best['score']:.4f}, t1={best['t1']:.3f}, t2={best['t2']:.3f}")
    print(f"       entropy_norm={best['entropy']:.3f}, boundary_ratio={best['boundary']:.3f}, neutral_ratio={best['neutral']:.3f}")
    print(f"       class_ratio={ratios}  # [강부정,부정,중립,긍정,강긍정]")

    # YAML 업데이트 (+ 백업)
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    if cfg_path.exists():
        bak = cfg_path.with_suffix(".bak.yaml")
        shutil.copy2(cfg_path, bak)
        print(f"[OK] YAML 백업 생성 → {bak.name}")
    cfg.setdefault(args.source, {})
    cfg[args.source]["t1"] = float(best["t1"])
    cfg[args.source]["t2"] = float(best["t2"])
    save_yaml(cfg_path, cfg)
    print(f"[OK] YAML 업데이트 완료 → {cfg_path}")

    # 재라벨링 및 저장
    df = df.copy()
    df["sent_5class"] = apply_5class(s, best["t1"], best["t2"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".xlsx":
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 재라벨링 저장 → {out_path}")

if __name__ == "__main__":
    main()
