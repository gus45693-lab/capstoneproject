#!/usr/bin/env bash
set -euo pipefail

# --------- 공통 설정 ----------
TEXT_COL="본문"
CFG="configs/thresholds.yaml"

say() { printf "\n\033[1;36m[%s]\033[0m %s\n" "$1" "${2:-}"; }

# thresholds.yaml 기본값 보정
if [[ ! -f "$CFG" ]]; then
  say "INIT" "create $CFG (defaults)"
  mkdir -p configs
  cat > "$CFG" <<YAML
news: {t1: 0.35, t2: 0.90}
community: {t1: 0.20, t2: 0.65}
YAML
fi

# --------- NEWS 파이프라인 ----------
say "NEWS" "start"
mkdir -p data/processed outputs_reports_news_final outputs_keywords_news outputs_analysis_news

NEWS_PROC="data/processed/news_5class.xlsx"
NEWS_FINAL="data/processed/news_5class_final.xlsx"

if [[ ! -f "$NEWS_PROC" ]]; then
  # 원본 엑셀 자동 탐색 (첫 번째 파일 사용)
  RAW_NEWS=$(ls data/raw/NewsResult_*.xlsx 2>/dev/null | head -n1 || true)
  if [[ -n "${RAW_NEWS}" ]]; then
    say "NEWS" "infer 5-class from ${RAW_NEWS}"
    python infer_news_to_5class.py \
      --input "${RAW_NEWS}" --text-col "${TEXT_COL}" \
      --model "snunlp/KR-FinBERT-SC" --out "${NEWS_PROC}"
  else
    say "NEWS" "skip infer (no raw) — using existing processed if any"
  fi
fi

if [[ -f "$NEWS_PROC" ]]; then
  say "NEWS" "apply thresholds -> ${NEWS_FINAL}"
  python scripts/relabel_5class.py \
    --input "${NEWS_PROC}" --output "${NEWS_FINAL}" \
    --source news --config "${CFG}"

  say "NEWS" "build reports"
  python build_sentiment_reports.py \
    --sentiment "${NEWS_FINAL}" --outdir "outputs_reports_news_final"

  say "NEWS" "top keywords"
  python scripts/top_keywords_by_class.py \
    --input "${NEWS_FINAL}" --text-col "${TEXT_COL}" --label-col "sent_5class" \
    --outdir "outputs_keywords_news" --topn 20 --min-count 2 --make-charts

  say "NEWS" "neutral vs length"
  python scripts/neutral_vs_length.py \
    --input "${NEWS_FINAL}" --text-col "${TEXT_COL}" --label-col "sent_5class" \
    --outdir "outputs_analysis_news"
else
  say "NEWS" "SKIP — processed news file not found (${NEWS_PROC})"
fi

# --------- KAKAO 파이프라인 ----------
say "KAKAO" "start"
mkdir -p data/processed outputs_reports_kakao_final outputs_keywords_kakao outputs_analysis_kakao

KAKAO_PROC="data/processed/kakao_5class.xlsx"
KAKAO_FINAL="data/processed/kakao_5class_final.xlsx"

if [[ ! -f "$KAKAO_PROC" ]]; then
  if [[ -f "data/raw/kakao_community.xlsx" ]]; then
    SRC="data/raw/kakao_community.xlsx"
    say "KAKAO" "infer 5-class from ${SRC}"
    python infer_news_to_5class.py \
      --input "${SRC}" --text-col "${TEXT_COL}" \
      --model "snunlp/KR-FinBERT-SC" --out "${KAKAO_PROC}"
  elif [[ -f "data/raw/kakao_community.csv" ]]; then
    say "KAKAO" "CSV 원본 감지 — 먼저 엑셀로 저장해서 사용 권장 (또는 infer 스크립트 보정)"
    exit 1
  fi
fi

if [[ -f "$KAKAO_PROC" ]]; then
  say "KAKAO" "apply thresholds -> ${KAKAO_FINAL}"
  python scripts/relabel_5class.py \
    --input "${KAKAO_PROC}" --output "${KAKAO_FINAL}" \
    --source community --config "${CFG}"

  say "KAKAO" "build reports"
  python build_sentiment_reports.py \
    --sentiment "${KAKAO_FINAL}" --outdir "outputs_reports_kakao_final"

  say "KAKAO" "top keywords"
  python scripts/top_keywords_by_class.py \
    --input "${KAKAO_FINAL}" --text-col "${TEXT_COL}" --label-col "sent_5class" \
    --outdir "outputs_keywords_kakao" --topn 20 --min-count 5 --make-charts

  say "KAKAO" "neutral vs length"
  python scripts/neutral_vs_length.py \
    --input "${KAKAO_FINAL}" --text-col "${TEXT_COL}" --label-col "sent_5class" \
    --outdir "outputs_analysis_kakao"

  # (선택) auto vs final 비교: auto 파일이 있을 때만
  if [[ -f "data/processed/kakao_5class_auto_applied.xlsx" ]]; then
    mkdir -p outputs_compare_kakao_final
    say "KAKAO" "compare auto vs final"
    python scripts/compare_kakao_versions.py \
      --auto "data/processed/kakao_5class_auto_applied.xlsx" \
      --aggr "${KAKAO_FINAL}" \
      --outdir "outputs_compare_kakao_final"
  fi
else
  say "KAKAO" "SKIP — processed kakao file not found (${KAKAO_PROC})"
fi

# --------- 뉴스 vs 카카오 길이-중립 비교 ----------
if [[ -d "outputs_analysis_news" && -d "outputs_analysis_kakao" ]]; then
  say "COMPARE" "news vs kakao length-neutral"
  mkdir -p outputs_compare_len
  python scripts/compare_neutral_length_news_vs_kakao.py \
    --news-dir "outputs_analysis_news" \
    --kakao-dir "outputs_analysis_kakao" \
    --outdir "outputs_compare_len"
fi

# --------- 패키징 ----------
say "PACKAGE" "build deliverables"

# NEWS
mkdir -p deliverables/news
cp -f "${CFG}" deliverables/news/ 2>/dev/null || true
cp -f "${NEWS_FINAL}" deliverables/news/ 2>/dev/null || true
cp -f outputs_reports_news_final/* deliverables/news/ 2>/dev/null || true
cp -Rf outputs_reports_news_final/charts deliverables/news/ 2>/dev/null || true
cp -f outputs_keywords_news/top_keywords.xlsx deliverables/news/ 2>/dev/null || true
cp -Rf outputs_keywords_news/charts_* deliverables/news/ 2>/dev/null || true
cp -f outputs_analysis_news/* deliverables/news/ 2>/dev/null || true
cp -f outputs_compare_len/*news_vs_kakao*.png deliverables/news/ 2>/dev/null || true

# KAKAO
mkdir -p deliverables/kakao
cp -f "${CFG}" deliverables/kakao/ 2>/dev/null || true
cp -f "${KAKAO_FINAL}" deliverables/kakao/ 2>/dev/null || true
cp -f outputs_reports_kakao_final/* deliverables/kakao/ 2>/dev/null || true
cp -Rf outputs_reports_kakao_final/charts deliverables/kakao/ 2>/dev/null || true
cp -f outputs_keywords_kakao/top_keywords.xlsx deliverables/kakao/ 2>/dev/null || true
cp -Rf outputs_keywords_kakao/charts_* deliverables/kakao/ 2>/dev/null || true
cp -f outputs_analysis_kakao/* deliverables/kakao/ 2>/dev/null || true
cp -f outputs_compare_kakao_final/*  deliverables/kakao/ 2>/dev/null || true
cp -f outputs_explain_kakao*/explain_samples.xlsx deliverables/kakao/ 2>/dev/null || true
cp -Rf outputs_explain_kakao*/html deliverables/kakao/ 2>/dev/null || true

# ZIPs
( cd deliverables && zip -qr news_package_$(date +%Y%m%d).zip news || true )
( cd deliverables && zip -qr kakao_package_$(date +%Y%m%d).zip kakao || true )

say "DONE" "pipeline finished."
