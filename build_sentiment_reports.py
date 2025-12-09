import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams["font.family"] = "AppleGothic"; matplotlib.rcParams["axes.unicode_minus"] = False


plt.rcParams["axes.unicode_minus"] = False  # minus 표시 깨짐 방지

# ---- 유틸: 컬럼 자동탐지 ----
def detect_column(candidates, columns):
    for cand in candidates:
        for col in columns:
            if cand in str(col).lower():
                return col
    return None

def ensure_outdirs(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "charts").mkdir(parents=True, exist_ok=True)

# ---- 메인 ----
def main():
    p = argparse.ArgumentParser(
        description="감성 통계/리포트 + (선택) 주가 상관/시각화 생성"
    )
    p.add_argument(
        "--sentiment",
        required=True,
        help="5단계 라벨/스코어가 포함된 엑셀 경로 (예: nvida_5class.xlsx)",
    )
    p.add_argument(
        "--price",
        default="",
        help="(선택) 주가 CSV/XLSX 경로. date/close(또는 종가/price) 컬럼을 포함하면 자동 병합",
    )
    p.add_argument(
        "--outdir",
        default="outputs_reports",
        help="결과물 출력 폴더 (기본: outputs_reports)",
    )
    args = p.parse_args()

    sent_path = Path(args.sentiment).expanduser()
    price_path = Path(args.price).expanduser() if args.price else None
    outdir = Path(args.outdir).expanduser()
    ensure_outdirs(outdir)
    charts = outdir / "charts"

    # ---------- 1) 감성 파일 읽기 ----------
    if not sent_path.exists():
        print(f"[ERR] sentiment 파일이 없습니다: {sent_path}")
        sys.exit(1)

    try:
        if sent_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(sent_path)
        else:
            df = pd.read_csv(sent_path)
    except Exception as e:
        print(f"[ERR] 감성 파일 읽기 실패: {e}")
        sys.exit(1)

    # 필수 컬럼 확인
    for col in ["sent_5class", "s_value"]:
        if col not in df.columns:
            print(f"[ERR] '{col}' 컬럼이 없습니다. 5단계 라벨/스코어 생성부터 수행하세요.")
            sys.exit(1)

    # 날짜 컬럼 자동탐지 및 정규화
    date_col = detect_column(["date", "날짜", "일자", "time", "datetime"], df.columns)
    if date_col is None:
        print("[WARN] 날짜 컬럼이 없어 인덱스 기반 가상 날짜를 생성합니다.")
        df["__date__"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(np.arange(len(df)), unit="D")
    else:
        try:
            df["__date__"] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            df["__date__"] = pd.to_datetime(
                df[date_col].astype(str).str.replace(".", "-"), errors="coerce"
            )
        # 완전히 NaT이면 가상 날짜 부여
        if df["__date__"].isna().all():
            print("[WARN] 날짜 파싱 실패 → 가상 날짜 생성")
            df["__date__"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(np.arange(len(df)), unit="D")

    # ---------- 2) 감성 분포/통계(뉴스 감성만) ----------
    cats = ["강한부정", "부정", "중립", "긍정", "강한긍정"]
    # 분포표
    dist = df["sent_5class"].value_counts().reindex(cats).fillna(0)
    dist_ratio = (dist / dist.sum() * 100).round(2)
    summary_table = pd.DataFrame(
        {"구분": cats, "건수": dist.astype(int).values, "비율(%)": dist_ratio.values}
    )
    summary_table["평균 s_value"] = df["s_value"].mean()

    # 저장
    summary_csv = outdir / "sentiment_distribution_summary.csv"
    summary_xlsx = outdir / "sentiment_distribution_summary.xlsx"
    summary_table.to_csv(summary_csv, index=False)
    summary_table.to_excel(summary_xlsx, index=False)
    print(f"[OK] 감성 통계 저장: {summary_csv.name}, {summary_xlsx.name}")

    # 시각화(분포 막대)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(cats, dist_ratio.values, alpha=0.85)
    ax.set_title("5단계 감성 분포(뉴스/커뮤니티 집합)", fontsize=13, fontweight="bold")
    ax.set_ylabel("비율(%)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(charts / "sentiment_distribution_single.png", dpi=200)
    plt.close(fig)

    # ---------- 3) 일자별 감성 집계 ----------
    for c in cats:
        df[f"flag_{c}"] = (df["sent_5class"] == c).astype(int)
    daily = df.groupby(df["__date__"].dt.date).agg(
        mean_s=("s_value", "mean"),
        **{f"ratio_{c}": (f"flag_{c}", "mean") for c in cats},
    )
    daily_path = outdir / "daily_sentiment_only.csv"
    daily.to_csv(daily_path)
    print(f"[OK] 일자별 감성 집계 저장: {daily_path.name}")

    # 시각화(일자별 mean_s)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(daily.index, daily["mean_s"])
    ax2.set_title("일자별 평균 s_value")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("mean s_value")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    fig2.tight_layout()
    fig2.savefig(charts / "timeseries_mean_s_only.png", dpi=200)
    plt.close(fig2)

    # ---------- 4) (선택) 주가 데이터 병합 & 상관/시각화 ----------
    did_price = False
    if price_path and price_path.exists():
        # 가격 파일 읽기
        try:
            if price_path.suffix.lower() in [".xlsx", ".xls"]:
                px = pd.read_excel(price_path)
            else:
                # CSV 인코딩 후보
                tried = False
                for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
                    try:
                        px = pd.read_csv(price_path, encoding=enc)
                        tried = True
                        break
                    except Exception:
                        continue
                if not tried:
                    px = pd.read_csv(price_path)  # 마지막 시도
        except Exception as e:
            print(f"[WARN] 주가 파일 읽기 실패: {e}")
            px = None

        if px is not None:
            px_date = detect_column(["date", "날짜", "일자", "time", "datetime"], px.columns)
            px_price = detect_column(["close", "종가", "price", "adjclose", "adj_close"], px.columns)
            if px_date and px_price:
                # 날짜 정규화
                px["__date__"] = pd.to_datetime(px[px_date], errors="coerce").dt.date
                price_daily = px.groupby("__date__")[px_price].mean().to_frame("price")
                aligned = daily.join(price_daily, how="inner")
                aligned["return_1d"] = aligned["price"].pct_change().shift(-1)

                # 저장
                aligned_path = outdir / "aligned_daily.csv"
                aligned.to_csv(aligned_path)
                print(f"[OK] 감성+주가 정렬 저장: {aligned_path.name}")

                # 상관
                valid = aligned[["mean_s", "return_1d"]].dropna()
                if len(valid) >= 3:
                    corr_pearson = valid.corr(method="pearson").iloc[0, 1]
                    corr_spearman = valid.corr(method="spearman").iloc[0, 1]
                else:
                    corr_pearson = np.nan
                    corr_spearman = np.nan
                corr_df = pd.DataFrame(
                    {
                        "metric": ["pearson", "spearman"],
                        "corr_with_nextday_return": [corr_pearson, corr_spearman],
                    }
                )
                corr_path = outdir / "correlations.csv"
                corr_df.to_csv(corr_path, index=False)
                print(f"[OK] 상관 저장: {corr_path.name}")

                # 시각화(다음날 수익률)
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.plot(aligned.index, aligned["return_1d"])
                ax3.set_title("일자별 다음날 수익률")
                ax3.set_xlabel("Date")
                ax3.set_ylabel("next-day return")
                ax3.grid(axis="y", linestyle="--", alpha=0.5)
                fig3.tight_layout()
                fig3.savefig(charts / "timeseries_nextday_return.png", dpi=200)
                plt.close(fig3)

                # 산점도
                fig4, ax4 = plt.subplots(figsize=(6, 5))
                ax4.scatter(valid["mean_s"], valid["return_1d"], alpha=0.7)
                ax4.set_title(f"산점도 (Pearson={corr_pearson:.3f}, Spearman={corr_spearman:.3f})")
                ax4.set_xlabel("mean s_value")
                ax4.set_ylabel("next-day return")
                ax4.grid(True, linestyle="--", alpha=0.5)
                fig4.tight_layout()
                fig4.savefig(charts / "scatter_mean_s_vs_nextday_return.png", dpi=200)
                plt.close(fig4)

                did_price = True
            else:
                print("[WARN] 주가 파일에서 날짜/가격 컬럼을 찾지 못했습니다. 감성 통계만 생성합니다.")
        else:
            print("[WARN] 주가 파일을 읽지 못했습니다. 감성 통계만 생성합니다.")
    else:
        if price_path:
            print(f"[WARN] 주가 파일이 없습니다: {price_path}")

    # ---------- 5) 리드미(요약) ----------
    md = []
    md.append("# 보고서 요약 (자동 생성)\n")
    md.append(f"- 입력 감성 파일: `{sent_path}`")
    if price_path:
        md.append(f"- 입력 주가 파일: `{price_path}` (존재: {price_path.exists()})")
    md.append(f"- 출력 폴더: `{outdir}`\n")
    md.append("## 감성 통계 요약")
    md.append(f"- 총 샘플 수: {len(df)}")
    md.append(f"- 평균 s_value: {df['s_value'].mean():.6f}")
    md.append("- 등급별 비율(%)")
    for c, r in zip(cats, dist_ratio.values):
        md.append(f"  - {c}: {r:.2f}%")
    if did_price:
        md.append("\n## 주가 상관/시각화")
        md.append("- `aligned_daily.csv`, `correlations.csv` 생성")
        md.append("- `charts/`에 시계열/산점도 이미지 저장")
    else:
        md.append("\n## 주가 파일이 없거나 인식 실패 → 감성 통계만 생성됨")
    (outdir / "README_build_report.md").write_text("\n".join(md), encoding="utf-8")
    print("[OK] 보고서 요약: README_build_report.md")

if __name__ == "__main__":
    main()
