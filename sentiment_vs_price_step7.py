import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False  # minus sign rendering

def detect_column(candidates, columns):
    """열 이름 자동감지: candidates 리스트(소문자 키워드) 중 하나라도 포함되면 해당 열 반환"""
    for cand in candidates:
        for col in columns:
            if cand in str(col).lower():
                return col
    return None

def main():
    parser = argparse.ArgumentParser(description='Step7: sentiment vs price correlation')
    parser.add_argument(
        '--input',
        type=str,
        default='/mnt/data/processed/nvida_5class.xlsx',
        help='Input file path (default: /mnt/data/processed/nvida_5class.xlsx)'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='/mnt/data/outputs_reports',
        help='Output directory (default: /mnt/data/outputs_reports)'
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser()
    out_dir = Path(args.outdir).expanduser()
    charts = out_dir / "charts"

    print(f"[INFO] Input path : {in_path}")
    print(f"[INFO] Output dir : {out_dir}")

    if not in_path.exists():
        print("[ERR] 입력 파일을 찾을 수 없습니다.")
        print("      → --input 경로를 확인하세요.")
        print("      예) --input /absolute/path/to/yourfile.xlsx")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)

    # 파일 읽기
    try:
        # 확장자에 따라 읽기 분기해도 되지만 여기선 xlsx 기준
        df = pd.read_excel(in_path)
    except Exception as e:
        print(f"[ERR] 파일 읽기 실패: {e}")
        sys.exit(1)

    # 필수 감성 컬럼 확인
    required_cols = ['sent_5class', 's_value']
    for rc in required_cols:
        if rc not in df.columns:
            print(f"[ERR] 필요한 컬럼({rc})이 없습니다. Step4에서 5단계 감성 컬럼 생성이 되었는지 확인하세요.")
            sys.exit(1)

    # 날짜/가격 컬럼 자동 탐지
    date_col = detect_column(['date', '날짜', '일자', 'time', 'datetime'], df.columns)
    price_col = detect_column(['close', '종가', 'price', 'adjclose', 'adj_close'], df.columns)

    print(f"[INFO] Detected date column : {date_col}")
    print(f"[INFO] Detected price column: {price_col}")

    # robust 날짜 파싱
    if date_col is not None:
        try:
            df['__date__'] = pd.to_datetime(df[date_col], errors='coerce')
        except Exception:
            df['__date__'] = pd.to_datetime(df[date_col].astype(str).str.replace('.', '-'), errors='coerce')
        # 여전히 NaT가 있으면 앞 10자 슬라이스로 재시도
        mask = df['__date__'].isna()
        if mask.any():
            df.loc[mask, '__date__'] = pd.to_datetime(
                df.loc[mask, date_col].astype(str).str.slice(0, 10),
                errors='coerce'
            )
        if df['__date__'].isna().all():
            print("[WARN] 날짜 파싱 실패 → 인덱스 기반 가상 날짜 생성")
            df['__date__'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(np.arange(len(df)), unit='D')
    else:
        print("[WARN] 날짜 컬럼을 찾지 못함 → 인덱스 기반 가상 날짜 생성")
        df['__date__'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(np.arange(len(df)), unit='D')

    # 등급 one-hot 플래그
    cats = ['강한부정', '부정', '중립', '긍정', '강한긍정']
    for c in cats:
        df[f'flag_{c}'] = (df['sent_5class'] == c).astype(int)

    # 일자 집계: 평균 s와 등급 비율
    daily = df.groupby(df['__date__'].dt.date).agg(
        mean_s=('s_value', 'mean'),
        **{f'ratio_{c}': (f'flag_{c}', 'mean') for c in cats}
    )

    # 가격/수익률 처리
    if price_col is not None:
        price_daily = df.groupby(df['__date__'].dt.date)[price_col].mean().to_frame('price')
        aligned = daily.join(price_daily, how='inner')
        aligned['return_1d'] = aligned['price'].pct_change().shift(-1)  # 다음날 수익률

        # 저장
        aligned_path = out_dir / 'aligned_daily.csv'
        aligned.to_csv(aligned_path)
        print(f"[OK] 저장: {aligned_path}")

        # 상관계수 계산
        valid = aligned[['mean_s', 'return_1d']].dropna()
        if len(valid) >= 3:
            corr_pearson = valid.corr(method='pearson').iloc[0, 1]
            corr_spearman = valid.corr(method='spearman').iloc[0, 1]
        else:
            corr_pearson = np.nan
            corr_spearman = np.nan

        corr_df = pd.DataFrame({
            'metric': ['pearson', 'spearman'],
            'corr_with_nextday_return': [corr_pearson, corr_spearman]
        })
        corr_path = out_dir / 'correlations.csv'
        corr_df.to_csv(corr_path, index=False)
        print(f"[OK] 저장: {corr_path}")

        # 시각화 1: mean_s 시계열
        fig1, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(aligned.index, aligned['mean_s'])
        ax1.set_title('일자별 평균 s_value')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('mean s_value')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        fig1.tight_layout()
        fig1.savefig(charts / 'timeseries_mean_s.png', dpi=200)
        plt.close(fig1)

        # 시각화 2: 다음날 수익률 시계열
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.plot(aligned.index, aligned['return_1d'])
        ax2.set_title('일자별 다음날 수익률')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('next-day return')
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        fig2.tight_layout()
        fig2.savefig(charts / 'timeseries_nextday_return.png', dpi=200)
        plt.close(fig2)

        # 시각화 3: 산점도
        sc = valid
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.scatter(sc['mean_s'], sc['return_1d'], alpha=0.7)
        ax3.set_title(f'산점도 (Pearson={corr_pearson:.3f}, Spearman={corr_spearman:.3f})')
        ax3.set_xlabel('mean s_value')
        ax3.set_ylabel('next-day return')
        ax3.grid(True, linestyle='--', alpha=0.5)
        fig3.tight_layout()
        fig3.savefig(charts / 'scatter_mean_s_vs_nextday_return.png', dpi=200)
        plt.close(fig3)

        note = f"가격컬럼[{price_col}] 감지됨 — 상관계수(Pearson={corr_pearson:.4f}, Spearman={corr_spearman:.4f})"
    else:
        # 가격이 없을 때: 감성 집계만
        daily_only_path = out_dir / 'daily_sentiment_only.csv'
        daily.to_csv(daily_only_path)
        print(f"[OK] 저장: {daily_only_path}")
        note = "가격 컬럼이 없어 감성 집계만 생성"

    # 리포트 메모
    md_path = out_dir / 'README_step7.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Step7 — 감성 vs 가격 상관분석 결과\n')
        f.write(f'- 입력 파일: {in_path}\n')
        f.write(f'- 날짜 컬럼: {date_col}\n')
        f.write(f'- 가격 컬럼: {price_col}\n')
        f.write(f'- 메모: {note}\n')
    print(f"[OK] 저장: {md_path}")

if __name__ == '__main__':
    main()
