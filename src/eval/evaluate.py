import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib as mpl
import matplotlib.dates as mdates
from datetime import timedelta

# 1. 한글 폰트 설정 (윈도우: 'Malgun Gothic')
mpl.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 2. 감성 분석 결과 파일 불러오기
sentiment_df = pd.read_csv('20250504_12_57_51_sk-hynix_sentiment.csv')
sentiment_df['날짜'] = pd.to_datetime(sentiment_df['날짜'])

# 3. 최근 6개월 데이터만 추출
end_date = sentiment_df['날짜'].max()
start_date = end_date - pd.DateOffset(months=6)
sentiment_df = sentiment_df[sentiment_df['날짜'].between(start_date, end_date)]

# 4. 날짜별 평균 감성 점수 계산
sentiment_daily = sentiment_df.groupby('날짜')['sent_score'].mean().reset_index()

# 5. 주가 데이터 자동 다운로드 (SK하이닉스: 000660.KS)
ticker = "000660.KS"
stock_start = start_date.strftime("%Y-%m-%d")
stock_end = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
stock_data = yf.download(ticker, start=stock_start, end=stock_end)
stock_df = stock_data[['Close']].reset_index()
stock_df.rename(columns={'Close': '주가', 'Date': '날짜'}, inplace=True)
stock_df['날짜'] = pd.to_datetime(stock_df['날짜'])

# 6. 시각화
fig, ax1 = plt.subplots(figsize=(12, 7))

# 감성 점수 (왼쪽 Y축)
ax1.set_xlabel('날짜', fontsize=12)
ax1.set_ylabel('감성 점수', color='tab:blue', fontsize=12)
ax1.plot(sentiment_daily['날짜'], sentiment_daily['sent_score'],
         color='tab:blue', marker='o', linestyle='--', linewidth=2, markersize=8, label='감성 점수')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, linestyle='--', alpha=0.7)

# 주가 (오른쪽 Y축)
ax2 = ax1.twinx()
ax2.set_ylabel('주가 (KRW)', color='tab:red', fontsize=12)
ax2.plot(stock_df['날짜'], stock_df['주가'],
         color='tab:red', marker='s', linestyle='-', linewidth=2, markersize=8, label='주가')
ax2.tick_params(axis='y', labelcolor='tab:red')

# x축: 6개월 범위, 월 단위 major tick, 연-월 표시
ax1.set_xlim([start_date, end_date])
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

fig.autofmt_xdate()  # x축 날짜 레이블 자동 회전

# 공통 설정
plt.title("SK하이닉스 뉴스 감성 분석 vs 주가 추이\n(최근 6개월)", fontsize=14, pad=20)
fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
plt.tight_layout()
plt.show()
