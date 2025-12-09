import pandas as pd
import re

# 1. 데이터 불러오기
df = pd.read_csv('sk_hynix_naver_news_6months.csv')

# 2. 결측값 및 중복 제거
df = df.drop_duplicates(subset=['제목', '날짜'])
df = df.dropna(subset=['제목', '날짜'])

# 3. 텍스트 전처리 함수
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'<.*?>', '', text)  # HTML 태그 제거
    text = re.sub(r'[^\w\s가-힣.,!?~\'\"%()\[\]-]', '', text)  # 한글, 영문, 숫자, 기본 구두점만 남김
    text = re.sub(r'\s+', ' ', text)  # 연속 공백 하나로
    text = re.sub(r'([!?~])\1+', r'\1', text)  # 연속 기호 정리
    text = text.strip(' "\'')
    return text



# 4. 제목 전처리 적용 및 길이 제한(300자)
df['제목'] = df['제목'].apply(clean_text)
df['제목'] = df['제목'].apply(lambda x: x[:300])

# 5. 날짜 포맷 통일 (예: '2025.05.04 13:45' → '2025-05-04')
def clean_date(date_str):
    # 날짜만 추출 (시간 제거)
    date = str(date_str).split()[0].replace('.', '-')
    return date

df['날짜'] = df['날짜'].apply(clean_date)

# 6. 전처리 후 공백/결측값 제거
df = df[df['제목'].str.strip() != '']
df = df.dropna(subset=['제목', '날짜'])

# 7. 저장
df.to_csv('20250504_13_46_40_sk-hynix_preprocessed.csv', index=False, encoding='utf-8-sig')
