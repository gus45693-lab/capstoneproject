# 기본적인 라이브러리 불러오기
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# 데이터 불러오기
data = pd.read_csv('20250504_13_46_40_sk-hynix_preprocessed.csv')  # 파일 경로는 환경에 맞게 수정

# KR-FinBert 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")

# 제목 감성분석 함수 정의
def title_sent(text):
    try:
        inputs = tokenizer(str(text), return_tensors='pt', truncation=True, max_length=128)
        output = model(**inputs)
        logits = output.logits.detach().cpu().numpy()[0]
        probs = nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        positive, negative, neutral = float(probs[2]), float(probs[0]), float(probs[1])
        return positive, negative, neutral
    except Exception as e:
        return 0.0, 0.0, 0.0

# 감성분석 적용
data['title_sent_analysis'] = data['제목'].apply(title_sent)

# 결과 컬럼 분리
data['positive'] = data['title_sent_analysis'].apply(lambda x: x[0])
data['negative'] = data['title_sent_analysis'].apply(lambda x: x[1])
data['neutral'] = data['title_sent_analysis'].apply(lambda x: x[2])

# 감성지수(긍정+중립) 산출
data['sent_score'] = data['positive'] + data['neutral']

# 결과 저장
data.to_csv('20250504_12_57_51_sk-hynix_sentiment.csv', index=False, encoding='utf-8-sig')
