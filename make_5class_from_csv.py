import pandas as pd
from pathlib import Path

# 원본 CSV 경로 (네 맥 기준 경로로 수정해줘)
input_path = Path("/Users/yuchanghyeon/PycharmProjects/capstoneproject/data/raw/nvida.csv")

# 출력 폴더
output_path = Path("/Users/yuchanghyeon/PycharmProjects/capstoneproject/data/processed/nvida_5class.xlsx")

def label_5class(s, t1=0.55, t2=0.75):
    if s <= -t2:
        return "강한부정"
    elif s <= -t1:
        return "부정"
    elif -t1 < s < t1:
        return "중립"
    elif s < t2:
        return "긍정"
    else:
        return "강한긍정"

# CSV 읽기
df = pd.read_csv(input_path)

# positive, negative 컬럼 자동 감지
pos_col = next((c for c in df.columns if "pos" in c.lower()), None)
neg_col = next((c for c in df.columns if "neg" in c.lower()), None)

if pos_col and neg_col:
    df["s_value"] = df[pos_col] - df[neg_col]
    df["sent_5class"] = df["s_value"].apply(label_5class)
    df.to_excel(output_path, index=False)
    print(f"[OK] 변환 완료: {output_path}")
else:
    print("[ERR] positive / negative 컬럼을 찾지 못했습니다. CSV 컬럼 이름을 확인하세요.")
