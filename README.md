# Capstone Sentiment Pipeline (5-class)

## 목표
- 1학기(3클래스: 부정/중립/긍정)에서 **2학기 5클래스(강부정/부정/중립/긍정/강긍정)**로 확장
- 뉴스/커뮤니티(카카오) **동일 파이프라인** 적용 + 임계값 자동탐색/수동조정 + 설명가능성(하이라이트)

---

## 환경
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

import matplotlib as mpl
mpl.rcParams["font.family"]="AppleGothic"; mpl.rcParams["axes.unicode_minus"]=False


