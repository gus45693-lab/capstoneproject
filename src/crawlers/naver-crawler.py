import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import time
import random

# 1. 검색어 설정
query = 'SK하이닉스'
query_encoded = requests.utils.quote(query)

# 2. 날짜 설정 (최근 6개월)
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

# 3. 날짜 포맷
end_date_str = end_date.strftime('%Y.%m.%d')
start_date_str = start_date.strftime('%Y.%m.%d')

# 4. 결과 저장용
results = []

# 5. 기본 URL
base_url = f"https://search.naver.com/search.naver?where=news&query={query_encoded}&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds={start_date_str}&de={end_date_str}"
print(f"[INFO] 크롤링 대상 URL 예시:\n{base_url}")

# 6. 세션 설정 + 헤더 강화 (봇 차단 우회)
session = requests.Session()
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.naver.com/",
}

# 7. 크롤링 (페이지당 10개씩, 최대 기사 200~300개 권장)
max_pages = 10  # 너무 높이면 위험 (1000→100으로 낮춤)

for page in tqdm(range(1, max_pages * 10, 10)):
    url = f"{base_url}&start={page}"
    res = session.get(url, headers=headers)
    if res.status_code != 200:
        print(f"[ERROR] 페이지 {page} 요청 실패 (코드 {res.status_code})")
        break

    soup = BeautifulSoup(res.text, 'html.parser')

    # 최신 네이버 뉴스 제목 + 링크 셀렉터
    news_links = soup.select('a.lu8Lfh20c9DvvP05mqBf.tym_MoKIfC84Aqvg9SKg')

    if not news_links:
        print(f"[INFO] 페이지 {page} 더 이상 뉴스 없음 (크롤링 종료)")
        break

    for tag in news_links:
        title_span = tag.find('span')
        title = title_span.text.strip() if title_span else '제목없음'

        # 날짜 찾기 (정확한 최신 셀렉터로)
        parent_div = tag.find_parent('div')
        pub_date = '날짜없음'
        if parent_div:
            pub_date_tag = parent_div.select_one('div.sds-comps-profile-info > span:nth-child(3) span')
            pub_date = pub_date_tag.text.strip() if pub_date_tag else '날짜없음'

        results.append({
            'title': title,
            'pub_date': pub_date
        })

    # 요청 간격 랜덤화 (사람처럼)
    time.sleep(random.uniform(0.7, 2.0))

# 8. 저장 + 확인
df = pd.DataFrame(results)
print(f"\n[INFO] 총 수집된 뉴스 기사 수: {len(df)}개")
print(df.head())

# CSV 저장 (title, pub_date만 저장)
df.to_csv('naver_news_sk_hynix.csv', index=False, encoding='utf-8-sig')
