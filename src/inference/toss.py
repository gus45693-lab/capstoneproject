import time
import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# 크롬 드라이버 설정
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 타겟 URL
url = 'https://tossinvest.com/stocks/US19990122001/community'
driver.get(url)
time.sleep(3)

# 스크롤을 내려서 게시글을 계속 로딩
SCROLL_PAUSE_TIME = 2
MAX_POSTS = 1000




last_height = driver.execute_script("return document.body.scrollHeight")
post_texts = set()
today = datetime.datetime.now().date()

while len(post_texts) < MAX_POSTS:
    # 스크롤
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)

    # 게시글(article) 단위로 수집
    articles = driver.find_elements(By.CSS_SELECTOR, 'ul > div > div > article ')
    for article in articles:
        try:
            # 시간 정보 가져오기
            time_span = article.find_element(By.CSS_SELECTOR, 'header time span')
            time_text = time_span.text.strip()

            # '방금 전', '분 전', '시간 전', '오늘' 또는 오늘 날짜 포맷이면 수집
            if any(keyword in time_text for keyword in ['방금', '분 전', '시간 전', '오늘']):
                content = article.text.strip()
                if len(content) > 5:
                    post_texts.add(content)



        except:
            continue  # 예외 발생 시 해당 게시글 무시

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

print(f"오늘 작성된 게시글 {len(post_texts)}개 수집 완료")

# CSV 저장
df = pd.DataFrame({'content': list(post_texts)})
df.to_csv('toss_today.csv', index=False, encoding='utf-8')
print("CSV 저장 완료: toss_today.csv")

driver.quit()
