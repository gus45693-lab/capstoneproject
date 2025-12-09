from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import UnexpectedAlertPresentException
from selenium.webdriver.common.alert import Alert
from datetime import datetime
import pandas as pd
import time
import os
import random


class CrawlWithCode:
    def __init__(self, code, BASE_URL='https://finance.naver.com/item/board.nhn?code='):
        self.code = code
        self.BASE_URL = BASE_URL
        self.data = []  # 제목+날짜 저장용 리스트

        chrome_options = Options('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        # User-Agent 랜덤 설정
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        ]
        chrome_options.add_argument(f'user-agent={random.choice(user_agents)}')

        self.driver = webdriver.Chrome(
            service=Service('C:/Users/funck/PycharmProjects/PythonProject/chromedriver.exe'),
            options=chrome_options
        )

    def page_saver(self, page):
        try:
            url = self.BASE_URL + self.code + "&page=" + str(page)
            self.driver.get(url)

            # 랜덤 지연 추가 (1~3초)
            time.sleep(random.uniform(1, 3))

            # 페이지 존재 여부 확인
            if "존재하지 않는 페이지" in self.driver.page_source:
                return False

            rows = self.driver.find_elements(By.CSS_SELECTOR,
                                             '#content > div.section.inner_sub > table.type2 > tbody > tr')

            for row in rows:
                try:
                    title_elem = row.find_element(By.CSS_SELECTOR, 'td.title > a')
                    date_elem = row.find_element(By.CSS_SELECTOR, 'td:nth-child(1) > span')
                    title = title_elem.get_attribute('title')
                    date = date_elem.text
                    self.data.append((title, date))
                except:
                    continue

            return True

        except UnexpectedAlertPresentException:
            print("경고창 발생! 처리 진행...")
            alert = Alert(self.driver)
            print(f"경고 내용: {alert.text}")
            alert.accept()
            time.sleep(2)
            return False

    def tocsv(self, filename, PATH):
        print(f"수집된 글 수: {len(self.data)}")

        filename = datetime.today().strftime("%Y%m%d_%H_%M_%S") + "_" + filename
        os.makedirs(PATH, exist_ok=True)
        full_path = os.path.join(PATH, f"{filename}.csv")

        df = pd.DataFrame(self.data, columns=['제목', '날짜'])
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f"CSV 저장 완료: {full_path}")

    def close(self):
        self.driver.quit()


# 사용 예시
if __name__ == "__main__":
    crawler = CrawlWithCode(code="000660")  # SK하이닉스
    max_page = 6  # 안전한 최대 페이지 수 설정

    for i in range(1, max_page + 1):
        success = crawler.page_saver(i)
        print(f"{i} 페이지 크롤링 완료")

        if not success:
            print(f"{i} 페이지에서 오류 발생. 크롤링 중단")
            break

        # 매 10페이지마다 5초 추가 휴식
        if i % 10 == 0:
            time.sleep(5)

    crawler.tocsv("sk-hynix", "C:/data/")
    crawler.close()
