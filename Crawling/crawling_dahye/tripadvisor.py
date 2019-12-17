from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import pandas as pd

tripadvisor = "https://www.tripadvisor.co.kr"

pages = []

for i in range(0,1921,30):
    pages.append(
        "https://www.tripadvisor.co.kr/RestaurantSearch-g294197-c10642-oa"
        + str(i) + "-Seoul.html#EATERY_OVERVIEW_BOX")

n = open("C:/Users/dadadah_ye/Downloads/name.txt", "a", encoding="utf-8")
a = open("C:/Users/dadadah_ye/Downloads/address.txt", "a", encoding="utf-8")
r = open("C:/Users/dadadah_ye/Downloads/review.txt", "a", encoding="utf-8")
s = open("C:/Users/dadadah_ye/Downloads/score.txt", "a", encoding="utf-8")

for page in pages: # 페이지 링크
    result = urlopen(page)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')

    tags = soup.findAll('a',
                        attrs={'class':
                        'restaurants-list-ListCell__restaurantName--2aSdo'})
    
    cafe_link = []
    urls = []
    urls_keyword = []

    for tag in tags:
        cafe_link.append(tag.get('href'))

    for i in range(0,len(cafe_link)):
        cafe_link[i] = tripadvisor + cafe_link[i]
        url_sp, url_keyword_sp = cafe_link[i].split('-Reviews')
        urls.append(url_sp+'-Reviews-or')
        urls_keyword.append(url_keyword_sp)
    
    for i in range(0,len(urls)):
        url = urls[i]
        url_keyword = urls_keyword[i]

        result = urlopen(url)
        html = result.read()
        soup = BeautifulSoup(html, 'html.parser')

        tags_name = soup.findAll('h1',
                                 attrs={'class':'ui_header h1'})
        tags_location = soup.findAll('div',
                                     attrs={'class':
            'restaurants-detail-overview-cards-LocationOverviewCard__addressLink--1pLK4 restaurants-detail-overview-cards-LocationOverviewCard__detailLink--iyzJI'})
        tags_review = soup.findAll('a',
                                   attrs={'class':'pageNum last taLnk'})
        
        #리뷰
        last_num = 0
        
        for tag in tags_review:
            last_num = int(tag.text)
    
        for i in range(0,(last_num*10)+1,10):
            url_num = str(i)
            total_url = url + url_num + url_keyword
            
            driver = webdriver.Chrome("C:/Users/dadadah_ye/Documents/text-mining/crawling_dahye/chromedriver")
            driver.get(total_url)
            time.sleep(5)        
            
            try:
                driver.find_elements_by_css_selector(".ulBlueLinks")[0].click()
                time.sleep(5)
            except:
                pass
            
            try:
                for j in range(0,10):            
                    reviews = driver.find_elements_by_css_selector(
                            ".review-container")
                     
                    time.sleep(3)
                    
                    rating_code = reviews[j].find_element_by_css_selector(
                            ".ui_bubble_rating")
                    
                    # 5점이면 50,4점이면 40
                    cls_attr = rating_code.get_attribute("class")
                    cls_attr = str(cls_attr).split("ui_bubble_rating bubble_")
                    cafe_score = int(cls_attr[1])
                    
                    s.write(str(cafe_score))
                    print(cafe_score)
                    s.write("\n")
                    
                    Temp_review = reviews[j].find_element_by_css_selector(
                            ".partial_entry").text
                    cafe_review = Temp_review.replace("\n"," ")
    
                    r.write(cafe_review)
                    print(cafe_review)
                    r.write("\n") 
                    
                    #이름
                    for tag_name in tags_name:
                        n.write(tag_name.text)
                        print(tag_name.text)
                        n.write("\n")

                    #주소
                    a.write(tags_location[0].text)
                    print(tags_location[0].text)
                    a.write("\n")
            except:
                pass
            
        driver.quit()
        
n.close()
a.close()       
r.close()
s.close()

# 이름 불러오기
name = open("C:/Users/dadadah_ye/Downloads/name.txt",
            'r', encoding='utf-8-sig')
names = name.readlines() 

# 주소 불러오기
address = open("C:/Users/dadadah_ye/Downloads/address.txt",
               'r', encoding='utf-8-sig')
addresses = address.readlines()

# 리뷰 불러오기
review = open("C:/Users/dadadah_ye/Downloads/review.txt",
              'r', encoding='utf-8-sig')
reviews = review.readlines()

# 리뷰 개수 불러오기
score = open("C:/Users/dadadah_ye/Downloads/score.txt",
             'r', encoding='utf-8-sig')
scores = score.readlines()

# 데이터 생성
tripadvisor = pd.DataFrame({'name': names,
                     'address': addresses,
                     'review': reviews})

# 이름 공백 제거
for i in tripadvisor.index:
    if ' ' in tripadvisor['name'][i]:
        tripadvisor['name'][i] = tripadvisor['name'][i].replace(' ', '')

# train 데이터 생성
train = pd.DataFrame({'review': reviews,
                     'score': scores})
train = train.drop_duplicates()

for i in train.index:
    train['score'][i] = int(train['score'][i])
    
train['score'] = (train['score'] > 20).astype(int)
train['score'].value_counts()

train.to_csv('C:/Users/dadadah_ye/Downloads/train_tripadvisor.csv', encoding='utf-8-sig', index=False)
