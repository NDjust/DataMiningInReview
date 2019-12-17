# 필요 패키지 import
from selenium import webdriver
import pandas as pd


# 크롬 브라우저 열기
driver = webdriver.Chrome('C:/chromedriver_win32/chromedriver')
# 식신 웹페이지로 이동
driver.get('https://www.siksinhot.com')
# 검색창에 "서울특별시 카페" 입력
driver.find_element_by_xpath(
        '//*[@id="header"]/div[2]/div[1]/div/div/input').send_keys('서울특별시 카페')
# 입력에 대해 클릭하여 이동하기
driver.find_element_by_xpath(
        '//*[@id="header"]/div[2]/div[1]/div/a/div/div').click()
# [검색 결과] 핫플레이스, 일반, 테마, 리뷰, 기타가 있는데 우리는 리뷰들을 크롤링 할 것이기 때문에 리뷰를 클릭 => 리뷰가 모인 곳으로 이동하게 됨
driver.find_element_by_xpath(
        '//*[@id="contents"]/div/div/div[1]/div/div/ul/li[4]/a').click()

'''
"더보기"에 대한 xpath를 copy해와서 click 이벤트를 주자
리뷰는 총 12851개이고 "더보기"를 누를때마다 5개씩 새로 추가됨
기존 5개의 리뷰 can see
(12851-5+1)/5 = 2569.4 => 총 2570번 loop
'''
import time
# 크롬 드라이버에 크롤러의 "더보기" 클릭 속도를 따라가지못하여 충분한 수의 반복횟수 설정
for i in range(6000):
    # more_view_clicks('//*[@id="schMove4"]/div/a/span')
    time.sleep(10)
    try:
        # "더보기"에 대한 xpath를 html 변수에 저장
        html = driver.find_element_by_xpath('//*[@id="schMove4"]/div/a/span')
        html.click() # 저장된 xpath에 대한 click 이벤트 발생
    except:
        print("Error")
    print('Count of Click Event: ', str(i))


# "더보기"가 모두 클릭된 html 저장
with open("C:/data/crawling_page_source.html", 'w', encoding='utf-8') as f:
    f.write(driver.page_source)

# 크롬 드라이버 닫기
driver.close()

#%%
# 저장된 html source 이용
driver = webdriver.Chrome('C:/chromedriver_win32/chromedriver')
driver.get('C:/data/crawling_page_source.html')

# 리뷰 긁어오기
review_class_list = driver.find_elements_by_class_name('score_story') # 리뷰에 대한 클래스 저장
num = 0
# 해당 게시글의 리뷰를 한개씩 긁어오면서 한줄씩 저장
with open('C:/data/review_content.csv', 'w', encoding='utf-8') as f:
    for review in review_class_list:
        # 리뷰는 해당 클래스의 p 태그 안에 들어있음
        f.write(review.find_element_by_tag_name('p').text+'\n')
        print("Success:", str(num))
        num += 1
    if num == len(review_class_list): print("last review crawling")

# 카페명 긁어오기
# 카페명에 대한 클래스 저장
name_class_list = driver.find_elements_by_class_name('store_area_menu')
num = 0
# 해당 게시글의 카페명을 한개씩 긁어오면서 한줄씩 저장
with open('C:/data/name_content.csv', 'w', encoding='utf-8') as f:
    for name in name_class_list:
         # 카페명은 해당 클래스의 a 태그 안에 들어있음
        f.write(name.find_element_by_tag_name('a').text+'\n')
        print("Success:", str(num))
        num += 1
    if num == len(name_class_list): print("last review crawling")

# 카페주소 긁어오기
 # 카페의 주소에 대한 클래스 저장
address_class_list = driver.find_elements_by_class_name('store_area_menu')
num = 0
# 해당 카페의 주소를 한개씩 긁어오면서 한줄씩 저장
with open('C:/data/address_content.csv', 'w', encoding='utf-8') as f:
    for address in address_class_list:
         # 카페의 주소는 해당 클래스의 span 태그 안에 들어있음
        f.write(address.find_element_by_tag_name('span').text+'\n')
        print("Success:", str(num))
        num += 1
    if num == len(address_class_list): print("last review crawling")

# 업종 긁어오기
 # 해당 가게의 업종에 대한 클래스 저장
category_class_list = driver.find_elements_by_class_name('store_area_menu')
num = 0
with open('C:/data/category_content.txt', 'w', encoding='utf-8') as f:
    for category in category_class_list:
         # 업종은 해당 클래스의 em 태그 안에 들어있음
        f.write(category.find_element_by_tag_name('em').text+'\n')
        print('Success: ', str(num))
        num += 1
    if num == len(category_class_list2): print('last CATEGORY crawling')


# 리뷰 작성자가 부여한 점수(score) 긁어오기
score_list = [] # score를 쌓기 위한 list
for i in range(1,10992):
    ''' 
    식신 사이트의 리뷰에는 score를 추가 안해도 되어, score가 없는 리뷰들이 있기때문에
    해당 태그가 없는 css selector 존재 => 에러 발생 => 예외처리로 해결
    '''
    try:
        score_list.append(driver.find_element_by_css_selector('#schMove4 > div > ul > li:nth-child({}) > div > div > div.score_story > span > strong'.format(i)).text)
        print("Successful: ", i) # 확인을 위한 출력문
    except:
        print('There is no score')
        score_list.append('') # score가 없으면 빈 문자열 추가

driver.close() # 크롬 드라이버 닫기

# 크롤링 파일 합치기

# 카페명 불러오기
datapath = "C:/data/name_content.txt" # 경로설정
# 파일 한줄씩 읽어들이기
name = open(datapath, 'r', encoding='UTF-8')
names = name.readlines() 

# 카페 주소 불러오기
datapath = "C:/data/address_content.txt" # 경로설정
#파일 한줄씩 읽어들이기
address = open(datapath, 'r', encoding='UTF-8')
addresses = address.readlines()

# 리뷰 불러오기
datapath = "C:/data/review_content.txt" # 경로설정
# 파일 한줄씩 읽어들이기
review = open(datapath, 'r', encoding='UTF-8')
reviews = review.readlines()

# 업종 불러오기
datapath = "C:/data/category_content.txt" # 경로설정
# 파일 한줄씩 읽어들이기
category = open(datapath, 'r', encoding='UTF-8')
categories = category.readlines()

# 데이터프레임으로 합치기
import pandas as pd
data = {'name':names, 
        'address':addresses, 
        'category':categories, 
        'review':reviews, 
        'score':score_list}
siksin = pd.DataFrame(data)

# csv 파일로 저장
siksin.to_csv('C:/data/siksin.csv', mode='w')


# 카테고리가 카페/디저트가 아닌 행들 존재 => 지워주자
no_cafe_index = []
# 카페/디저트가 아닌 인덱스 추출하기 위한 반복문
for i in range(len(categories)):
    if categories[i] != '카페/디저트\n':
        no_cafe_index.append(i)
print(no_cafe_index)
print(len(no_cafe_index)) # 개수확인 
# 카테고리가 카페/디저트가 아닌 행 지우기
for no_cafe in no_cafe_index:
    siksin = siksin.drop(no_cafe,)
# 확인
(siksin['category'] != '카페/디저트\n').sum() # 확인결과 모든 카테고리는 '카페/디저트'

# 주소가 서울이 아닌 타지역이 있는 것을 확인
no_seoul_address = []
for i in range(len(addresses)):
    if '서울-' not in addresses[i]:
        no_seoul_address.append(addresses[i])

# 개수확인
len(no_seoul_address) # 총 102개

# 제거하기
for no_seoul in no_seoul_address:
    for i in siksin.index:
        if siksin['address'][i] == no_seoul:
            siksin = siksin.drop(i,)

#%% 10월 11일 금요일 작업
# \n 제거 하기
for i in range(len(siksin)):
    for j in range(4):
        if '\n' in siksin.iloc[i,j]:
            siksin.iloc[i,j] = siksin.iloc[i,j].replace('\n', '')

# 카페명의 공백 제거하기
for i in siksin.index:
    if ' ' in siksin['name'][i]:
        siksin['name'][i] = siksin['name'][i].replace(' ', '')


# 카페 개수 확인하기 => 카페에 대한 리뷰 수 구하기
siksin_uniq = siksin['name'].unique() # 중복제거한 cafe들의 name
len(siksin_uniq) # 총 1928개의 카페


review_count = [] # 카페들의 리뷰수를 저장하기 위한 리스트
for i in siksin_uniq:
    # 확인을 위한 출력문
    print("\n",i,"\n",siksin.groupby('name').get_group(i).count()[1])
    # (카페명, 리뷰 수) 형태로 저장
    review_count += [(i,siksin.groupby('name').get_group(i).count()[1])]

siksin["review_counts"] = "" # siksin 데이터프레임에 review_counts라는 feature 추가

# siksin 데이터 프레임의 feature인 review_counts에 해당 카페의 리뷰 수 value 추가
for i in siksin.index: # 행 수 만큼의 반복문
     # review_counts는 리스트 안에 (카페명, 리뷰수) 형태의 튜플로 저장
    for name, number in review_count:
        if siksin["name"][i] == name: # 해당 행(카페명)이 review_counts안에 있는 카페명과 같으면
            siksin["review_counts"][i] = number # 해당 값 추가
            print(name, number) # 해당 카페의 리뷰수를 보기 위한 출력문

# category 양식 맞추기 "카페/디저트" -> "카페"
for i in siksin.index:
    if siksin['category'][i] == '카페/디저트':
        siksin['category'][i] = '카페'
# 확인
print((siksin['category'] != '카페').sum())

# address 양식 맞추기 ex) "서울-강남, 고려대/성신여대" -> "강남"
for i in siksin.index:
    remove_seoul = siksin['address'][i].split('-')  #'맨앞의 서울-' 제거
    extract_address = remove_seoul[1].split(',') #강남, 고려대/성신여대에서 ','를 기준으로 split
    siksin['address'][i] = extract_address[0] # 필요로하는 "강남"만 추출
# 확인
print(siksin['address'].values)
for i in siksin.index:
    # 강남, 강북, 송파 등 이러한 식으로 저장되었기때문에 글자수로 확인
    if len(siksin['address'][i]) > 2 or len(siksin['address'][i]) < 2:
        print(siksin['address'][i])

# 최종 데이터프레임 siksin_final.csv라는 이름으로 저장하기
siksin.to_csv("C:/data/siksin_final.csv", encoding='utf-8', index=False)

# siksin 데이터프레임의 열 추출(review, score)
review_and_score = siksin[['review','score']]
# review_and_score.csv로 파일 저장
review_and_score.to_csv('C:/data/review_and_score.csv', 
                        encoding='utf-8', 
                        index=False)
review_and_score


# review_and score 데이터프레임에서 score가 null값인 행들 지우기
for i in review_and_score['score'].index: # 데이터프레임의 인덱스를 이용한 반복문
    if review_and_score['score'][i] == '':
        review_and_score = review_and_score.drop(i,) # 제거후 6678개의 행으로 줆

# 데이터 imbalance check
print(len(review_and_score[review_and_score['score'] == '1.0'])) # 점수가 1.0인것은 38개
print(len(review_and_score[review_and_score['score'] == '2.0'])) # 점수가 2.0인것은 24개
print(len(review_and_score[review_and_score['score'] == '3.0'])) # 점수가 3.0인것은 415개
print(len(review_and_score[review_and_score['score'] == '4.0'])) # 점수가 4.0인것은 2975개
print(len(review_and_score[review_and_score['score'] == '5.0'])) # 점수가 5.0인것은 3226개

'''
score변수의 labeling을 위한 확인 작업:
score 1.0~2.0 => 부정, 4.0~5.0 => 긍정, 그러나 3.0은 애매하므로 리뷰를 살펴봄
'''
review_and_score[review_and_score['score'] == '3.0'] # 3.0의 점수가 부여된 리뷰들을 살펴보니 대부분 긍정적임

# labeling을 위한 반복문 생성
for i in review_and_score['score'].index:
    if review_and_score['score'][i] == '1.0' or review_and_score['score'][i] == '2.0':
        review_and_score['score'][i] = '0' # 부정 클래스를 0으로 설정
    else:
        review_and_score['score'][i] = '1' # 긍정 클래스를 1로 설정

# data imbalance check & 올바르게 labeling 되었는지 확인
print(len(review_and_score[review_and_score['score'] == '1'])) # 긍정클래스 6616개
print(len(review_and_score[review_and_score['score'] == '0'])) # 부정클래스 62개 
# data imbalance가 너무 심함

# null 제거 and 긍정/부적 레이블 설정한 데이터프레임 저장
review_and_score.to_csv('C:/data/review_joohyung.csv',
                        encoding='utf-8',
                        index=False)

# 데이터 불러오기
review_joohyung = pd.read_csv('C:/data/review_joohyung.csv', encoding='utf-8')
# 나단이가 작업한 데이터 불러오기(망고플레이트 사이트의 review와 score를 labeling한 데이터)
review_nadan = pd.read_csv('C:/data/review_data.csv', encoding = 'utf-8')
# 나의 데이터와 나단이의 데이터 합치기
mango_siksin_review_score = pd.concat([review_joohyung, review_nadan])
# data imbalance check
print(len(mango_siksin_review_score[mango_siksin_review_score['score'] == 0])) # 부정 클래스 5722개
print(len(mango_siksin_review_score[mango_siksin_review_score['score'] == 1])) # 긍정 클래스 28212개

# 합친 데이터 저장
mango_siksin_review_score.to_csv('C:/data/mango_siksin_review_score.csv',
                                 encoding='utf-8',
                                 index=False)


import pandas as pd
# trip advisor(다혜) 데이터 불러오기
trip = pd.read_csv('C:/data/train_tripadvisor.csv', encoding='utf-8')
# trip advisor 데이터 확인
trip.head()
len(trip) # 총 8612행
# checking data imbalance of trip advisor
print(len(trip[trip['score'] == 0])) # 부정 클래스 331개
print(len(trip[trip['score'] == 1])) # 긍정 클래스 8281개
# 어제 합친 mango(나단), siksin(주형) 합친 데이터 불러오기
mango_siksin = pd.read_csv('C:/data/mango_siksin_review_score.csv',
                           encoding='utf-8')
# 모두 합치기
mango_siksin_trip = pd.concat([mango_siksin, trip])
len(mango_siksin_trip) # 제대로 합쳐졌는지 확인 기존 33934개의 행+ 다혜의 8612개의 행
# data imbalance check
# 부정 클래스 확인 (기존 mango_siksin의 5722개 + trip의 331개)
print(len(mango_siksin_trip[mango_siksin_trip['score'] == 0])) # 6053개
# 긍정 클래스 확인 (기존 mango_siksin의 28212개 + trip의 8281개)
print(len(mango_siksin_trip[mango_siksin_trip['score'] == 1])) # 36493개
# 최종 데이터 저장
mango_siksin_trip.to_csv('C:/data/all_review_score.csv', encoding='utf-8')
