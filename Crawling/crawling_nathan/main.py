from crawling import save_link
from crawling import get_information
from crawling import get_review

import io
import pandas as pd
import sys

# 터미널에 한글 입/출력 인코딩 방지.
sys.stdin = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')


def main(keyword, link_file_name, df_content):
    """
    Final run crawling.

    input keyword restaurant save data & save data file to csv file.

    :param keyword: restaurant keyword
    :param link_file_name: save link file name
    :param dataframe_name: datframe name
    :return: None
    """
    save_link(keyword, link_file_name)
    links = [line.rstrip('\n') for line in open('./{}.txt'.format(link_file_name))]

    if df_content == "review":
        print("검색한 키워드에 대한 모든 리뷰와 점수들을 긁어 옵니다.")
        save_review(links)
    elif df_content == "all_info":
        print("검색한 키워드에 대한 모든 정보들을 긁어 옵니다.")
        save_all_info(links)

    print("Save Data Complete!!")

    return 0


def save_all_info(links):
    # make data frame and save csv file
    df = pd.DataFrame(columns=["name", "loc", "category",
                               "review_counts", "good_counts",
                               "well_counts", "bad_counts", "reviews"])

    # add data
    for i, link in enumerate(links):
        df.loc[i] = get_information(link)

    df.to_csv('./{}.csv'.format("all_info"),
              encoding='utf-8-sig', index=False)  # 한글 문자 처리.

    print("Save Data Complete!!")
    return 0


def save_review(links):
    df = pd.DataFrame(columns=["review", "score"])

    # add review & score data.
    for link in links:
        for review_score in get_review(link):
            if len(df.columns) == len(review_score):
                df.loc[len(df)] = review_score

    df.to_csv('./{}.csv'.format("review"), encoding='utf-8-sig', index=False)  # 한글 문자 처리.
    print("Save Data Complete!!")

    return 0


if __name__ == "__main__":
    choice_type = ["review", "all_info"]
    choice_num = int(input("리뷰와 평점 정보만 긁어 오실거면 0, 모든 정보들을 가져오실거면 1을 입력해주세요: "))

    if choice_num == 0 or choice_num == 1:
        keyword = input("정보를 가져오실 키워드를 입력해주세요: ")
        link_file_name = input("검색한 가게들 링크가 딤긴 txt 파일을 저장할 이름을 입력해주세요: ")
        print("크롤링을 시작합니다.")
        main(keyword, link_file_name, choice_type[choice_num])
    else:
        print("잘못된 번호를 입력하셨습니다. 다시 실행해주세요.")

