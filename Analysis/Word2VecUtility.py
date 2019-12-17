from konlpy.tag import Okt
from multiprocessing import Pool

import re
import nltk

import pandas as pd
import numpy as np


class Word2VecUtility(object):

    @staticmethod
    def __get_stop_words():
        """ 한국어 불용어 리스트 가져오기.
        영어는 nltk안에 내장된 불용어 리스트가 있지만
        한국어의 경우 없기 때문에 자료를 다운 받아 사용.

        Reference - https://bab2min.tistory.com/544
        :return: stop words
        """
        stop_words = []

        with open('./한국어불용어100.txt', 'r', encoding='utf-8-sig') as f:
            while True:
                a = f.readline()
                if a:
                    a = a.split()[0]
                    stop_words.append(a)
                else:
                    break
        return stop_words

    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False, stop_words=[]):
        # 1. 특수문자를 공백으로 바꿔줌
        # review_text = re.sub('[^a-zA-Z]', ' ', review_text) -> English
        review_text = re.sub("[^가-힣-ㄱ-ㅎㅏ-ㅣ\\s]", " ", review)

        # 3. 어간추출 (konlpy okt tokenize 사용)
        okt = Okt()
        words = okt.morphs(review_text, stem=True)

        # 4. 불용어 목록 가져오기
        stop_words = Word2VecUtility.__get_stop_words()

        # 5. 불용어 제거
        if remove_stopwords:
            words = [w for w in words if not w in stop_words]

        # 6. 리스트 형태로 반환
        return words

    @staticmethod
    def review_to_join_words(review, remove_stopwords=False ):
        words = Word2VecUtility.review_to_wordlist(\
            review, remove_stopwords=False)
        join_words = ' '.join(words)
        return join_words


    @staticmethod
    def review_to_sentences( review, remove_stopwords=False):
        okt = Okt()

        # 1. nltk tokenizer를 사용해서 단어로 토큰화 하고 공백 등을 제거한다.
        raw_sentences = okt.morphs(review.strip())

        # 2. 각 문장을 순회한다.
        sentences = []
        for raw_sentence in raw_sentences:
            # 비어있다면 skip
            if len(raw_sentence) > 0:
                # 태그제거, 알파벳문자가 아닌 것은 공백으로 치환, 불용어제거
                sentences.append(\
                    Word2VecUtility.review_to_wordlist(\
                    raw_sentence, remove_stopwords))
        return sentences


    # 참고 : https://gist.github.com/yong27/7869662
    # http://www.racketracer.com/2016/07/06/pandas-in-parallel/
    # 속도 개선을 위해 멀티 스레드로 작업하도록
    @staticmethod
    def _apply_df(args):
        df, func, kwargs = args
        return df.apply(func, **kwargs)

    @staticmethod
    def apply_by_multiprocessing(df, func, **kwargs):
        # 키워드 항목 중 workers 파라메터를 꺼냄
        workers = kwargs.pop('workers')
        # 위에서 가져온 workers 수로 프로세스 풀을 정의
        pool = Pool(processes=workers)
        # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠 작업
        result = pool.map(Word2VecUtility._apply_df, [(d, func, kwargs)
                for d in np.array_split(df, workers)])
        pool.close()
        # 작업 결과를 합쳐서 반환
        return pd.concat(result)