from konlpy.tag import Twitter
from konlpy.tag import Okt
import re
from collections import Counter
from networkx import Graph
import networkx
from networkx import pagerank
from itertools import combinations
import itertools

# tagger 중 둘 중 하나만 사용.
# 거의 똑같은 것 같지만 Okt가 Twitter의 상위버전이므로 우선 사용
# 수집한 데이터로 비교해보는 게 좋을듯(근데 내가 봤을땐 똑같음)
twt = Twitter()
okt = Okt()


class Sentence:

    # 두 문장사이의 유사도 측정
    @staticmethod
    def co_occurence(sentence1, sentence2):
        p = sum((sentence1.bow & sentence2.bow).values())
        q = sum((sentence1.bow | sentence2.bow).values())
        return p / q if q else 0

    def __init__(self, text, index=0):
        self.index = index
        self.text = text
        self.nouns = okt.nouns(self.text)
        self.bow = Counter(self.nouns)

    def __eq__(self, another):
        return hasattr(another, 'index') and self.index == another.index

    def __hash__(self):
        return self.index  # 원문에서의 문장위치를 나타내는 데 쓰임


class TextRank(object):

    def __init__(self, text):
        self.text = text.strip()  # 문장의 좌우측 공백제거
        self.build()

    def build(self):

        self._build_sentences()
        self._build_graph()
        self.pageranks = networkx.pagerank(self.graph, weight='weight')
        self.reordered = sorted(self.pageranks, key=self.pageranks.get, reverse=True)

    def _build_sentences(self):
        dup = {}

        candidates = re.split(r'(?:(?<=[^0-9])\.|\n)', self.text)  # . , \n 기준으로 문장 나눠줌
        self.sentences = []
        index = 0
        for candidate in candidates:
            # 문장이 존재하고 끝에 .이나 공백으로 끝나면 다시 한번 제거
            while len(candidate) and (candidate[-1] == '.' or candidate[-1] == ' '):
                candidate = candidate.strip(' ').strip('.')

            # 문장이 존재하지만 dup안에 없으면 즉, 새로운
            if len(candidate) and candidate not in dup:
                dup[candidate] = True
                self.sentences.append(Sentence(candidate + '.', index))
                index += 1
        del dup
        del candidates

    def _build_graph(self):
        self.graph = Graph()  # 그래프 객체 생성
        self.graph.add_nodes_from(self.sentences)  # 문장을 노드로 생성

        # 문장을 음절단위로 쪼개서 combination을 만듦
        '''
        예를들어, '가나다라'
        [('가', '나'), ('가', '다'), ('가', '라'), ('나', '다'), ('나', '라'), ('다', '라')]
        '''
        for sent1, sent2 in combinations(self.sentences, 2):  #
            weight = self._jaccard(sent1, sent2)  # jaccard 거리를 이용하여 edge에 가중치 생성

            # jaccard거리를 이용하여 가중치가 있으면 두 노드 사이에 가중치 엣지 생성
            if weight:
                self.graph.add_edge(sent1, sent2, weight=weight)

    # jaccard거리를 이용한 두 문장사이의 유사도 측정
    def _jaccard(self, sent1, sent2):
        p = sum((sent1.bow & sent2.bow).values())
        q = sum((sent1.bow | sent2.bow).values())
        return p / q if q else 0

    def summarize(self, count=3, verbose=True):
        results = sorted(self.reordered[:count], key=lambda sentence: sentence.index)
        results = [result.text for result in results]
        if verbose:
            return '\n'.join(results)
        else:
            return results


if __name__ == "__main__":
    # test
    test1 = "Sentiment analysis is contextual mining of text which identifies and extracts subjective information in source material, and helping a business to understand the social sentiment of their brand, product or service while monitoring online conversations. However, analysis of social media streams is usually restricted to just basic sentiment analysis and count based metrics. This is akin to just scratching the surface and missing out on those high value insights that are waiting to be discovered. So what should a brand do to capture that low hanging fruit?"
    textrank1 = TextRank(test1)
    print(textrank1.summarize())  # gives you some text
    print(textrank1.summarize(3, verbose=False))  # up to 3 sentences, returned as list

    test2 = "Now this is where things get really interesting. To derive actionable insights, it is important to understand what aspect of the brand is a user discussing about. For example: Amazon would want to segregate messages that related to: late deliveries, billing issues, promotion related queries, product reviews etc. On the other hand, Starbucks would want to classify messages based on whether they relate to staff behavior, new coffee flavors, hygiene feedback, online orders, store name and location etc. But how can one do that?"
    textrank2 = TextRank(test2)
    print(textrank2.summarize())
    print(textrank2.summarize(3, verbose=False))