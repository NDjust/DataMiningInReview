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
    test1 = "스페인 ‘아스’는 9일 '스페인 항공사 부엘링 하비에르 산체스-프리에토 사장이 메시만 특별대우 한 것 아니냐는 의혹을 받았다'고 보도했다.발단은 프리에토 사장의 인터뷰였다. 현재 바르셀로나 시내에서 약 10km 떨어진 엘 프랫 공항 확장이 지연되고 있다. 이와 관련해 ‘왜 계속 미뤄지냐’고 묻자, 프리에토 사장이 '비행기가 메시 집 상공을 날 수 없다'고 한마디 던졌다. 이 때문에 SNS가 들끓었다. 메시가 아무리 슈퍼스타라고 하나, 지나친 특권 아니냐는 의혹이 제기됐다.‘아스’가 추적 끝에 사실이 아님을 공개했다. 매체는 “메시의 집 위를 비행해선 안 되고, 낮잠을 방해받는다는 등 프리에토 사장이 그런 의도로 한 말이 아니다'라고 강조했다.메시는 바르셀로나 시내에서 약 25km 떨어진 가라프에 살고 있다. 가라프 공원에는 스페인 환경법 보호를 받는 멸종 위기 동식물이 서식하고 있다. 때문에 엘 프랫에서 출발,도착하는 비행기는 이 지역을 우회해야 한다. ‘메시 특권’ 의혹에 명확한 답을 내놨다”고 설명했다.프리에토 사장이 무심코 한 말에 메시가 큰 홍역을 치를 뻔했다."
    textrank1 = TextRank(test1)
    print(textrank1.summarize())  # gives you some text
    print(textrank1.summarize(3, verbose=False))  # up to 3 sentences, returned as list

    test2 = "[스포티비뉴스=박대성 기자] 유벤투스가 폴 포그바 재영입에 관심이다. 하지만 맨체스터 유나이티드가 책정한 몸값을 줄 생각은 없다.이탈리아 매체 ‘칼치오메르카토’는 9일(한국시간) “투토스포르트에 따르면 유벤투스가 여전히 포그바에게 관심이 있다. 포그바는 지난 여름 이적 시장에서 다양한 클럽과 연결됐고, 실제 새로운 도전을 원했다”고 밝혔다.문제는 이적료다. 맨유는 포그바 몸값으로 1억 5000만 유로(약 1915억 원)를 원했다. 레알 마드리드가 여름에 포그바에게 접근했지만 높은 이적료로 협상을 철회했다. 유벤투스도 마찬가지다. 유벤투스는 포그바 재영입에 1억 2000만 유로(약 1532억 원) 이상 지불할 계획이 없다.맨유는 포그바 몸값을 고수할 방침이다. 구단 고위층과 올레 군나르 솔샤르 감독 모두 팀 핵심 선수로 분류했다. 실제 협상에서는 더 높은 가격으로 레알 마드리드와 유벤투스 관심을 차단할 가능성도 있다."
    textrank2 = TextRank(test2)
    print(textrank2.summarize())
    print(textrank2.summarize(3, verbose=False))
