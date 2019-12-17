from .TextPreProcessor import TextPreProcessor

import nltk
import numpy as np
import os


class DataSet:
    """ Counter Vectorerizing DATASET

    Counter Vectorizing 을 해주는 데이터 셋 클래스.
    """

    def __init__(self, df, feature, label):
        self.x = df[feature]
        self.y = df[label]

    @staticmethod
    def _set_clean_data(self):
        """ Text Data cleaning.
        Text Processor 를 활용해 Input data들을 모두 Tokenize를 해줘,
        Word List로 변환.

        :param self: Object
        :rtype: list
        :return: Cleaning data
        """
        clean_data = []
        clean_data = TextPreProcessor.apply_by_multiprocessing(\
            self.x, TextPreProcessor.review_to_wordlist,\
            workers=4)
        return clean_data

    @staticmethod
    def _set_all_tokens(self):
        """ set all tokens.
        입력 데이터들의 토큰화 시킨 토큰들을 모아주는 함수.

        - vocabulary or 학습을 위해 준비.
        :rtype: list
        :return: all tokens
        """
        # Get all tokens
        tokens = [d for token in self._set_clean_data() for d in token]

        return tokens


    @staticmethod
    def ___get_texts_vocab_dict(self):
        all_text_vocab = nltk.Text(self._set_all_tokens(), name='NSMC')
        return all_text_vocab


    @staticmethod
    def get_term_frequency(self, doc, selected_count=10000) -> list:
        """ Get Text Word frequency.

        Input text data에서  단어 빈도
        :param doc: input text data.
        :param selected_count: vocabulary 에서
        :return:
        """
        text = nltk.Text(self._set_all_tokens(), name='NMSC') # 앞에서 미리 텍스트 처리 하고 여기서는 받아와서 쓰는 용도로 이 부분 처리 다시 하기
        selected_words = [f[0] for f in text.vocab().most_common(selected_count)]

        return [doc.count(word) for word in selected_words]

    @staticmethod
    def get_dataset(self):
        """ Get Counter Vectorizing Data Set.
        텍스트의 단어 빈도 수를 Feature 로 생성한 DataSet.

        :rtype: numpy array
        :return: x(feature data), y(target data)
        """
        x = np.asarray([self.get_term_frequency(d) for d in self.x]).astype("float32")
        y = np.asarray(self.y).astype("float32")

        return x, y

    @staticmethod
    def save_dataset(self,  data_name: str, data_in_path="./data_in/") -> None:
        """ Save Data Set.
        Numpy array 로 데이터 셋 저장해주는 함수.

        :param self:
        :param data_name: save data name
        :param data_in_path: save data path. (default './data_in/')
        :return: None
        """
        DATA_IN_PATH = data_in_path
        INPUT_DATA = 'nsmc_{}_input.npy'.format(data_name)
        LABEL_DATA = 'nsmc_{}_label.npy'.format(data_name)

        x, y = self.get_dataset()

        if not os.path.exists(DATA_IN_PATH):
            os.makedirs(DATA_IN_PATH)

        # 전처리 된 학습 데이터를 넘파이 형태로 저장
        np.save(open(DATA_IN_PATH + INPUT_DATA, 'wb'), x)
        np.save(open(DATA_IN_PATH + LABEL_DATA, 'wb'), y)

