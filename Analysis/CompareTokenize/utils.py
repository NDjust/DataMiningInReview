from imblearn.under_sampling import RandomUnderSampler
from konlpy.tag import Hannanum, Okt, Twitter, Kkma, Mecab

import pandas as pd
import re


def get_tokenize(token="okt"):
    """ Get Tokenize.

    :param token: input token method.
    :return: token method(konlpy)
    """
    if token == "kkma":
        return Kkma()
    elif token == "twitter":
        return Twitter()
    elif token == "hannaum":
        return Hannanum()
    elif token == "mecab":
        return Mecab()

    return Okt()

# Solve data imbalanced
def get_under_sampling(df, target_label):
    """ Under Sampling.

    :param df: dataframe
    :param target_label: target label
    :return: fit data imbalanced dataframe.
    """
    rus  = RandomUnderSampler(return_indices=True)
    X_tl, y_tl, id_tl = rus.fit_sample(df, df[target_label])

    # remake data frame.
    columns = df.columns
    df = pd.DataFrame(X_tl, columns=columns)
    # df = df.astype(float)

    return df


def remove_emoji(df):
    """ Remove emoji data.

    :param df: dataframe
    :return: remove emoji
    """
    EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    return EMOJI.sub(r'', df)


# replace \n
def get_replaceText(df):
    """  줄바꿈 문자를 마침표로 변경.
    문장 요약 알고리즘을 위한 전처리.

    :param df: data frame.
    :return: replace "\n" -> "."
    """
    text = df.replace("\n",".")
    return text


# regular expression
def get_cleanText(df):
    """ 특수문자 처리 정규식.

    :param df: dataframe
    :return: clean text.
    """
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', df)
    return text
