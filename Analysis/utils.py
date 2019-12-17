from imblearn.under_sampling import RandomUnderSampler
from konlpy.tag import Okt

import pandas as pd
import numpy as np
import re


def under_sampling(df, target_label):
    """ Random Under Sampling.

    Solve data imbalanced.
    :param df: data frame
    :return: down sampling data frame.
    """
    rus = RandomUnderSampler(return_indices=True)
    print("Data label imbalanced percentage 1 : {} 0 : {}",
          len(df[df['score'] == 1]) / len(df), len(df[df['score'] == 0]) / len(df))

    X_tl, y_tl, id_tl = rus.fit_sample(df, df[target_label])

    # remake data frame.
    columns = df.columns
    df = pd.DataFrame(X_tl, columns=columns)
    # df = df.astype(float)

    return df


def term_frequency(doc, selected_words):
    return [doc.count(word) for word in selected_words]


def tokenize(data):
    okt = Okt()
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(data, norm=True, stem=True)]

"""
def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률 긍정 리뷰.\n".format(review, score * 100))
    else:
        print("[{}]는 {:.2f}% 확률로 부정 리뷰.\n".format(review, (1 - score) * 100))
"""
        
def preprocessing(review, okt, remove_stopwords=False, stop_words=[]):
    review_text = re.sub("[^가-힣-ㄱ-ㅎㅏ-ㅣ\\s]", "", review)
    
    word_review = okt.morphs(review_text, stem=True)
    
    if remove_stopwords:
        word_review = [token for token in word_review if not token in stop_words]
        
    return word_review


