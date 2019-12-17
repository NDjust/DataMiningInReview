# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:25:54 2019

@author: 82108
"""

from konlpy.tag import Twitter
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

# load the data
df = pd.read_csv('C:/data/all_review_score.csv', encoding='utf-8')
# Null 제거
df = df.dropna()

# data imbalance check
print(len(df[df['score']==1]), len(df[df['score']==0]))

# definition of function for under sampling
def under_sampling(df, target_label):
    """ Random Under Samplingls
    
    
    Solve data imbalanced.
    :param df: data frame 
    :return: down sampling data frame.
    """
    rus  = RandomUnderSampler(return_indices=True)
    X_tl, y_tl, id_tl = rus.fit_sample(df, df[target_label])

    # remake data frame.
    columns = df.columns
    df = pd.DataFrame(X_tl, columns=columns)
    # df = df.astype(float)

    return df

df = under_sampling(df, "score")

# downsampling 적용한 data imbalance check
print(len(df[df['score'] == 1]), len(df[df['score'] == 0]))

# train/test data 나누기
from sklearn.model_selection import train_test_split

# train:test = 8:2
train, test = train_test_split(df, test_size=0.2)

# tokenizing pos 태깅
train_x = []
train_y = []

test_x = []
test_y = []


for i, row in train.iterrows():
    try:
        train_x.append(['/'.join(token) for token in twt.pos(row[1], norm=True, stem=True)])
        train_y.append(row[2])
    except TypeError:
        print("Error")

for i, row in test.iterrows():
    try:
        test_x.append(['/'.join(token) for token in twt.pos(row[1], norm=True, stem=True)])
        test_y.append(row[2])
    except TypeError:
        print("Error")


# token get & check numbers
tokens = [d for token in train_x for d in token]
print(len(tokens))

import nltk
from pprint import pprint
text = nltk.Text(tokens, name="NMSC")

print(len(text.tokens)) # all tokens
print(len(set(text.tokens))) # remove duplicate tokens
pprint(text.vocab().most_common(10)) # most common value top 10
'''
[('./Punctuation', 24320),
 ('이/Josa', 13973),
 ('\n/Foreign', 13915),
 ('하다/Verb', 13747),
 ('에/Josa', 10581),
 ('가/Josa', 9056),
 ('는/Josa', 7857),
 ('도/Josa', 7451),
 ('맛/Noun', 7304),
 ('은/Josa', 7081)]
'''

# visualization most 50 words.
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
%matplotlib inline

font_fname = 'c:/windows/fonts/gulim.ttc'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

plt.figure(figsize=(20,10))
text.plot(50)

# select most common value and add the data to train, test
''' 
Vectoring data with 10000 most commonly used tokens.
Use CountVectorization instead of One Hot Encoding.

This creates a word token in the set of review data and 
counts the number of each word to create the BOW encoded vector. 
'''
selected_words = [f[0] for f in text.vocab().most_common(10000)]

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

train_x = [term_frequency(d) for d in train_x]
test_x = [term_frequency(d) for d in test_x]

train_y = [c for c in train_y]
test_y = [c for c in test_y]


# Transform type to float.
import numpy as np

x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')
y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')

# Modeling.
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

y_score = model.fit(x_train, y_train, epochs=10, batch_size=512)
'''
Epoch 1/10
9176/9176 [==============================] - 1s 56us/sample - loss: 0.6144 - binary_accuracy: 0.6998
Epoch 2/10
9176/9176 [==============================] - 0s 45us/sample - loss: 0.4390 - binary_accuracy: 0.8308
Epoch 3/10
9176/9176 [==============================] - 0s 44us/sample - loss: 0.3465 - binary_accuracy: 0.8703
Epoch 4/10
9176/9176 [==============================] - 0s 46us/sample - loss: 0.2802 - binary_accuracy: 0.8927
Epoch 5/10
9176/9176 [==============================] - 0s 43us/sample - loss: 0.2337 - binary_accuracy: 0.9147
Epoch 6/10
9176/9176 [==============================] - 0s 45us/sample - loss: 0.1825 - binary_accuracy: 0.9357
Epoch 7/10
9176/9176 [==============================] - 0s 44us/sample - loss: 0.1490 - binary_accuracy: 0.9506
Epoch 8/10
9176/9176 [==============================] - 0s 45us/sample - loss: 0.1298 - binary_accuracy: 0.9564
Epoch 9/10
9176/9176 [==============================] - 0s 46us/sample - loss: 0.0954 - binary_accuracy: 0.9705
Epoch 10/10
9176/9176 [==============================] - 0s 43us/sample - loss: 0.1040 - binary_accuracy: 0.9665
'''


results = model.evaluate(x_test, y_test)
'''
2294/2294 [==============================] - 0s 59us/sample - loss: 0.4241 - binary_accuracy: 0.8662
'''
# 정확도 약 86%

# model 평가 using roc curve
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

y_pred = model.predict(x_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_curve = auc(fpr, tpr)
auc_curve # 약 93%

# Draw plot.
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='OKt Tokenize (area = {:.3f})'.format(auc_curve))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# 새로운 데이터에 적용하여 평가해보기
def tokenize(data):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in twt.pos(data, norm=True, stem=True)]

def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.^^\n".format(review, score * 100))
    else:
        print("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.^^;\n".format(review, (1 - score) * 100))


'''
test review 1, 2, 5 - positive
test review 3, 4 - negative.
'''

# model test
# postive review test
pos_test1 = "커피심사위원으로 활동하셨을만큼 커피전문가가 내려주는 커피맛!!"
pos_test2 = "커피의, 커피에 의한, 커피를 위한 카페"
pos_test3 = "커피향과 커피맛이 따스한 느낌의 공간"
pos_test4 = "카페가 있을 것 같지 않은 곳에 있어서 신선하고 좋았어요~ 커피를 안파는 것도 특이했지만 가래떡구이에 꿀 찍어 먹는 것도!ㅋㅋㅋ좋았습니당"
pos_test5 = "갬성카페! 깔끔한 커피, 숨은고수"
pos_test6 = "카페바라보다... 능내역쪽에 있는 카페인데~ 정말 분위기가 너무너무 좋습니다. 개인적으로는 데이트하기에 가장 좋은 카페가 아닌가 하는 생각이 들었습니다. 일단 능내역 주변이 데이트하기에 좋습니다. 거기에 카페바라보다에 가시면 탁트인 통유리로 이쁜 풍경을 바라보면서 커피한잔 하실 수 있는 여유가^^ 와이프님이나 여자친구님 데리고 가시면 점수 팍팍 얻으실 수 있는 카페입니다~ 커피맛도 나쁘지 않습니다~ㅎㅎㅎ"

# negative review test
neg_test1 = "그럭저럭 갈만해요"
neg_test2 = "디저트류는 없고 커피랑 티만 있음"
neg_test3 = "먹기 힘들어요 별로에요"
neg_test4 = "커피는 별루..."
neg_test5 = "잠실에 있는 콩당콩당 카페의 인테리어와 메뉴를 도용한 까페;;"

"

predict_pos_neg(pos_test1)
predict_pos_neg(pos_test2)
predict_pos_neg(pos_test3)
predict_pos_neg(pos_test4)
predict_pos_neg(pos_test5)
predict_pos_neg(pos_test6)

predict_pos_neg(neg_test1)
predict_pos_neg(neg_test2)
predict_pos_neg(neg_test3)
predict_pos_neg(neg_test4)
predict_pos_neg(neg_test5)