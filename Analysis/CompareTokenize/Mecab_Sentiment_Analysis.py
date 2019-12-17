# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:22:07 2019

@author: 82108
"""

import pandas as pd
import MeCab
from imblearn.under_sampling import RandomUnderSampler

# tagger by Mecab
mecab = MeCab.Tagger()

# 데이터 불러오기
df = pd.read_csv('C:/data/all_review_score.csv', encoding='utf-8-sig')
# Null 제거
df = df.dropna()

# data imbalance check
# 1인 클래스 35831, 0인 클래스 5735
print(len(df[df['score']==1]), len(df[df['score']==0]))

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

train, test = train_test_split(df, test_size=0.2)

# tokenizing pos 태깅
train_x = []
train_y = []

test_x = []
test_y = []

for i, row in train.iterrows():
    try:
        train_x.append(['/'.join(token) for token in mecab.parse(row[1])])
        train_y.append(row[2])
    except TypeError:
        print("Error")

for i, row in test.iterrows():
    try:
        test_x.append(['/'.join(token) for token in mecab.parse(row[1])])
        test_y.append(row[2])
    except TypeError:
        print("Error")

# get all tokens
tokens = [d for token in train_x for d in token]
print(len(tokens))

import nltk
from pprint import pprint
text = nltk.Text(tokens, name="NMSC")

print(len(text.tokens))
print(len(set(text.tokens)))
pprint(text.vocab().most_common(10))
'''
[(',', 5785959),
 ('*', 4036370),
 ('\n', 834604),
 ('\t', 825428),
 ('N', 585909),
 ('T', 470950),
 ('F', 453639),
 ('V', 428071),
 ('/', 392190),
 ('E', 357442)]
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

# select most common value and 
""" Vectoring data with 10000 most commonly used tokens.

Use CountVectorization instead of One Hot Encoding.

This creates a word token in the set of review data and 
counts the number of each word to create the BOW encoded vector. 
"""
selected_words = [f[0] for f in text.vocab().most_common(10000)]

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

train_x = [term_frequency(d) for d in train_x]
test_x = [term_frequency(d) for d in test_x]

train_y = [c for c in train_y]
test_y = [c for c in test_y]

import numpy as np

# Transform type to float.
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
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(2399,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

y_score = model.fit(x_train, y_train, epochs=10, batch_size=512)
results = model.evaluate(x_test, y_test) # 57%의 저조한 acuraccy



# Modeling by tensorflow
import tensorflow as tf
X = tf.placeholder(tf.float32, shape=[None, 2399])
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal(shape=[2399, 1000]))
b1 = tf.Variable(tf.truncated_normal(shape=[1000]))
layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

W2 = tf.Variable(tf.truncated_normal(shape=[1000, 500]))
b2 = tf.Variable(tf.truncated_normal(shape=[500]))
layer2 = tf.sigmoid(tf.matmul(layer1, W2)+b2)

W3 = tf.Variable(tf.truncated_normal(shape=[500, 100]))
b3 = tf.Variable(tf.truncated_normal(shape=[100]))
layer3 = tf.sigmoid(tf.matmul(layer2, W3)+b3)

W4 = tf.Variable(tf.truncated_normal(shape=[100,1]))
b4 = tf.Variable(tf.truncated_normal(shape=[1]))
output = tf.sigmoid(tf.matmul(layer3, W4)+b4)

cost = -tf.reduce_mean(Y*tf.log(output) + (1-Y)*tf.log(1-output))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
optimizer = tf.train.RMSprop(0.01).minimize(cost)
is_correct = tf.cast(output > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(is_correct, Y), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10000):
        if step % 100 == 0:
            print(sess.run([cost, accuracy, optimizer], feed_dict = {X:x_train, Y:y_train})) # acuraccy 50%..