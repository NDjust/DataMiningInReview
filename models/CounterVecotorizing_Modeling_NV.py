# import Tensorflow
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

# import keras
from keras.models import Sequential
# CNN
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten
# RNN
from keras.layers import Embedding, SpatialDropout1D, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
# Radam optimization.
from keras_radam.training import RAdamOptimizer

# import sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier


# import lightgbm
import lightgbm as lgb

import numpy as np


class RNN:
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 250
    # This is fixed.
    EMBEDDING_DIM = 100

    def train(self, x, y):

        model = Sequential()
        model.add(Embedding(RNN.MAX_NB_WORDS, RNN.EMBEDDING_DIM, input_length=len(x)))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(13, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        epochs = 5
        batch_size = 64

        history = model.fit(x, y, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                            callbacks=[EarlyStopping(monitor='val_loss',
                                                     patience=3, min_delta=0.0001)])

        return model, history

class CNN:

    def train(self, x, y, optimizer='adam'):

        model = Sequential()
        model.add(Conv1D(32, kernel_size=(3, 3), input_shape=(10000,), activation='relu'))
        model.add(Conv1D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        if optimizer == 'adam':
            optimizer = optimizers.Adam(lr=0.001)
        elif optimizer == 'RMS':
            optimizer = optimizers.RMSprop(lr=0.001)
        elif optimizer == "Radm":
            optimizer = RAdamOptimizer(learning_rate=1e-3)

        model.compile(optimizer=optimizer,
                      loss=losses.binary_crossentropy,
                      metrics=[metrics.binary_accuracy])

        model_y_score = model.fit(x, y, epochs=10, batch_size=512)

        print("model score = {}".format(model_y_score))

        return model


class DNN:

    def train(self, x, y, optimizer='adam'):
        """

        :param x: input data
        :param y: input label
        :param optimizer: selected optimization (adam, radam, rmsprorp)
        :return: trained model
        """
        model = models.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        if optimizer == 'adam':
            optimizer = optimizers.Adam(lr=0.001)
        elif optimizer == 'RMS':
            optimizer = optimizers.RMSprop(lr=0.001)
        elif optimizer == "Radm":
            optimizer = RAdamOptimizer(learning_rate=1e-3)

        model.compile(optimizer=optimizer,
                        loss=losses.binary_crossentropy,
                        metrics=[metrics.binary_accuracy])

        model_y_score = model.fit(x, y, epochs=10, batch_size=512)

        print("model score = {}".format(model_y_score))

        return model


class Xgb:

    GRID_PARAMS = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

    def __init__(self):
        self.xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

        self.params = {
            "learning_rate": 0.02,
            "n_estimators": 600,
            "objective": 'binary:logistic',
            "silent": True,
            "nthread": 1,
            'min_child_weight': 1,
            'gamma': 0.5,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'max_depth': 3

        }

    def train_randomized_search(self, x, y, folds=3):
        param_comb = 5

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

        random_search = RandomizedSearchCV(self.xgb, param_distributions=Xgb.GRID_PARAMS, n_iter=param_comb, scoring='roc_auc',
                                           n_jobs=4, cv=skf.split(x, y), verbose=3, random_state=1001)

        random_search.fit(x, y)

        print('\n Best estimator:')
        print(random_search.best_estimator_)
        print('\n Best hyperparameters:')
        print(random_search.best_params_)

        # replace in the best from the grid search
        self.params['min_child_weight'] = random_search.best_params_['min_child_weight']
        self.params['gamma'] = random_search.best_params_['gamma']
        self.params['colsample_bytree'] = random_search.best_params_['colsample_bytree']
        self.params['max_depth'] = random_search.best_params_['max_depth']
        self.params['subsample'] = random_search.best_params_['subsample']

        model = XGBClassifier(self.params)

        return model


class Lgb:

    def __init__(self):
        self.params = {'boosting_type': 'gbdt',
              'max_depth': -1,
              'objective': 'binary',
              'nthread': 3,  # Updated from nthread
              'num_leaves': 64,
              'learning_rate': 0.05,
              'max_bin': 512,
              'subsample_for_bin': 200,
              'subsample': 1,
              'subsample_freq': 1,
              'colsample_bytree': 0.8,
              'reg_alpha': 5,
              'reg_lambda': 10,
              'min_split_gain': 0.5,
              'min_child_weight': 1,
              'min_child_samples': 5,
              'scale_pos_weight': 1,
              'num_class': 1,
              'metric': 'binary_error'}

    def train_grid_search(self, x, y):
        # 학습시킬 파라미터 설정.
        gridParams = {
            'learning_rate': [0.005],
            'n_estimators': [40],
            'num_leaves': [6, 8, 12, 16],
            'boosting_type': ['gbdt'],
            'objective': ['binary'],
            'random_state': [501],  # Updated from 'seed'
            'colsample_bytree': [0.65, 0.66],
            'subsample': [0.7, 0.75],
            'reg_alpha': [1, 1.2],
            'reg_lambda': [1, 1.2, 1.4],
        }

        mdl = lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 n_jobs=3,  # Updated from 'nthread'
                                 silent=True,
                                 max_depth=self.params['max_depth'],
                                 max_bin=self.params['max_bin'],
                                 subsample_for_bin=self.params['subsample_for_bin'],
                                 subsample=self.params['subsample'],
                                 subsample_freq=self.params['subsample_freq'],
                                 min_split_gain=self.params['min_split_gain'],
                                 min_child_weight=self.params['min_child_weight'],
                                 min_child_samples=self.params['min_child_samples'],
                                 scale_pos_weight=self.params['scale_pos_weight'])

        grid = GridSearchCV(mdl, gridParams,
                            verbose=0,
                            cv=4,
                            n_jobs=2)

        # 학습시작
        grid.fit(x, y)

        # check Best estimator
        print('\n Best estimator:')
        print(grid.best_estimator_)
        print('\n Best hyperparameters:')
        print(grid.best_params_)

        # grid search로 나온 Best parameter를 모델의 최종 parameter로 설정.
        self.params['colsample_bytree'] = grid.best_params_['colsample_bytree']
        self.params['learning_rate'] = grid.best_params_['learning_rate']
        self.params['num_leaves'] = grid.best_params_['num_leaves']
        self.params['reg_alpha'] = grid.best_params_['reg_alpha']
        self.params['reg_lambda'] = grid.best_params_['reg_lambda']
        self.params['subsample'] = grid.best_params_['subsample']

        print('Set up params: ')
        print(self.params)

        model = lgb.LGBMClassifier(**self.params)

        return model



class SVM:

    def __init__(self):
        self.params = {'C': 0.1,
              'gamma':  0.1,
              'kernel': ['rbf']}

    def train_grid_search(self, x, y):
        # Create parameters to search
        grid_params = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

        grid = GridSearchCV(SVC(), grid_params, refit=True, verbose=3)

        grid.fit(x, y)

        # Best parameter 확인.
        print(grid.best_params_)
        print(grid.best_score_)

        # grid search로 나온 Best parameter를 모델의 최종 parameter로 설정.
        self.params['C'] = grid.best_params_['C']
        self.params['gamma'] = grid.best_params_['gamma']
        self.params['kernel'] = grid.best_params_['kernel']

        print('Set up params: ')
        print(self.params)

        model = SVC(**self.params)

        return model


class MLModel:

    def __init__(self):
        self.train_accrs = []
        self.validation_accrs = []
        self.confusion_matrixs = []

    def get_trained_model(self, model, train_x, train_y):
        """ Trained Ml model

        매개변수로 전달된 모델을 학습시키는 함수.

        Xgboost, ligthgb, svm은 Gridsearch 로 best parameter 선택 후 진행.
        그 외의 ML 모델들은 Default parameter 로 학습 진행.

        :param model: select ML model
        :param x: input data
        :param y: input label
        :return: trained model
        """
        if model == 'RF':
            model = RandomForestClassifier()
        elif model == 'Adaboost':
            model = AdaBoostClassifier()
        elif model == "Logistic":
            model = LogisticRegression()
        elif model == "xgb":
            # set-up params by using randomized search
            model = Xgb().train_randomized_search(train_x, train_y)
        elif model == 'lgb':
            # set-up params by using grid search
            model = Lgb().train_grid_search(train_x, train_y)
        elif model == 'svm':
            # set-up params by using grid search
            model = SVM().train_grid_search(train_x, train_y)

        model.fit(train_x, train_y)

        return model

    def get_metric(self):
        pass



# load train data
nsmc_train_x = np.load('E:/20171484/2019/2 - 텍스트마이닝/팀플/data_in_동사_명사_처리후/nsmc_train_input.npy')
nsmc_train_y = np.load('E:/20171484/2019/2 - 텍스트마이닝/팀플/data_in_동사_명사_처리후/nsmc_train_label.npy')

# load test data
nsmc_test_x = np.load('E:/20171484/2019/2 - 텍스트마이닝/팀플/data_in_동사_명사_처리후/nsmc_test_input.npy')
nsmc_test_y = np.load('E:/20171484/2019/2 - 텍스트마이닝/팀플/data_in_동사_명사_처리후/nsmc_test_label.npy')

# load validation data
nsmc_validation_x = np.load('E:/20171484/2019/2 - 텍스트마이닝/팀플/data_in_동사_명사_처리후/nsmc_validation_input.npy')
nsmc_validation_y = np.load('E:/20171484/2019/2 - 텍스트마이닝/팀플/data_in_동사_명사_처리후/nsmc_validation_label.npy')

mlmodel = MLModel()
rnn = RNN()
cnn = CNN()
dnn = DNN()


# RNN
RNN_model, RNN_history = rnn.train(nsmc_train_x, nsmc_train_y)
RNN_pred = RNN_model.predict(nsmc_test_x)

RNN_rocauc = roc_auc_score(nsmc_test_y, RNN_pred)
RNN_precision = precision_score(nsmc_test_y, RNN_pred)
RNN_recall = recall_score(nsmc_test_y, RNN_pred)
RNN_f1score = f1_score(nsmc_test_y, RNN_pred)
RNN_accuracy = accuracy_score(nsmc_test_y, RNN_pred)


# CNN
CNN_model = cnn.train(nsmc_train_x, nsmc_train_y,"adam")


# RF
RF_model = mlmodel.get_trained_model('RF', nsmc_train_x, nsmc_train_y)
RF_pred = RF_model.predict(nsmc_test_x)

RF_rocauc = roc_auc_score(nsmc_test_y, RF_pred)
RF_precision = precision_score(nsmc_test_y, RF_pred)
RF_recall = recall_score(nsmc_test_y, RF_pred)
RF_f1score = f1_score(nsmc_test_y, RF_pred)
RF_accuracy = accuracy_score(nsmc_test_y, RF_pred)


# Adaboost
Adaboost_model = mlmodel.get_trained_model('Adaboost', nsmc_train_x, nsmc_train_y)
Adaboost_pred = Adaboost_model.predict(nsmc_test_x)

Adaboost_rocauc = roc_auc_score(nsmc_test_y, Adaboost_pred)
Adaboost_precision = precision_score(nsmc_test_y, Adaboost_pred)
Adaboost_recall = recall_score(nsmc_test_y, Adaboost_pred)
Adaboost_f1score = f1_score(nsmc_test_y, Adaboost_pred)
Adaboost_accuracy = accuracy_score(nsmc_test_y, Adaboost_pred)

#metrics.plot_confusion_matrix(nsmc_test_y, Adaboost_pred, normalize=True)
#fpr2, tpr2, thresholds2 = roc_curve(nsmc_test_y, Adaboost_pred)


# Logistic
Logistic_model = mlmodel.get_trained_model('Logistic', nsmc_train_x, nsmc_train_y)
Logistic_pred = Logistic_model.predict(nsmc_test_x)

Logistic_rocauc = roc_auc_score(nsmc_test_y, Logistic_pred)
Logistic_precision = precision_score(nsmc_test_y, Logistic_pred)
Logistic_recall = recall_score(nsmc_test_y, Logistic_pred)
Logistic_f1score = f1_score(nsmc_test_y, Logistic_pred)
Logistic_accuracy = accuracy_score(nsmc_test_y, Logistic_pred)

#metrics.plot_confusion_matrix(nsmc_test_y, Logistic_pred, normalize=True)
#fpr3, tpr3, thresholds3 = roc_curve(nsmc_test_y, Logistic_pred)


# xgb
Xgb_model = mlmodel.get_trained_model('xgb', nsmc_train_x, nsmc_train_y)
Xgb_pred = Xgb_model.predict(nsmc_test_x)

Xgb_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.8, gamma=1.5,
              learning_rate=0.02, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=None, n_estimators=600, n_jobs=1,
              nthread=1, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=True, subsample=0.6, verbosity=1).fit(nsmc_train_x, nsmc_train_y)

Xgb_pred = Xgb_model.predict(nsmc_test_x)

Xgb_rocauc = roc_auc_score(nsmc_test_y, Xgb_pred)
Xgb_precision = precision_score(nsmc_test_y, Xgb_pred)
Xgb_recall = recall_score(nsmc_test_y, Xgb_pred)
Xgb_f1score = f1_score(nsmc_test_y, Xgb_pred)
Xgb_accuracy = accuracy_score(nsmc_test_y, Xgb_pred)



#lgb
Lgb_model = mlmodel.get_trained_model('lgb', nsmc_validation_x, nsmc_validation_y)

Lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.66,
               importance_type='split', learning_rate=0.005, max_bin=512,
               max_depth=-1, min_child_samples=5, min_child_weight=1,
               min_split_gain=0.5, n_estimators=40, n_jobs=3, num_leaves=16,
               objective='binary', random_state=501, reg_alpha=1, reg_lambda=1,
               scale_pos_weight=1, silent=True, subsample=0.7,
               subsample_for_bin=200, subsample_freq=1).fit(nsmc_train_x, nsmc_train_y)
Lgb_pred = Lgb_model.predict(nsmc_test_x)

Lgb_rocauc = roc_auc_score(nsmc_test_y, Lgb_pred)
Lgb_precision = precision_score(nsmc_test_y, Lgb_pred)
Lgb_recall = recall_score(nsmc_test_y, Lgb_pred)
Lgb_f1score = f1_score(nsmc_test_y, Lgb_pred)
Lgb_accuracy = accuracy_score(nsmc_test_y, Lgb_pred)


# svm
SVM_model = SVC(C=10, gamma=0.001, kernel='rbf').fit(nsmc_train_x, nsmc_train_y)
SVM_pred = SVM_model.predict(nsmc_test_x)

SVM_rocauc = roc_auc_score(nsmc_test_y, SVM_pred)
SVM_precision = precision_score(nsmc_test_y, SVM_pred)
SVM_recall = recall_score(nsmc_test_y, SVM_pred)
SVM_f1score = f1_score(nsmc_test_y, SVM_pred)
SVM_accuracy = accuracy_score(nsmc_test_y, SVM_pred)


# ConfusionMatrix
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(nsmc_test_y, Adaboost_pred, normalize=True, title='Adaboost Contusion Matrix')
skplt.metrics.plot_confusion_matrix(nsmc_test_y, Lgb_pred, normalize=True, title='Lightgbm Contusion Matrix')
skplt.metrics.plot_confusion_matrix(nsmc_test_y, Logistic_pred, normalize=True, title='LogisticRegression Contusion Matrix')
skplt.metrics.plot_confusion_matrix(nsmc_test_y, RF_pred, normalize=True, title='RandomForest Contusion Matrix')
skplt.metrics.plot_confusion_matrix(nsmc_test_y, SVM_pred, normalize=True, title='SVM Contusion Matrix')
skplt.metrics.plot_confusion_matrix(nsmc_test_y, Xgb_pred, normalize=True, title='Xgboost Contusion Matrix')

import matplotlib.pyplot as plt

fpr1, tpr1, thresholds1 = roc_curve(nsmc_test_y, Adaboost_pred)
fpr2, tpr2, thresholds2 = roc_curve(nsmc_test_y, Lgb_pred)
fpr3, tpr3, thresholds3 = roc_curve(nsmc_test_y, Logistic_pred)
fpr4, tpr4, thresholds4 = roc_curve(nsmc_test_y, RF_pred)
fpr5, tpr5, thresholds5 = roc_curve(nsmc_test_y, SVM_pred)
fpr6, tpr6, thresholds6 = roc_curve(nsmc_test_y, Xgb_pred)

plt.plot([0,1], [0,1], 'k')
plt.plot(fpr1, tpr1, label='Adaboost')
plt.plot(fpr2, tpr2, label='Lightgbm')
plt.plot(fpr3, tpr3, label='LogisticRegression')
plt.plot(fpr4, tpr4, label='RandomForest')
plt.plot(fpr5, tpr5, label='SVM')
plt.plot(fpr6, tpr6, label='Xgboost')
plt.legend()
plt.xlabel('Fall-Out')
plt.ylabel('Recall')
plt.title('ROC curve')
plt.show()