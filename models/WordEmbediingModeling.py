from .TrainModel import train_ML_model
from Metrics.ModelMetrics import view_clf_eval
import pandas as pd
import numpy as np

DATA_IN_PATH = './data/'
TRAIN_INPUT_DATA = 'nsmc_train_input.npy'
TRAIN_LABEL_DATA = 'nsmc_train_label.npy'
TEST_INPUT_DATA = 'nsmc_test_input.npy'
TEST_LABEL_DATA = 'nsmc_test_label.npy'
VAL_INPUT_DATA = 'nsmc_validation_input.npy'
VAL_LABEL_DATA = 'nsmc_validation_label.npy'

# load train dataset
train_x = np.load(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'rb'))
train_y = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'rb'))

# load validation dataset
val_x = np.load(open(DATA_IN_PATH + VAL_INPUT_DATA, 'rb'))
val_y = np.load(open(DATA_IN_PATH + VAL_LABEL_DATA, 'rb'))

# load test dataset
test_x = np.load(open(DATA_IN_PATH + TEST_INPUT_DATA, 'rb'))
test_y = np.load(open(DATA_IN_PATH + TEST_LABEL_DATA, 'rb'))

# random forest
rf_test = train_ML_model("RF", train_x, train_y)
y_pred = rf_test.predict(test_x)
view_clf_eval(test_y, y_pred)

# adaboost
adb = train_ML_model("Adaboost", train_x, train_y)
y_pred = adb.predict(test_x)
view_clf_eval(test_y, y_pred)

# svm
svm_test = train_ML_model("svm", train_x, train_y)
y_pred = svm_test.predict(test_x)
view_clf_eval(test_y, y_pred)

# logistic
logistic_test = train_ML_model("Logistic", train_x, train_y)
y_pred = logistic_test.predict(test_x)
view_clf_eval(test_y, y_pred)

# xgboost
xgb_test = train_ML_model("xgb", train_x, train_y)
y_pred = xgb_test.predict(test_x)
view_clf_eval(test_y, y_pred)
