# import sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

# import lightgbm
import lightgbm as lgb


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

    # default parameter model
    def get_default_model(self):
        return self.xgb

    def train_randomized_search(self, x, y, folds=3):
        """ Randomized Search.
        Randomized Search 알고리즘을 활용해 Xgboost 모델의 최적의 파라미터가
        들어간 모델 생성.

        :param x: train_x (np.array)
        :param y: train_y (np.array)
        :param folds: validation folds.
        :return: best parameter model.
        """
        param_comb = 5

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

        random_search = RandomizedSearchCV(self.xgb, param_distributions=Xgb.GRID_PARAMS, n_iter=param_comb,
                                           scoring='roc_auc', n_jobs=4,
                                           cv=skf.split(x, y), verbose=3, random_state=1001)

        random_search.fit(x, y)

        print('\n Best estimator:')
        print(random_search.best_estimator_)
        print('\n Best hyperparameters:')
        print(random_search.best_params_)

        # replace in the best from the RandomizedSearchCV
        self.params['min_child_weight'] = random_search.best_params_['min_child_weight']
        self.params['gamma'] = random_search.best_params_['gamma']
        self.params['colsample_bytree'] = random_search.best_params_['colsample_bytree']
        self.params['max_depth'] = random_search.best_params_['max_depth']
        self.params['subsample'] = random_search.best_params_['subsample']

        model = XGBClassifier(**self.params)

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

    def get_default_model(self):
        return lgb.LGBMClassifier()

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


    def get_default_model(self):
        return SVC()

    def train_grid_search(self, x, y):
        # Create parameters to search
        grid_params = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

        grid = GridSearchCV(SVC(), grid_params, refit=True, verbose=3)

        grid.fit(x, y)

        # check Best estimator
        print('\n Best estimator:')
        print(grid.best_estimator_)
        print('\n Best hyperparameters:')
        print(grid.best_params_)

        # grid search로 나온 Best parameter를 모델의 최종 parameter로 설정.
        self.params['C'] = grid.best_params_['C']
        self.params['gamma'] = grid.best_params_['gamma']
        self.params['kernel'] = grid.best_params_['kernel']

        print('Set up params: ')
        print(self.params)

        model = SVC(**self.params)  # input **args.

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
