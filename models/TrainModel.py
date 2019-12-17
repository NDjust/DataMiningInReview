from .MachineLearningModels import *


def train_ML_model(model, train_x, train_y, parameter="defualt"):
    """ Trained Ml model

    매개변수로 전달된 모델을 학습시키는 함수.

    Xgboost, ligthgb, svm은 Gridsearch 로 best parameter 선택 후 진행.
    그 외의 ML 모델들은 Default parameter 로 학습 진행.

    :param model: select ML model
    :param x: input data
    :param y: input label
    :param parameter: set parameter default or searching
    :return: trained model
    """

    if model == 'RF':
        model = RandomForestClassifier()
    elif model == 'Adaboost':
        model = AdaBoostClassifier()
    elif model == "Logistic":
        model = LogisticRegression()
    elif model == "xgb":
        if parameter == "searching":
            # set-up params by using randomized search
            model = Xgb().train_randomized_search(train_x, train_y)
        else:
            model = Xgb().get_default_model()
    elif model == 'lgb':
        if parameter == "searching":
            # set-up params by using grid search
            model = Lgb().train_grid_search(train_x, train_y)
        else:
            model = Lgb().get_default_model()
    elif model == 'svm':
        if parameter == "searching":
            # set-up params by using grid search
            model = SVM().train_grid_search(train_x, train_y)
        else:
            model = SVM().get_default_model()

    model.fit(train_x, train_y)

    return model
