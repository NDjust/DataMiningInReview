 # import model metric
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix


def view_clf_eval(y_test, pred):
    """Classification Evaluation Result.

    Metric Method.

    1. Confusion.
    2. Accuracy
    3. precision
    4. recall
    5. f1-score
    6. roc_score

    :param y_test: test data real values.
    :param pred: model predication values.
    :return: None
    """
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)

    precision = precision_score(y_test, pred, pos_label=1)  # set pos_label.
    recall = recall_score(y_test, pred)

    f1 = f1_score(y_test, pred)
    roc_score = roc_auc_score(y_test, pred)

    print('오차 행렬\n')
    print(confusion)
    print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현율 : {2:.4f}, '
          'f1-score : {3:.4f}, auc 값 : {4:.4f}'.format(accuracy, precision, recall, f1, roc_score))

    return None