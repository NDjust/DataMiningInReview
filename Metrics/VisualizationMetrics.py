 # import model metric
from sklearn.metrics import confusion_matrix, auc

import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sb


def plot_confusion_matrix(y_true, y_pred, class_names, vmax=None,
                          normed=True, title='Confusion matrix'):
    """  Plot Confusion Matix.(Model Metric)

    :param y_true: Real true value
    :param y_pred: prediction value
    :param ax: axis
    :param vmax: None
    :param normed:  Check Normalized
    :param title: plot title name
    :return: Node
    """
    matrix = confusion_matrix(y_true,y_pred)
    if normed:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    sb.heatmap(matrix, vmax=vmax, annot=True, square=True, ax=ax,
               cmap=plt.cm.Blues_r, cbar=False, linecolor='black',
               linewidths=1, xticklabels=class_names)
    plt.set_title(title, y=1.20, fontsize=16)
    #ax.set_ylabel('True labels', fontsize=12)
    plt.set_xlabel('Predicted labels', y=1.10, fontsize=12)
    plt.set_yticklabels(class_names, rotation=0)


def plot_acc_loss_history(history):
    """ plot keras Model accuracy & loss history.

    :param history: Keras modeling result history.
    :return: None
    """
    fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

    # summarize history for accuracy
    axis1.plot(history.history['acc'], label='Train', linewidth=3)
    axis1.plot(history.history['val_acc'], label='Validation', linewidth=3)
    axis1.set_title('Model accuracy', fontsize=16)
    axis1.set_ylabel('accuracy')
    axis1.set_xlabel('epoch')
    axis1.legend(loc='upper left')

    # summarize history for loss
    axis2.plot(history.history['loss'], label='Train', linewidth=3)
    axis2.plot(history.history['val_loss'], label='Validation', linewidth=3)
    axis2.set_title('Model loss', fontsize=16)
    axis2.set_ylabel('loss')
    axis2.set_xlabel('epoch')
    axis2.legend(loc='upper right')
    plt.show()


def get_rates(actives, scores):
    """
    :type actives: list[sting]
    :type scores: list[tuple(string, float)]
    :rtype: tuple(list[float], list[float])
    """

    tpr = [0.0]  # true positive rate
    fpr = [0.0]  # false positive rate
    nractives = len(actives)
    nrdecoys = len(scores) - len(actives)

    foundactives = 0.0
    founddecoys = 0.0
    for idx, (id, score) in enumerate(scores):
        if id in actives:
            foundactives += 1.0
        else:
            founddecoys += 1.0

        tpr.append(foundactives / float(nractives))
        fpr.append(founddecoys / float(nrdecoys))

    return tpr, fpr


def depict_ROC_curve(actives, scores, label, color, filename, randomline=True):
    """
    :type actives: list[sting]
    :type scores: list[tuple(string, float)]
    :type color: string (hex color code)
    :type fname: string
    :type randomline: boolean
    """

    plt.figure(figsize=(4, 4), dpi=80)

    setup_roc_curve_plot(plt)
    add_roc_curve(plt, actives, scores, color, label)
    save_roc_curve_plot(plt, filename, randomline)


def setup_roc_curve_plot(plt):
    """
    :type plt: matplotlib.pyplot
    """

    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)


def add_roc_curve(plt, actives, scores, color, label):
    """
    :type plt: matplotlib.pyplot
    :type actives: list[sting]
    :type scores: list[tuple(string, float)]
    :type color: string (hex color code)
    :type label: string
    """

    tpr, fpr = get_rates(actives, scores)
    roc_auc = auc(fpr, tpr)

    roc_label = '{} (AUC={:.3f})'.format(label, roc_auc)
    plt.plot(fpr, tpr, color=color, linewidth=2, label=roc_label)


def save_roc_curve_plot(plt, filename, randomline=True):
    """
    :type plt: matplotlib.pyplot
    :type fname: string
    :type randomline: boolean
    """

    if randomline:
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(filename)
