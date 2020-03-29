import matplotlib.pyplot as plt

import numpy as np
import itertools
import pandas as pd
from sklearn.preprocessing import scale
import re
from io import StringIO

import matplotlib
matplotlib.rc('font', serif='Helvetica Neue')
font = 15
lw = 1
ms=7
lw2 = 1

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param cm: array, confusion matrix
    :param classes: list, list of defined classes in the model
    :param normalize: bool, info if cm should be normalized
    :param title: str, title of the chart
    :param cmap: color of the chart
    return: plot of the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=font)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=font)
    plt.yticks(tick_marks, classes, fontsize=font)
    plt.xlim([-0.5,0.5+max(tick_marks)])
    plt.ylim([-0.5, 0.5 + max(tick_marks)])

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=font,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=font)
    plt.xlabel('Predicted label', fontsize=font)
    plt.tight_layout()


def class_feature_importance(X, Y, feature_importances):
    N, M = X.shape
    X = scale(X)

    out = {}
    for c in set(Y):
        out[c] = dict(
            zip(range(N), np.mean(X[Y == c, :], axis=0) * feature_importances)
        )

    return out


def pandas_classification_report(report):
    """
    Function return df of classification report
    :param report: array, classification report
    :return: dataframe, classification report
    """
    report = (report.split("accuracy")[0] + report.split("accuracy")[1].split("\n")[1])
    report = re.sub(r" +", " ", report).replace("\n ", "\n").replace("macro avg", "avg/total").split("\nweighted")[0]
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)
    return report_df
