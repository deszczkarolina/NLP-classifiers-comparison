import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc


def plot_model_training_history(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_roc_curve(pos_pred_proba, y_true):
    auc_roc = roc_auc_score(y_true, pos_pred_proba)
    print("ROC auc score: ", auc_roc)
    false_postive_rate, true_positive_rate, thresholds = roc_curve(y_true, pos_pred_proba)
    plt.plot(false_postive_rate, true_positive_rate, marker='.', label='ROC curve (auc = %0.3f)' % auc_roc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def plot_PR_curve(pos_pred_proba, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, pos_pred_proba)
    auc_pr = auc(recall, precision)
    print("P-R auc score: ", auc_pr)
    plt.plot(recall, precision, marker='.', label='PR curve (auc = %0.3f)' % auc_pr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


def evaluate_model(y_pred_proba, y_true, labels):
    y_pred = np.argmax(y_pred_proba, axis=-1)
    print('Confusion matrix:')
    print(sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels))
    print('Precision, Recall, F1, accuracy metrics:')
    # precision, recall, F1, accuracy
    print(sklearn.metrics.classification_report(y_true, y_pred))

    if len(labels) == 2:
        pos_pred_proba = np.delete(y_pred_proba, 0, 1)
        plot_roc_curve(pos_pred_proba, y_true)
        plot_PR_curve(pos_pred_proba, y_true)
