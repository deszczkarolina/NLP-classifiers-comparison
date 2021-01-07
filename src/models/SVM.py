import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC

from resources import config

params = config.SVM['parameters']


def vectorize(text):
    corpus = list(text)
    tfidf = TfidfVectorizer(max_features=params['TFIDF_MAX_FEATURES'])
    tfidf.fit(corpus)
    return tfidf.transform(corpus)


def train(train_features, train_labels):
    svm_model = LinearSVC()
    svm_model.fit(train_features, train_labels)
    return svm_model


def test(svm_model, test_features, test_labels):
    prediction = svm_model.predict(test_features)

    metrics = pd.DataFrame(
        index=['TF-IDF SVM'],
        columns=['Precision', 'Recall', 'F1 score', 'support']
    )
    metrics.loc['TF-IDF SVM'] = precision_recall_fscore_support(test_labels, prediction, average='binary')
    return metrics


# TODO integrate into train.py
def main():
    data = pd.read_csv('../../resources/data/SPAM.csv', encoding='latin-1')
    total_samples = len(data['text'])
    labels = data['label']
    tfidf_features = vectorize(data['text'])

    train_cutoff = int(np.floor(params['TRAIN_PERCENT'] * total_samples))

    train_features = tfidf_features[0:train_cutoff]
    test_features = tfidf_features[train_cutoff + 1: total_samples]
    train_labels = labels[0:train_cutoff]
    test_labels = labels[train_cutoff + 1: total_samples]

    model = train(train_features, train_labels)
    metrics = test(model, test_features, test_labels)
    print(metrics)


if __name__ == '__main__':
    main()
