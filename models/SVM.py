import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from resources import train_config

params = train_config.SVM['parameters']


def train(train_df):
    labels = train_df['label']
    tfidf_features, tfidf = vectorize(train_df['text'])
    model = LinearSVC().fit(tfidf_features, labels)
    save(tfidf, model)
    return tfidf, model


def predict(tfidf, model, test_texts):
    test_tfidf_features, tfidf = vectorize(test_texts, tfidf)
    return model.predict(test_tfidf_features)


def vectorize(texts, tfidf=None):
    corpus = list(texts)
    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=params['TFIDF_MAX_FEATURES'])
        tfidf.fit(corpus)
    return tfidf.transform(corpus), tfidf


def save(tfidf, model):
    to_save = {'tfidf': tfidf, 'model': model}
    joblib.dump(to_save, train_config.SVM['MODEL_LOCATION'])
