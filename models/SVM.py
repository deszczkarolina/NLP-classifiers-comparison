import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC

from resources import train_config

params = train_config.SVM['parameters']


def train(train_df):
    labels = train_df['label']
    tfidf_features, tfidf = vectorize(train_df['text'])
    model = get_SVC()
    model.fit(tfidf_features, labels)
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


def get_SVC():
    if params['KERNEL'] == 'linear':
        return LinearSVC(
            C=params['C_REGULARIZATION'],
            class_weight=params['CLASS_WEIGHT'],
        )
    else:
        return SVC(
            C=params['C_REGULARIZATION'],
            class_weight=params['CLASS_WEIGHT'],
            kernel=params['KERNEL']
        )


def save(tfidf, model):
    to_save = {'tfidf': tfidf, 'model': model}
    joblib.dump(to_save, train_config.SVM['MODEL_LOCATION'])


def load(saved_model_path):
    loaded = joblib.load(saved_model_path)
    return loaded['tfidf'], loaded['model']
