import string

import en_core_web_sm
import numpy as np
import pandas as pd
import tensorflow as tf
from spacy.lang.en.stop_words import STOP_WORDS


def remove_stopwords_and_lemmatize_pipe(doc):
    return ' '.join([token.lemma_ for token in doc
                     if token.text not in STOP_WORDS])


def remove_stopwords_and_lemmatize(texts):
    nlp = en_core_web_sm.load(disable=['tagger', 'parser', 'ner'])
    preprocessed_text = []
    for doc in nlp.pipe(texts, batch_size=20):
        preprocessed_text.append(remove_stopwords_and_lemmatize_pipe(doc))
    return np.array(preprocessed_text)


def clean_text(texts):
    # lower
    texts = texts.apply(lambda text: text.lower())
    # punctuation and numbers removal
    punct_table = str.maketrans('', '', string.punctuation)
    digit_table = str.maketrans('', '', string.digits)
    texts = texts.apply(lambda text: text.translate(punct_table))
    texts = texts.apply(lambda text: text.translate(digit_table))
    # remove stopwords and lemmatize
    return remove_stopwords_and_lemmatize(texts)


def process_text(file_name):
    df = pd.read_csv(file_name)
    df['text'] = clean_text(df['text'])
    classes_num = df.label.unique()
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(df['text'].values, tf.string), tf.cast(df['label'].values, tf.int32)))
    return dataset, classes_num
