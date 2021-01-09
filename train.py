import tensorflow as tf
import numpy as np

from resources import train_config as config
from evaluate import evaluate_model
from utils.text_processing import load_dataset
from models import RNN, CNN, BERT


def train():
    df, classes = load_dataset(config.DATASET_LOCATION, config.CLEAN_TEXT)
    total_samples = len(df['text'])
    train_cutoff = int(np.floor(config.TRAIN_PERCENT * total_samples))
    train_df = df[0:train_cutoff]
    test_df = df[train_cutoff + 1: total_samples]

    if config.BERT["ENABLED"]:
        print('Training BERT model')
        train_dataset, test_dataset = convert_to_tf_types(train_df, test_df)
        BERT_model = BERT.train(train_dataset, test_dataset, len(classes))
        y_pred_proba = BERT_model.predict(test_df['text'])
        print('Evaluation of BERT model on test data')
        evaluate_model(y_pred_proba, test_df['label'], classes)
        tf.keras.backend.clear_session()

    if config.SVM["ENABLED"]:
        a = 1

    if config.CNN["ENABLED"]:
        print('Training CNN model')
        train_dataset, test_dataset = convert_to_tf_types(train_df, test_df)
        CNN_model = CNN.train(train_dataset, test_dataset, len(classes))
        y_pred_proba = CNN_model.predict(test_df['text'])
        print('Evaluation of CNN model on test data')
        evaluate_model(y_pred_proba, test_df['label'], classes)
        tf.keras.backend.clear_session()

    if config.RNN["ENABLED"]:
        print('Training RNN model')
        train_dataset, test_dataset = convert_to_tf_types(train_df, test_df)
        RNN_model = RNN.train(train_dataset, test_dataset, len(classes))
        y_pred_proba = RNN_model.predict(test_df['text'])
        print('Evaluation of RNN model on test data')
        evaluate_model(y_pred_proba, test_df['label'], classes)
        tf.keras.backend.clear_session()


def convert_to_tf_types(train_df, test_df):
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(train_df['text'].values, tf.string), tf.cast(train_df['label'].values, tf.int32)))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(test_df['text'].values, tf.string), tf.cast(test_df['label'].values, tf.int32)))

    # creates len(dataset)/BATCH_SIZE batches. Shuffle takes first BUFFER_SIZE elements from dataset,
    # than randomly takes BATCH_SIZE elements from buffer. Those elements are replaced with next ones from original
    # dataset. This is repeated until are samples are consumed.
    # prefetching allows later elements to be prepared while the current element is being processed. Improves latency
    # and throughput

    train_dataset = train_dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset


if __name__ == '__main__':
    train()
