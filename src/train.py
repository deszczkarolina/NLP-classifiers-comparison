import tensorflow as tf
import numpy as np

import evaluate
import resources.train_config as config
from utils import text_processing
from models import RNN, CNN, BERT


def train():
    dataset_path = config.DATASET_LOCATION
    df, classes = text_processing.process_text(dataset_path)
    total_samples = len(df['text'])
    train_cutoff = int(np.floor(config.TRAIN_PERCENT * total_samples))
    train_df = df[0:train_cutoff]
    test_df = df[train_cutoff + 1: total_samples]

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

    if config.BERT["ENABLED"]:
        BERT_model = BERT.train(train_dataset, test_dataset, len(classes))
        y_pred_proba = BERT_model.predict(test_df['text'])
        print('Evaluation of BERT model on test data')
        evaluate.evaluate_model(y_pred_proba, test_df['label'], classes)

    if config.SVM["ENABLED"]:
        a = 1

    if config.CNN["ENABLED"]:
        CNN_model = CNN.train(train_dataset, test_dataset, len(classes))
        y_pred_proba = CNN_model.predict(test_df['text'])
        print('Evaluation of CNN model on test data')
        evaluate.evaluate_model(y_pred_proba, test_df['label'], classes)

    if config.RNN["ENABLED"]:
        RNN_model = RNN.train(train_dataset, test_dataset, len(classes))
        y_pred_proba = RNN_model.predict(test_df['text'])
        print('Evaluation of RNN model on test data')
        evaluate.evaluate_model(y_pred_proba, test_df['label'], classes)


if __name__ == '__main__':
    train()
