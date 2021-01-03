import tensorflow as tf

import resources.config as config
import text_processing
from models import RNN, CNN


def train():
    train_data_path = config.train_data

    dataset = text_processing.process_text(train_data_path)
    train_dataset = dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE)
    test_dataset = dataset.batch(config.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    if config.BERT["enabled"]:
        a = 1

    if config.SVM["enabled"]:
        a = 1

    if config.CNN["enabled"]:
        CNN.train(train_dataset, test_dataset)

    if config.RNN["enabled"]:
        RNN.train(train_dataset, test_dataset)
