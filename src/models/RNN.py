import tensorflow as tf
from resources import config


def train(train_dataset, test_dataset):
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=config.VOCAB_SIZE, standardize=None, ngrams=config.NGRAM)
    encoder.adapt(train_dataset.map(lambda text, label: text))
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=config.RNN['parameters']['EMBEDDING_LAYER_OUTPUT_DIM'],
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.RNN['parameters']['LSTM_OUTPUT_DIM'], )),
        tf.keras.layers.Dense(config.RNN['parameters']['DENSE_LAYER_OUTPUT_DIM'],
                              activation=config.RNN['parameters']['DENSE_LAYER_ACTIVATION']),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=config.RNN['parameters']['TRAIN_EPOCHS'],
                        validation_data=test_dataset,
                        validation_steps=config.RNN['parameters']['VALIDATION_STEPS'])
    model.save(config.RNN.model_location)