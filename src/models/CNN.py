import tensorflow as tf

from resources import config


def train(train_dataset, test_dataset, classes_num):
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=config.VOCAB_SIZE, standardize=None, ngrams=config.NGRAM)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=config.CNN['parameters']['EMBEDDING_LAYER_OUTPUT_DIM'],
            mask_zero=True, name='Embedding'),
        tf.keras.layers.Conv1D(config.CNN['parameters']['FILTERS_NUMBER'], config.CNN['parameters']['FILTERS_SIZE'],
                               activation=config.CNN['parameters']['CONV_LAYER_ACTIVATION'], name='Convolution'),
        tf.keras.layers.GlobalMaxPooling1D(name='Pooling'),
        tf.keras.layers.Dense(config.CNN['parameters']['DENSE_LAYER_OUTPUT_DIM'],
                              activation=config.CNN['parameters']['DENSE_LAYER_ACTIVATION'], name='Dense'),
        tf.keras.layers.Dense(classes_num, activation=config.CNN['parameters']['OUTPUT_LAYER_ACTIVATION'],
                              name='Classification'),
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(config.CNN['parameters']['LEARNING_RATE']),
                  metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=config.CNN['parameters']['TRAIN_EPOCHS'],
                        validation_data=test_dataset,
                        validation_steps=config.CNN['parameters']['VALIDATION_STEPS'])

    model.save(config.CNN['model_location'])
