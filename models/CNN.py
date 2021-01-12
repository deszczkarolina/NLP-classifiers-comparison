import tensorflow as tf

from resources import train_config
from utils.evaluate import plot_model_training_history

params = train_config.CNN['PARAMETERS']


def train(train_dataset, test_dataset, classes_num):
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=train_config.VOCAB_SIZE, standardize=None, ngrams=train_config.NGRAM)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=params['EMBEDDING_LAYER_OUTPUT_DIM'],
            mask_zero=True, name='Embedding'),
        tf.keras.layers.Conv1D(params['FILTERS_NUMBER'], params['FILTERS_SIZE'],
                               activation=params['CONV_LAYER_ACTIVATION'], name='Convolution'),
        tf.keras.layers.GlobalMaxPooling1D(name='Pooling'),
        tf.keras.layers.Dense(params['DENSE_LAYER_OUTPUT_DIM'],
                              activation=params['DENSE_LAYER_ACTIVATION'], name='Dense'),
        tf.keras.layers.Dense(classes_num, activation=params['OUTPUT_LAYER_ACTIVATION'],
                              name='Classification'),
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(params['LEARNING_RATE']),
                  metrics=['accuracy'])


    max_validation_steps = tf.data.experimental.cardinality(test_dataset).numpy()
    if params['VALIDATION_STEPS'] <= max_validation_steps:
        validation_steps = params['VALIDATION_STEPS']
    else:
        validation_steps = max_validation_steps

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=params['EARLY_STOPPING_PATIENCE_STEPS'],
                                                      restore_best_weights=True)

    history = model.fit(train_dataset, epochs=params['TRAIN_EPOCHS'],
                        validation_data=test_dataset, callbacks=[early_stopping],
                        validation_steps=validation_steps)

    print(model.summary())

    model.save(train_config.CNN['MODEL_LOCATION'])
    plot_model_training_history(history)
    return model
