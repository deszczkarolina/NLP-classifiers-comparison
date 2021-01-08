import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

from evaluate import plot_model_training_history
from resources import train_config as config

map_name_to_handle = {
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
}

map_model_to_preprocess = {
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
}

params = config.BERT['parameters']


def train(train_dataset, test_dataset, classes_num):
    bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'

    tfhub_handle_encoder = map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='Text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='Preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(params['DROPOUT_RATE'], name='Dropout')(net)
    net = tf.keras.layers.Dense(classes_num, activation=params['OUTPUT_LAYER_ACTIVATION'],
                                name='Classifier')(net)
    model = tf.keras.Model(text_input, net)

    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    num_train_steps = steps_per_epoch * params['TRAIN_EPOCHS']
    num_warmup_steps = int(params['WARM_UP_STEPS_RATIO'] * num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=params['LEARNING_RATE'],
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(model.summary())

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

    plot_model_training_history(history)

    model.save(config.BERT['MODEL_LOCATION'])
    return model
