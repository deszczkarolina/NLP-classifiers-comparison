import tensorflow as tf
import tensorflow_hub as hub
from official.nlp import optimization

from resources import config


def train(train_dataset, test_dataset, classes_num):
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
    tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='Text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='Preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(config.BERT['parameters']['DROPOUT_RATE'], name='Dropout')(net)
    net = tf.keras.layers.Dense(classes_num, activation=config.BERT['parameters']['OUTPUT_LAYER_ACTIVATION'],
                                name='Classifier')(net)
    model = tf.keras.Model(text_input, net)

    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    num_train_steps = steps_per_epoch * config.BERT['parameters']['TRAIN_EPOCHS']
    num_warmup_steps = int(config.BERT['parameters']['WARM_UP_STEPS_RATIO'] * num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=config.BERT['parameters']['LEARNING_RATE'],
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=config.BERT['parameters']['TRAIN_EPOCHS'],
                        validation_data=test_dataset,
                        validation_steps=config.BERT['parameters']['VALIDATION_STEPS'])

    model.save(config.BERT['model_location'])
