from tensorflow import keras

import evaluate
from utils import text_processing
from resources import validate_config as config


def validate():
    validate_path = config.DATASET_LOCATION
    validate_dataset, classes = text_processing.process_text(validate_path)

    if config.BERT["ENABLED"]:
        BERT_model = keras.models.load_model(config.BERT['MODEL_LOCATION'], compile=False)
        y_pred_proba = BERT_model.predict(validate_dataset['text'])
        print('Evaluation of ' + config.BERT['NAME'])
        evaluate.evaluate_model(y_pred_proba, validate_dataset['label'], classes)

    if config.SVM["ENABLED"]:
        SVM_model = keras.models.load_model(config.SVM['MODEL_LOCATION'])
        y_pred_proba = SVM_model.predict(validate_dataset['text'])
        print('Evaluation of ' + config.SVM['NAME'])
        evaluate.evaluate_model(y_pred_proba, validate_dataset['label'], classes)

    if config.CNN["ENABLED"]:
        CNN_model = keras.models.load_model(config.CNN['MODEL_LOCATION'])
        y_pred_proba = CNN_model.predict(validate_dataset['text'])
        print('Evaluation of ' + config.CNN['NAME'])
        evaluate.evaluate_model(y_pred_proba, validate_dataset['label'], classes)

    if config.RNN["ENABLED"]:
        RNN_model = keras.models.load_model(config.RNN['MODEL_LOCATION'])
        y_pred_proba = RNN_model.predict(validate_dataset['text'])
        print('Evaluation of ' + config.RNN['NAME'])
        evaluate.evaluate_model(y_pred_proba, validate_dataset['label'], classes)


if __name__ == '__main__':
    validate()
