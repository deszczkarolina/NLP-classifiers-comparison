DATASET_LOCATION = "resources/data/SPAM.csv"
CLEAN_TEXT = True
TRAIN_PERCENT = 0.8

# text processing
VOCAB_SIZE = 1000
NGRAM = 1

# training
BUFFER_SIZE = 50000
BATCH_SIZE = 64

# models
BERT = {
    'ENABLED': False,
    'parameters': {
        "TRAIN_EPOCHS": 1,
        "LEARNING_RATE": 1e-4,
        "WARM_UP_STEPS_RATIO": 0.1,
        "DROPOUT_RATE": 0.1,
        "VALIDATION_STEPS": 30,
        "OUTPUT_LAYER_ACTIVATION": "softmax",
        "EARLY_STOPPING_PATIENCE_STEPS": 3,
    },
    'MODEL_LOCATION': "BERT_local_1ep"
}

SVM = {
    'ENABLED': True,
    'parameters': {
        "TFIDF_MAX_FEATURES": 6000,
    },
    'MODEL_LOCATION': "output/SVM_6000"
}

CNN = {
    'ENABLED': False,
    'parameters': {
        "EMBEDDING_LAYER_OUTPUT_DIM": 64,
        "FILTERS_NUMBER": 128,
        "FILTERS_SIZE": 5,
        "CONV_LAYER_ACTIVATION": "relu",
        "DENSE_LAYER_OUTPUT_DIM": 10,
        "DENSE_LAYER_ACTIVATION": "relu",
        "OUTPUT_LAYER_ACTIVATION": "softmax",
        "TRAIN_EPOCHS": 45,
        "VALIDATION_STEPS": 30,
        "LEARNING_RATE": 1e-4,
        "EARLY_STOPPING_PATIENCE_STEPS": 3,
    },
    'MODEL_LOCATION': "CNN_45_validation_exc"
}

RNN = {
    'ENABLED': False,
    'parameters': {
        "EMBEDDING_LAYER_OUTPUT_DIM": 64,
        "LSTM_OUTPUT_DIM": 64,
        "DENSE_LAYER_OUTPUT_DIM": 64,
        "DENSE_LAYER_ACTIVATION": "relu",
        "TRAIN_EPOCHS": 20,
        "VALIDATION_STEPS": 30,
        "LEARNING_RATE": 1e-4,
        "EARLY_STOPPING_PATIENCE_STEPS": 3,
    },
    'MODEL_LOCATION': "model_location"
}
