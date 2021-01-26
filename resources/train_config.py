DATASET_LOCATION = "resources/data/Isoflavones.csv"
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
    'PARAMETERS': {
        "TRAIN_EPOCHS": 10,
        "LEARNING_RATE": 1e-4,
        "WARM_UP_STEPS_RATIO": 0.1,
        "DROPOUT_RATE": 0.1,
        "VALIDATION_STEPS": 30,
        "OUTPUT_LAYER_ACTIVATION": "softmax",
        "EARLY_STOPPING_PATIENCE_STEPS": 5,
    },
    'MODEL_LOCATION': "output/BERT_SPAM"
}

SVM = {
    'ENABLED': True,
    'PARAMETERS': {
        "TFIDF_MAX_FEATURES": 6000,
        "C_REGULARIZATION": 2.5,
        "CLASS_WEIGHT": {
            0: 1,  # ham
            1: 1   # spam
        },
        "KERNEL": "linear"
    },
    'MODEL_LOCATION': "output/SVM_Isofl"
}

CNN = {
    'ENABLED': True,
    'PARAMETERS': {
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
        "EARLY_STOPPING_PATIENCE_STEPS": 5,
    },
    'MODEL_LOCATION': "output/CNN_Isofl"
}

RNN = {
    'ENABLED': False,
    'PARAMETERS': {
        "EMBEDDING_LAYER_OUTPUT_DIM": 64,
        "LSTM_OUTPUT_DIM": 64,
        "DENSE_LAYER_OUTPUT_DIM": 64,
        "DENSE_LAYER_ACTIVATION": "relu",
        "TRAIN_EPOCHS": 20,
        "VALIDATION_STEPS": 30,
        "LEARNING_RATE": 1e-4,
        "EARLY_STOPPING_PATIENCE_STEPS": 5,
    },
    'MODEL_LOCATION': "output/Isofl_RNN"
}
