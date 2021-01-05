train_data = "resources/data/IMDB.csv"
test_data = "test_location"

# text processing
VOCAB_SIZE = 1000
NGRAM = 1
OUTPUT_MODE = "tf-idf"

# training
BUFFER_SIZE = 10000
BATCH_SIZE = 64

# models
BERT = dict(
    enabled=True,
    parameters={
        "TRAIN_EPOCHS": 2,
        "LEARNING_RATE": 1e-4,
        "WARM_UP_STEPS_RATIO": 0.1,
        "DROPOUT_RATE": 0.1,
        "OUTPUT_LAYER_ACTIVATION": "softmax",
    },
    model_location="model_location"
)

SVM = dict(
    enabled=False,
    parameters={
        "TFIDF_MAX_FEATURES": 6000,
        "TRAIN_PERCENT": 0.7
    },
    model_location="model_location"
)

CNN = dict(
    enabled=False,
    parameters={
        "EMBEDDING_LAYER_OUTPUT_DIM": 64,
        "FILTERS_NUMBER": 128,
        "FILTERS_SIZE": 5,
        "CONV_LAYER_ACTIVATION": "relu",
        "DENSE_LAYER_OUTPUT_DIM": 10,
        "DENSE_LAYER_ACTIVATION": "relu",
        "OUTPUT_LAYER_ACTIVATION": "softmax",
        "TRAIN_EPOCHS": 2,
        "VALIDATION_STEPS": 30,
        "LEARNING_RATE": 1e-4
    },
    model_location="model_location"
)

RNN = dict(
    enabled=True,
    parameters={
        "EMBEDDING_LAYER_OUTPUT_DIM": 64,
        "LSTM_OUTPUT_DIM": 64,
        "DENSE_LAYER_OUTPUT_DIM": 64,
        "DENSE_LAYER_ACTIVATION": "relu",
        "TRAIN_EPOCHS": 10,
        "VALIDATION_STEPS": 30,
        "LEARNING_RATE": 1e-4
    },
    model_location="model_location"
)
