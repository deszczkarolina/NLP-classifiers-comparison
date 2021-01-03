train_data = "train_location"
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
    parameters=dict(),
    model_location="model_location"
)

SVM = dict(
    enabled=False,
    parameters=dict(),
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
        "OUTPUT_LAYER_ACTIVATION": "sigmoid",
        "TRAIN_EPOCHS": 10,
        "VALIDATION_STEPS": 30,
        "LEARNING_RATE": 1e-4
    },
    model_location="model_location"
)

RNN = dict(
    enabled=False,
    parameters={
        "EMBEDDING_LAYER_OUTPUT_DIM": 64,
        "LSTM_OUTPUT_DIM": 64,
        "DENSE_LAYER_OUTPUT_DIM": 64,
        "DENSE_LAYER_ACTIVATION": "relu",
        "TRAIN_EPOCHS": 10,
        "VALIDATION_STEPS": 30
    },
    model_location="model_location"
)
