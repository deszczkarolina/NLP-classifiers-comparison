DATASET_LOCATION = "../resources/data/IMDB_validate.csv"

# models to evaluate
BERT = {
    'ENABLED': False,
    'NAME': 'Bert classifier',
    'MODEL_LOCATION': "BERT_local_1ep"
}

SVM = {
    'ENABLED': False,
    'NAME': 'SVM classifier',
    'MODEL_LOCATION': "BERT_local_1ep"
}

RNN = {
    'ENABLED': False,
    'NAME': 'RNN network',
    'MODEL_LOCATION': "../resources/RNN_SPAM_30ep"
}

CNN = {
    'ENABLED': True,
    'NAME': 'CNN network 15ep',
    'MODEL_LOCATION': "../resources/CNN_15_earlystopcross_fil7"
}
