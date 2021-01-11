DATASET_LOCATION = "resources/data/SPAM_validate.csv"
CLEAN_TEXT = True

# models to evaluate
BERT = {
    'ENABLED': False,
    'NAME': 'Bert classifier',
    'MODEL_LOCATION': "BERT_local_1ep"
}

SVM = {
    'ENABLED': True,
    'NAME': 'SVM classifier',
    'MODEL_LOCATION': "output/SVM_6000_SPAM"
}

RNN = {
    'ENABLED': False,
    'NAME': 'RNN network',
    'MODEL_LOCATION': "../resources/RNN_SPAM_30ep"
}

CNN = {
    'ENABLED': False,
    'NAME': 'CNN network 15ep',
    'MODEL_LOCATION': "../resources/CNN_15_earlystopcross_fil7"
}
