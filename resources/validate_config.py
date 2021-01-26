DATASET_LOCATION = "resources/data/Isoflavones_validate.csv"
CLEAN_TEXT = True

# models to evaluate
BERT = {
    'ENABLED': False,
    'NAME': 'Bert classifier',
    'MODEL_LOCATION': "output/BERT_SPAM"
}

SVM = {
    'ENABLED': True,
    'NAME': 'SVM classifier',
    'MODEL_LOCATION': "output/SVM_Isofl"
}

RNN = {
    'ENABLED': False,
    'NAME': 'RNN network',
    'MODEL_LOCATION': "output/RNN_SPAM"
}

CNN = {
    'ENABLED': True,
    'NAME': 'CNN network',
    'MODEL_LOCATION': "output/CNN_Isofl"
}
