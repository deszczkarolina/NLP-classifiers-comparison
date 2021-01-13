# NLP classifiers

## Aim of the project
The aim of this project was to create some text classifiers models and compare their performance 
on different datasets. We've decided to check how SVM, RNN, CNN and BERT models will manage the task of text classification
on IMDB reviews and SMS/SPAM datasets (both binary classifications) and on mails topic detection (multiclass).

## How to use it
There are two scripts, first (train.py) for models creation and training, second (validate.py) for evaluation of saved models on validation dataset.
Scripts use configuration files (train_config.py, validate_config.py respectively) which are located in resources folder.  
