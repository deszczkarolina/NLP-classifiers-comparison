from os import path

import pandas as pd
import numpy as np

# project root directory
current_file_dir = path.dirname(path.realpath(__file__))
root_dir = path.abspath(path.join(current_file_dir, "..", ".."))


def saveProcessedFiles(data, processed_file_path, validation_file_path, validation_percent):
    total_samples = len(data)
    validation_cutoff = int(np.floor(validation_percent * total_samples))
    validation_data = data[0:validation_cutoff]
    train_data = data[validation_cutoff + 1: total_samples]

    train_data.to_csv(processed_file_path, index=False)
    validation_data.to_csv(validation_file_path, index=False)


def process_IMDB(file_path, processed_file_path=path.join(root_dir, "resources/data/IMDB.csv"),
                 validation_file_path=path.join(root_dir, "resources/data/IMDB_validate.csv"), validation_percent=0.2):
    df = pd.read_csv(file_path)
    df.sentiment[df.sentiment == "positive"] = 1
    df.sentiment[df.sentiment == "negative"] = 0
    df = df.rename(columns={"sentiment": "label", "review": "text"})
    saveProcessedFiles(df, processed_file_path, validation_file_path, validation_percent)


def process_SPAM(file_path, processed_file_path=path.join(root_dir, "resources/data/SPAM.csv"),
                 validation_file_path=path.join(root_dir, "resources/data/SPAM_validate.csv"), validation_percent=0.2):
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df.replace({'label': {'ham': 0, 'spam': 1}})
    df.to_csv(processed_file_path, index=False)
    saveProcessedFiles(df, processed_file_path, validation_file_path, validation_percent)

