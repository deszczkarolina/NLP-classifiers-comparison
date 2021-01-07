from os import path

import pandas as pd

# project root directory
current_file_dir = path.dirname(path.realpath(__file__))
root_dir = path.abspath(path.join(current_file_dir, "..", ".."))


def process_IMDB(file_path, processed_file_path=path.join(root_dir, "resources/data/IMDB.csv")):
    df = pd.read_csv(file_path)
    df.sentiment[df.sentiment == "positive"] = 1
    df.sentiment[df.sentiment == "negative"] = 0
    df = df.rename(columns={"sentiment": "label", "review": "text"})
    df.to_csv(processed_file_path, index=False)


def process_SPAM(file_path, processed_file_path=path.join(root_dir, "resources/data/SPAM.csv")):
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df.replace({'label': {'ham': 0, 'spam': 1}})
    df.to_csv(processed_file_path, index=False)
