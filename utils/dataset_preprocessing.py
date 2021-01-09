import numpy as np

from utils.mails_utils import *

# project root directory
current_file_dir = path.dirname(path.realpath(__file__))
root_dir = path.abspath(path.join(current_file_dir, ".."))


def save_processed_files(data, processed_file_path, validation_file_path, validation_percent):
    total_samples = len(data)
    validation_cutoff = int(np.floor(validation_percent * total_samples))
    validation_data = data[0:validation_cutoff]
    train_data = data[validation_cutoff + 1: total_samples]

    train_data.to_csv(processed_file_path, index=False)
    validation_data.to_csv(validation_file_path, index=False)


def process_IMDB(
        file_path,
        processed_file_path=path.join(root_dir, "resources/data/IMDB.csv"),
        validation_file_path=path.join(root_dir, "resources/data/IMDB_validate.csv"),
        validation_percent=0.2
):
    df = pd.read_csv(file_path)
    df.sentiment[df.sentiment == "positive"] = 1
    df.sentiment[df.sentiment == "negative"] = 0
    df = df.rename(columns={"sentiment": "label", "review": "text"})
    save_processed_files(df, processed_file_path, validation_file_path, validation_percent)


def process_SPAM(
        file_path,
        processed_file_path=path.join(root_dir, "resources/data/SPAM.csv"),
        validation_file_path=path.join(root_dir, "resources/data/SPAM_validate.csv"),
        validation_percent=0.2
):
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df.replace({'label': {'ham': 0, 'spam': 1}})
    df.to_csv(processed_file_path, index=False)
    save_processed_files(df, processed_file_path, validation_file_path, validation_percent)


def process_MAILS(
        file_path,
        processed_file_path=path.join(root_dir, "resources/data/MAILS.csv"),
        validation_file_path=path.join(root_dir, "resources/data/MAILS_validate.csv"),
        validation_percent=0.2
):
    data_dir = extract_mails_data(file_path)
    labels = get_labels(data_dir)
    print("Class counts (original)", get_class_counts(data_dir, labels))
    labels = remove_duplicate_files(data_dir, labels)
    print("Class counts (after removing duplicate files)", get_class_counts(data_dir, labels))
    df = read_files_into_dataframe(data_dir, labels)
    print("Class counts (after removing duplicate texts)\n", get_class_counts_df(df))
    df = df.replace({'label': {'Crime': 0, 'Politics': 1, 'Science': 2}})
    save_processed_files(df, processed_file_path, validation_file_path, validation_percent)
    shutil.rmtree(data_dir)
