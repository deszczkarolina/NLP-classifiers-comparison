import shutil
import zipfile
from os import path, scandir, remove, listdir, rmdir

import pandas as pd


def extract_mails_data(file_path):
    working_dir = path.dirname(file_path)
    output_dir = path.join(working_dir, 'Data')
    if path.exists(output_dir) and path.isdir(output_dir):
        shutil.rmtree(output_dir)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(working_dir)
    return output_dir


def get_labels(data_dir):
    labels = []
    with scandir(data_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                labels.append(entry.name)
    labels.sort()
    return labels


def remove_duplicate_files(data_dir, labels):
    seen_files = set()
    remaining_labels = []
    for label in labels:
        label_dir = path.join(data_dir, label)
        with scandir(label_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    if entry.name in seen_files:
                        remove(entry.path)
                    else:
                        seen_files.add(entry.name)
        if len(listdir(label_dir)) != 0:
            remaining_labels.append(label)
        else:
            rmdir(label_dir)
    return remaining_labels


def read_files_into_dataframe(data_dir, labels):
    data = {}
    for label in labels:
        label_dir = path.join(data_dir, label)
        with scandir(label_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    with open(entry.path, "r", encoding="latin1") as file:
                        lines = file.readlines()
                        text = " ".join(lines).replace('\n', ' ')
                        data[len(data.keys())] = [label, text]
    data = pd.DataFrame(data).T
    data.columns = ['label', 'text']
    data = data.drop_duplicates(subset=['text'])
    return data


def get_class_counts(data_dir, labels):
    counts = dict()
    for label in labels:
        label_dir = path.join(data_dir, label)
        counts[label] = file_count(label_dir)
    return counts


def file_count(dir):
    return len([file for file in listdir(dir) if path.isfile(path.join(dir, file))])


def get_class_counts_df(data):
    return data.groupby('label').count()
