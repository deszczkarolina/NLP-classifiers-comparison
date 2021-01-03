import pandas as pd


def process_IMDB(file_path, processed_file_path="../../resources/data/IMDB.csv"):
    df = pd.read_csv(file_path)
    df.sentiment[df.sentiment == "positive"] = 1
    df.sentiment[df.sentiment == "negative"] = 0
    df = df.rename(columns={"sentiment": "label", "review": "text"})
    df.to_csv(processed_file_path, index=False)

# TODO przetworzanie pozostałych zbiorów danych do pliku csv o formacie text(string), label(int)
