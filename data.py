import os
import pandas as pd


def read_csv(file_path):
    """Read NREL csv file into data frame"""
    return pd.read_csv(file_path, header=0, index_col=False,
                       usecols=list(range(5, 20)), skiprows=2)


def read_csv_directory(directory_path):
    """Read NREL csv files from directory into data frame and concatenate."""
    files = os.listdir(directory_path)
    data_frames = []
    for file in sorted(files):
        file_path = os.path.join(directory_path, file)
        df = read_csv(file_path)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)
