import config
import os
import pandas as pd


def read_csv(file_path):
    """Read NREL csv file into data frame"""
    cols = [x for x in range(21) if x not in config.dropped_features]
    return pd.read_csv(file_path, header=0, index_col=False,
                       usecols=cols, skiprows=2)


def read_csv_directory(directory_path):
    """Read NREL csv files from directory into data frame and concatenate."""
    files = os.listdir(directory_path)
    data_frames = []
    for file in sorted(files):
        file_path = os.path.join(directory_path, file)
        df = read_csv(file_path)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


def read_csv_all_cities():
    """Read NREL csv files for every city and concatenate with additional city index"""
    data_frames = []
    city_index = 0
    for city in sorted(config.directories.keys()):
        path = config.directories[city]
        df = read_csv_directory(path)

        # Create new column with city index and append it to data frame
        new_column = pd.DataFrame([city_index]*df.shape[0])
        df = df.assign(city=new_column)

        data_frames.append(df)
        city_index += 1
    return pd.concat(data_frames, ignore_index=True)

