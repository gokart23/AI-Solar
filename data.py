import config
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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
        df = df.assign(City=new_column)

        data_frames.append(df)
        city_index += 1
    return pd.concat(data_frames, ignore_index=True)


def save_scaler_extrema(minimum, maximum, columns):
    """Pickle minimum and maximum values given by scaler for continuous features"""
    if not os.path.exists('data/norm_extrema'):
        os.makedirs('data/norm_extrema')

    dfmin = pd.DataFrame([minimum])
    dfmin.columns = columns
    dfmin.to_pickle('data/norm_extrema/norm_min.pkl')

    dfmax = pd.DataFrame([maximum])
    dfmax.columns = columns
    dfmax.to_pickle('data/norm_extrema/norm_max.pkl')


def preprocess():
    """Read raw csv data, preprocess and save to data/preprocessed.pkl"""
    df = read_csv_all_cities()

    # Drop night time hours
    df = df[~df.Hour.isin(config.dropped_hours)].reset_index(drop=True)

    # Convert hour, day, month and city into one-hot vectors
    onehot = ['Hour', 'Day', 'Month', 'City']
    dummified = pd.get_dummies(df, columns=onehot, prefix=onehot, prefix_sep='')
    column_names = dummified.columns

    # Normalize continuous features
    continuous_cols = dummified.iloc[:, :10]
    discrete_cols = dummified.iloc[:, 10:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(continuous_cols)
    save_scaler_extrema(scaler.data_min_, scaler.data_max_, column_names[:10])

    # Concatenate normalized continous features and discrete features
    norm_continuous_cols = pd.DataFrame(scaler.transform(continuous_cols))
    normalized = pd.concat([norm_continuous_cols, discrete_cols], axis=1,
                           ignore_index=True)
    normalized.columns = column_names

    normalized.to_pickle('data/preprocessed.pkl')


def read_data():
    """Read pickle containing preprocessed data"""
    return pd.read_pickle('data/preprocessed.pkl')


if __name__ == '__main__':
    preprocess()
