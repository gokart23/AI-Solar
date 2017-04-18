import config
import numpy as np
import os
import pandas as pd
import _pickle as pkl
from sklearn.preprocessing import MinMaxScaler


def read_csv(file_path):
    """Read NREL csv file into data frame"""
    cols = [x for x in range(21) if x not in config.dropped_features]
    return pd.read_csv(file_path, header=0, index_col=False,
                       usecols=cols, skiprows=2)


def read_raw_data(city=''):
    """Read NREL csv files into data frame and concatenate

    If city is not specified, concatenate data of all cities and append
    a new column containing city indices.
    
    :param city: City name string
    :return: Raw data DataFrame
    """
    if city:
        files = os.listdir(config.directories[city])
        data_frames = []
        for file in sorted(files):
            file_path = os.path.join(config.directories[city], file)
            df = read_csv(file_path)
            data_frames.append(df)
        return pd.concat(data_frames, ignore_index=True)

    # Concatenate data of all cities with additional city index
    data_frames = []
    city_index = 0
    for city_ in sorted(config.cities):
        df = read_raw_data(city_)

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


def preprocess(city=''):
    """Read raw raw data, preprocess and save to file
    
    If city is specified, read raw data of the city and save preprocessed
    data to data/preprocessed_<city>.pkl. Else, read raw data of all
    cities and save to data/preprocessed.pkl.
    """
    df = read_raw_data(city)

    # Drop night time hours
    df = df[~df.Hour.isin(config.dropped_hours)].reset_index(drop=True)

    # Convert hour, day, month and city into one-hot vectors
    onehot = ['Hour', 'Day', 'Month']
    if not city:
        onehot.append('City')

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

    save_path = 'data/preprocessed_' + city + '.pkl' if city else 'data/preprocessed.pkl'
    normalized.to_pickle(save_path)
    print('Saved preprocessed data to: ' + save_path)


def read_data(city=''):
    """Read pickle containing preprocessed data"""
    if city:
        return pd.read_pickle('data/preprocessed_' + city + '.pkl')
    return pd.read_pickle('data/preprocessed.pkl')


def input_output_split(df):
    """Separate input variables and output variables of data"""
    return df.ix[:, 'Dew Point':], df.ix[:, :'GHI']


def train_validate_test_split(df, test_percent=.2, validate_percent=.25, seed=2345):
    """Generate training/validation/testing data uniformly from dataset"""
    np.random.seed(seed)
    perm = np.random.permutation(df.index)

    test_end = int(test_percent * df.shape[0])
    test = df.ix[perm[:test_end]]

    validate_end = int((1-test_percent) * validate_percent * df.shape[0]) + test_end
    validate = df.ix[perm[test_end:validate_end]]

    train = df.ix[perm[validate_end:]]

    return train, validate, test


def create_dataset(city='', savefile='dataset'):
    """Generate training, validation and testing data from preprocessed data
    
    If city is not specified, generate for all cities.
    """
    df = read_data(city)
    df_train, df_val, df_test = train_validate_test_split(df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = input_output_split(df_train), input_output_split(df_val), input_output_split(df_test)

    # Not using to_pickle or np.savez in order to maintain DF structure
    save_path = 'data/' + savefile + '_' + city + '.pkl' if city else 'data/' + savefile + '.pkl'
    with open(save_path, 'wb') as f:
        pkl.dump( dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test), f)
    print('Saved generated dataset to: ' + save_path)


def load_dataset(savefile="data/dataset.pkl"):
    """Load the training, validation and testing data from saved file"""
    with open(savefile, "rb") as f:
        data = pkl.load(f)

    return data['X_train'],data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']


if __name__ == '__main__':
    preprocess()
    for city in config.cities:
        preprocess(city)

    create_dataset()
    for city in config.cities:
        create_dataset(city)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(savefile=config.datasets['Delhi'])
