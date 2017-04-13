import config
import data


def nrows_missing():
    """Number of rows containing missing data"""

    print('Number of rows containing missing data:')
    for city, directory_path in config.directories.items():
        df = data.read_csv_directory(directory_path)
        nrows = df.shape[0] - df.dropna().shape[0]
        print(city + '\t: ' + str(nrows))
    print('')


if __name__ == '__main__':
    nrows_missing()
