import config
import data
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import warnings


def is_stationary(timeseries):
    """Return true if time series is stationary using dickey-fuller test"""
    adf = adfuller(timeseries, autolag='AIC')
    test_statistic = adf[0]
    critical_value_10 = adf[4]['10%']
    print(test_statistic, critical_value_10)
    return test_statistic < critical_value_10


def test_stationarity():
    """Print stationarity for each feature for every city"""
    for city, directory_path in config.directories.items():
        print('\nPerforming Dicker-Fuller test for stationarity of %s data:' % city)
        print('Test Statistic\tCritical Value (10%)')

        df = data.read_csv_directory(directory_path)
        stationarities = []
        for ci in range(df.shape[1]):
            # Ignore warnings raised when nan is returned by adfuller
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                stationarities.append(is_stationary(df.iloc[:, ci]))

        # Pretty print stationarities using pandas dataframe
        sdf = pd.DataFrame([stationarities], columns=df.columns, index=[' '])
        print(city + ' data stationarity:')
        print(sdf)


if __name__ == '__main__':
    test_stationarity()
