import config
import data
import matplotlib.pyplot as plt
import numpy as np
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


def hourwise_dni_boxplot():
    """Plot DNI percentiles for each hour using boxplot"""
    df = data.read_csv_all_cities()

    hourwise_dni = df.groupby('Hour')['DNI'].apply(np.array)
    hourwise_dni = np.column_stack(hourwise_dni)

    fig = plt.figure()
    plt.boxplot(hourwise_dni, 0, '')
    fig.suptitle('Boxplot of DNI vs hour of day for all cities')
    plt.xlabel('Hour of day')
    plt.ylabel('Direct Normal Irradiance (W/m^2)')
    plt.show()

def plot_corr(df):
    """Function plots a graphical correlation matrix for input/ouput variables in dataframe."""

    corr = df.ix[:, :'Wind Speed'].corr()
    fig, ax = plt.subplots()
    plt.matshow(corr, cmap='jet')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=50, fontsize=8 )
    plt.yticks(range(len(corr.columns)), corr.columns, rotation=45, fontsize=6)
    plt.savefig("output/stats.full_confeature_correlation.png")
    plt.close()

    fig, ax = plt.subplots()
    in_out_corr = corr.ix[:, 3:][:3]
    plt.matshow(in_out_corr, cmap='jet')
    plt.colorbar()
    plt.xticks(range(len(corr.ix[0:2, 3:].columns)), corr.ix[0:2, 3:].columns, rotation=45, fontsize=8)
    plt.yticks(range(len(corr.ix[0:2, :3].columns)), corr.ix[0:2, :3].columns, rotation=45,fontsize=6)
    plt.savefig("output/stats.input_output_correlation.png")
    plt.close()

    fig, ax = plt.subplots()
    in_corr = corr.ix[:, 3:][3:]
    plt.matshow(in_corr, cmap='jet')
    plt.colorbar()
    plt.xticks(range(len(corr.ix[3:, 3:].columns)), corr.ix[3:, 3:].columns, rotation=45, fontsize=8)
    plt.yticks(range(len(corr.ix[3:, 3:].columns)), corr.ix[3:, 3:].columns, rotation=45,fontsize=6)
    plt.savefig("output/stats.input_confeature_correlation.png")
    plt.close()


if __name__ == '__main__':
    # hourwise_dni_boxplot()
    df = data.read_data()
    plot_corr(df)
