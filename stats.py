import config
import data
import metrics
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import feature_selection
from statsmodels.tsa.stattools import adfuller


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


def hourwise_dni_boxplot(city='', savefig=False):
    """Plot DNI percentiles for each hour using boxplot"""
    if city:
        df = data.read_raw_data(city)
    else:
        df =   data.read_csv_all_cities()

    hourwise_dni = df.groupby('Hour')['DNI'].apply(np.array)
    hourwise_dni = np.column_stack(hourwise_dni)

    fig = plt.figure()
    plt.boxplot(hourwise_dni, 0, '')
    fig.suptitle('Boxplot of DNI vs hour of day for all cities')
    plt.xlabel('Hour of day')
    plt.ylabel('Direct Normal Irradiance (W/m^2)')
    if savefig == True:
        plt.savefig("output/hourwise_dni.png")
    else:
        plt.show()



def plot_corr(df, savefig=False):
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

    return ["output/stats.full_confeature_correlation.png", "output/stats.input_output_correlation.png", "output/stats.input_confeature_correlation.png"]


def select_k_best(X, y, k=5, savefile="k_best_analysis.txt"):
    """Using chi squared (chi^2) statistical test for non-negative features to select k-best features from the dataframe"""
    print (X.shape, y.shape)
    test = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_regression, k=k)    
    fit = test.fit(X.values, y.values)
    sort_idx = np.argsort( fit.scores_ )[::-1]
    with open(config.output_directory + savefile, "w+") as f:
        metrics.print_write( '\n'.join( [ str((X.columns[ sort_idx[i] ], fit.scores_[ sort_idx[i] ])) for i in range(len(X.columns))] ), f )
    return [ str((X.columns[i], fit.scores_[i])) for i in range(len(X.columns))]

def feature_importance_bagging(X, y, savefile="feature_importance_bagging.txt"):
    model = ensemble.ExtraTreesRegressor()
    model.fit(X, y)
    sort_idx = np.argsort( model.feature_importances_ )[::-1]
    with open(config.output_directory + savefile, "w+") as f:
        # metrics.print_write( '\n'.join( [ str((X.columns[i], model.feature_importances_[i])) for i in range(len(X.columns))] ), f )
        # metrics.print_write( '\n', f)
        metrics.print_write( '\n'.join( [ str((X.columns[ sort_idx[i] ], model.feature_importances_[ sort_idx[i] ])) for i in range(len(X.columns))] ), f )

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.savefig(config.output_directory + "feature_importance_bagging.png")
    plt.close()

    return [ str((X.columns[ sort_idx[i] ], model.feature_importances_[ sort_idx[i] ])) for i in range(len(X.columns))]


def plot_mean_deviation():
    """Plot mean and standard deviation using errorbar plot
    
    Take input from array with three columns: city, mean and deviation.
    """
    df = pd.read_csv('output/dni_mean_deviation.csv', header=0, index_col=False)

    x = list(range(df.shape[0]))
    y = df['mean'][:15]
    e = df['deviation']

    plt.errorbar(x, y, e, linestyle='None', marker='o', fmt='-o')
    plt.bar(x, y, fill=False, width=0.4)
    plt.xticks(x, df['city'])
    plt.margins(0.15)
    plt.ylabel('DNI mean and deviation ($W/m^2$)')
    plt.show()


def plot_r2scores():
    """Plot R2 scores for each city and model"""
    df = pd.read_csv('output/r2data.csv', header=0, index_col=False)
    pos = list(range(df.shape[0]))
    width = 0.16
    alpha = 0.9
    # colors = ['#DFB48E', '#E3D298', '#7DC0C0', '#769BA8', '#787D77']
    colors = ['#2176AE', '#57B8FF', '#B66D0D', '#FBB13C', '#FE6847']

    fig, ax = plt.subplots(figsize=(10, 5))

    plt.bar(pos, df['Baseline'], width, alpha=alpha, color=colors[0])
    plt.bar([p+width for p in pos], df['Lasso'], width, alpha=alpha, color=colors[1])
    plt.bar([p+width*2 for p in pos], df['SVC'], width, alpha=alpha, color=colors[2])
    plt.bar([p+width*3 for p in pos], df['GBM'], width, alpha=alpha, color=colors[3])
    plt.bar([p+width*4 for p in pos], df['RF'], width, alpha=alpha, color=colors[4])

    ax.set_ylabel('$R^2$ score')
    ax.set_xticks([p+1.5*width for p in pos])
    ax.set_xticklabels(df['city'])

    # plt.xlim(min(pos)-width, max(pos)+width*4)
    # plt.ylim([0, max(df['Baseline'] + df['Lasso'] + df['SVC'])])
    plt.ylim([0.6, 1])

    plt.legend(['Baseline', 'Lasso', 'SVC', 'GBM', 'RF'], loc='upper left')
    plt.grid()
    plt.show()


def plot_confidences():
    """Plot Gaussian Naive Bayes and logistic regression confidence values for each city"""
    df = pd.read_csv('output/confidence.csv', header=0, index_col=False)
    pos = list(range(df.shape[0]))
    width = 0.33
    alpha = 0.9
    # colors = ['#DFB48E', '#E3D298', '#7DC0C0', '#769BA8', '#787D77']
    colors = ['#2176AE', '#57B8FF', '#B66D0D', '#FBB13C', '#FE6847']

    fig, ax = plt.subplots(figsize=(10, 5))

    plt.bar(pos, df['gnb'], width, alpha=alpha, color=colors[0])
    plt.bar([p+width for p in pos], df['logistic'], width, alpha=alpha, color=colors[3])

    ax.set_ylabel('Confidence')
    ax.set_xticks([1.07*p-0.1 for p in pos])
    ax.set_xticklabels(df['city'])

    # plt.xlim(min(pos)-width, max(pos)+width*4)
    # plt.ylim([0, max(df['Baseline'] + df['Lasso'] + df['SVC'])])
    plt.ylim([0.3, 1])

    plt.legend(['Gaussian Naive Bayes', 'Logistic Regression'], loc='upper left')
    plt.grid()
    plt.show()


def plot_kl_divergence():
    """Plot KL divergence values for different cites using bar plot"""
    df = pd.read_csv('output/divergence.csv', header=0, index_col=False)

    pos = list(range(df.shape[0]))
    plt.bar(pos, df['divergence'], color='#2176AE', tick_label=df['city'], width=0.5)
    plt.grid(axis='y', linestyle='dotted')
    plt.show()


if __name__ == '__main__':
    # hourwise_dni_boxplot()
    df = data.read_data()
    # plot_corr(df)
    
    inp_df, out_df = data.input_output_split(df)
    # scores = select_k_best(inp_df, out_df.ix[:, 0])
    feature_importance_bagging(inp_df, out_df.ix[:, 0])

    # print ( '\n'.join( [ (inp_df.columns[i], scores[i]) for i in range(out_df.shape[0])] ) )
    # plot_mean_deviation()
    # plot_r2scores()
    # plot_confidences()
    plot_kl_divergence()

