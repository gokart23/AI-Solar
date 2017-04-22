import data
import stats
import config
import metrics
import numpy as np
import matplotlib
import solar_city
import sklearn.svm as svm
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.ensemble as ensemble
import linear_models, random_forest_model 
from sklearn.model_selection import GridSearchCV


def get_feature_analysis_results(city_path):

    config.directories['test_city'] = city_path

    stats.hourwise_dni_boxplot(city='test_city', savefig=True)
    dni_plot_location = "output/hourwise_dni.png"

    data.preprocess('test_city')
    df = data.read_data('test_city')
    corr_plot_locations = stats.plot_corr(df, savefig=True)

    inp_df, out_df = data.input_output_split(df)
    # mi_scores = stats.select_k_best(inp_df, out_df.ix[:, 0])
    gini_scores = stats.feature_importance_bagging(inp_df, out_df.ix[:, 0])
    gini_scores = gini_scores[:15]

    return [ dni_plot_location, corr_plot_locations, gini_scores]

def get_loc_results(city_path):
    config.directories['test_city'] = city_path
    data.preprocess('test_city')
    data.create_dataset('test_city')
    config.datasets['test_city'] = 'data/dataset_test_city.pkl'

    kl_div = solar_city.baseline_comparison(city='test_city')
    gnb_conf = solar_city.gnb_model(test_city='test_city')
    lr_conf = solar_city.logistic_regression_model(test_city='test_city')


    return [kl_div, gnb_conf, lr_conf]

def get_radiation_results(city_path):
    config.directories['test_city'] = city_path
    data.preprocess('test_city')
    data.create_dataset('test_city')
    config.datasets['test_city'] = 'data/dataset_test_city.pkl'

    linear_results = linear_models.baseline_model(city='test_city')
    rf_results = random_forest_model.rf_model(city='test_city')

    mean = data.read_data('test_city')[['DNI']].mean()[0]
    power = (mean * 0.25 * 1.3 )

    return [linear_results, rf_results, power]


if __name__ == "__main__":
    # get_feature_analysis_results("data/NREL-Charanka-2000-2014")
    # get_radiation_results("data/NREL-Charanka-2000-2014")
    get_loc_results("data/NREL-Charanka-2000-2014")