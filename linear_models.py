import data
import config
import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def baseline_model(city='', savefile="models.baseline_model"):
    """Train a linear regression model without regularization and obtain predictions on test-set"""

    if not city:
        X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset()
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset(config.datasets[city])

    regr_lm = linear_model.LinearRegression()

    if city:
        print('\nFitting model for ' + city)
    else:
        print("\nFitting global model")
    regr_lm.fit(X_train, y_train)
    # print('Coefficients shape: \n', regr_lm.coef_.shape)

    print("Predicting values")
    pred_val = regr_lm.predict(X_val)
    pred_test = regr_lm.predict(X_test)
    results_heading = 'Baseline Model - Linear Regression without regularization'
    metrics.generate_results(city, y_val, y_test, pred_val, pred_test, savefile, results_heading)


def lasso_model(city='', max_degree=4, savefile="models.lasso_model"):
    """Train a linear regression model with L1-regularization (and polynomial kernel),
       validate according to alpha, and obtain predictions on test-set"""

    if not city:
        X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset()
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset(config.datasets[city])

    for cur_degree in range(2, max_degree):
        for dataset in [X_train, X_val, X_test]:
            dataset = np.hstack( (dataset, np.power(dataset, cur_degree) ) )

    if city:
        print('\nFitting model for ' + city)
    else:
        print("\nFitting global model")

    
    max_alpha, max_r2 = 0, 0
    for alpha in [0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.1]:
        regr_lm = linear_model.Lasso(alpha=alpha, max_iter=1e6)

        regr_lm.fit(X_train, y_train)
        print("Predicting values on", alpha)
        pred_val = regr_lm.predict(X_val)
        pred_test = regr_lm.predict(X_test)

        results_heading = 'Baseline Model - Linear Regression with L1-regularization'
        results_heading += "\nCurrent alpha:" + str(alpha)

        rmse_val, r2_val, _, _ = metrics.generate_results(city, y_val, y_test, pred_val, pred_test, savefile + "_" + str(alpha), results_heading)
        max_alpha, max_r2 = (alpha if max_r2 < r2_val else max_alpha), (r2_val if max_r2 < r2_val else max_r2)
    
    # TODO: Change this to just rename instead of recompute
    print ("Best model observed on validation for alpha", max_alpha, "with score", max_r2)
    regr_lm = linear_model.Lasso(alpha=max_alpha, max_iter=1e8)

    regr_lm.fit(X_train, y_train)
    print("Predicting values on best model")
    pred_val = regr_lm.predict(X_val)
    pred_test = regr_lm.predict(X_test)

    results_heading = 'Baseline Model - Linear Regression with L1-regularization'
    results_heading += "\nCurrent alpha:" + str(alpha)

    metrics.generate_results(city, y_val, y_test, pred_val, pred_test, savefile + "_best", results_heading)

    plt.figure()
    plt.plot(regr_lm.coef_, color='gold', linewidth=2, label='Lasso coefficients')
    plt.savefig(savefile + "_best_coef_plot.png")
    plt.close()
    
    return max_alpha, max_r2

def get_model_results(model):
    """ Function to run the specified model on all default parameters on global and local datasets"""
    model()
    for city_ in config.cities:
        model(city_)

if __name__ == "__main__":
    # get_model_results(baseline_model)

    get_model_results(lasso_model)
