import data
import config
import metrics
import numpy as np
import matplotlib
import sklearn.svm as svm
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.ensemble as ensemble
from sklearn.model_selection import GridSearchCV


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
    return metrics.generate_results(city, y_val, y_test, pred_val, pred_test, savefile, results_heading)    


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

        results_heading = 'Linear Regression with L1-regularization'
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
    plt.plot(regr_lm.coef_[0, :], color='gold', linewidth=2, label='Lasso coefficients')
    plt.savefig(savefile + "_best_coef_plot.png")
    plt.close()
    
    return max_alpha, max_r2

def svr_model(city='', max_degree=4, savefile="models.svr_model"):
    """Train a support vector regression (SVR) model with different kernels,
       validate according to alpha, and obtain predictions on test-set"""

    if not city:
        X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset()
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset(config.datasets[city])

    if city:
        print('\nFitting model for ' + city)
    else:
        print("\nFitting global model")
    
    for predictor_idx in range( y_train.shape[1] ):
        best_model, best_C, best_r2 = 0, 0, 0
        for C in [0.1, 1, 100, 1000]:
            max_model, max_r2 = 0, 0
            svr_rbf = svm.SVR(kernel='rbf', C=C)
            svr_lin = svm.SVR(kernel='linear', C=C)
            svr_poly = svm.SVR(kernel='poly', C=C)
            
            for idx, regr_lm in enumerate([svr_rbf, svr_lin, svr_poly]):        
                regr_lm.fit(X_train, y_train.ix[:, predictor_idx])
                print("Predicting values on", C)
                pred_val = regr_lm.predict(X_val)
                pred_test = regr_lm.predict(X_test)

                results_heading = 'Linear Regression with L1-regularization'
                results_heading += "\nCurrent predictor:" + str(predictor_idx)

                rmse_val, r2_val, _, _ = metrics.generate_results(city, y_val.ix[:, predictor_idx], y_test.ix[:, predictor_idx], pred_val, pred_test, savefile + "_" + str(C) + "_" + str(predictor_idx), results_heading, skip_save=True)

                print()
                max_model, max_r2 = (idx if max_r2 < r2_val else max_model), (r2_val if max_r2 < r2_val else max_r2)    

            best_C, best_model, best_r2 = (C if max_r2 > best_r2 else best_C), (max_model if  max_r2 > best_r2 else best_model), (r2_val if  max_r2 > best_r2 else best_r2)    
            
        print ("Best model observed on validation for C", best_C, "with score", best_r2, "is ", best_model)        

        regr_lm = list([ svm.SVR(kernel='rbf', C=best_C), svm.SVR(kernel='linear', C=best_C), svm.SVR(kernel='poly', C=best_C) ])[ best_model ]

        regr_lm.fit(X_train, y_train.ix[:, predictor_idx])
        print("Predicting values on best model")
        pred_val = regr_lm.predict(X_val)
        pred_test = regr_lm.predict(X_test)

        results_heading = 'Support Vector Regression'
        results_heading += "\nCurrent C, model and predictor_idx:" + str(best_C) + "," + str(best_model) + "," + str(predictor_idx)

        metrics.generate_results(city, y_val.ix[:, predictor_idx], y_test.ix[:, predictor_idx], pred_val, pred_test, savefile + "_best_" + str(predictor_idx), results_heading)

# if __name__ == "__main__":
    # metrics.get_model_results(baseline_model)

    # metrics.get_model_results(lasso_model)