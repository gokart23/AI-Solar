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

def rf_model(city='', savefile="models.rf_model"):
    """Train a linear regression model without regularization and obtain predictions on test-set"""

    if not city:
        X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset()
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset(config.datasets[city])

    clf = ensemble.RandomForestRegressor(n_jobs=-1)
    parameter_grid = {'n_estimators': [7, 10, 15, 20, 30], 'max_features' : ['auto', 'sqrt', 'log2']}
    regr_lm = GridSearchCV(clf, parameter_grid)

    if city:
        print('\nFitting model for ' + city)
    else:
        print("\nFitting global model")

    regr_lm.fit( np.vstack((X_train, X_val)), np.vstack((y_train, y_val)) )

    print("Predicting values")
    pred_val = regr_lm.predict(X_val)
    pred_test = regr_lm.predict(X_test)
    results_heading = 'Random Forest Regressor'
    results_heading += "\nBest params: " + str(regr_lm.get_params())
    metrics.generate_results(city, y_val, y_test, pred_val, pred_test, savefile, results_heading)


if __name__ == "__main__":
    metrics.get_model_results(rf_model)