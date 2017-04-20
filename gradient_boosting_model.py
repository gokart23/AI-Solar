import data
import config
import metrics
import numpy as np
import matplotlib.pyplot as plt
from config import gbm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib


def gradient_boosting_model(city='', modelsavefile='models/gbr', savefile='models.gradient_boosting_model'):
    """Train a gradient boosting model and obtain predictions on validation and test sets"""

    if city:
        X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset(config.datasets[city])
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset()

    # Create a model for each of output variables DHI, DNI and GHI
    nmodels = len(y_train.columns)
    gbr = [GradientBoostingRegressor(n_estimators=gbm['n_estimators'], learning_rate=gbm['learning_rate'],
                                     max_depth=gbm['max_depth'], random_state=0, loss='ls') for i in range(nmodels)]

    # Fit each model on training data
    if city:
        print('\nFitting model for ' + city, end='')
    else:
        print("\nFitting global model", end='')
    for i in range(nmodels):
        gbr[i].fit(X_train, y_train.iloc[:, i])
        print(' .', end='', flush=True)
    print('')

    # Save model to file
    save_path = modelsavefile + '_' + city + '.pkl' if city else modelsavefile + '.pkl'
    joblib.dump(gbr, save_path)

    # Predict each of three outputs and stack them into an array
    pred_val = np.column_stack([gbr[i].predict(X_val) for i in range(nmodels)])
    pred_test = np.column_stack([gbr[i].predict(X_test) for i in range(nmodels)])

    results_heading = 'Gradient Boosting Regression Model'
    metrics.generate_results(city, y_val, y_test, pred_val, pred_test, savefile, results_heading)


def plot_deviances(city=''):
    """Plot training and test set deviances of saved model"""
    if city:
        X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset(config.datasets[city])
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset()

    # Plot only for DNI
    op_index = 1
    gbr = load_gbr(city)[op_index]

    # compute test set deviance
    test_score = np.zeros((gbm['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(gbr.staged_predict(X_test)):
        test_score[i] = gbr.loss_(y_test.iloc[:, op_index], y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(gbm['n_estimators']) + 1, gbr.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(gbm['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.show()


def load_gbr(city='', savefile='models/gbr'):
    """Load pretrained gradient boosting regressor from file"""
    save_path = savefile + '_' + city + '.pkl' if city else savefile + '.pkl'
    return joblib.load(save_path)


if __name__ == "__main__":
    # gradient_boosting_model()
    # for city_ in config.cities:
    #     gradient_boosting_model(city_)

    plot_deviances()
