import data
import config
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as metrics


def baseline_model(city='', savefile="models.baseline_model"):
    """Train an un-normalized linear regression model and obtain predictions on test-set"""

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

    op_path = config.output_directory + savefile
    op_path += '-' + city + '.txt' if city else '.txt'
    op_file = open(op_path, "w+")
    op_file.write("Baseline Model - Un-normalized Linear Regression\n\n")

    # The root-mean-squared error (RMSE) on validation data
    rmse_val = np.sqrt(metrics.mean_squared_error(y_val, pred_val))
    print_write("Mean squared error(validation): " + str(rmse_val), op_file)

    # R2 coefficient on validation data: 1 is perfect prediction
    r2_val = metrics.r2_score(y_val, pred_val, multioutput='variance_weighted')
    print_write("R2 score(validation): " + str(r2_val), op_file)

    # The root-mean-squared error (RMSE) on test data
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, pred_test))
    print_write("Mean squared error(test): " + str(rmse_test), op_file)

    # R2 coefficient on test data: 1 is perfect prediction
    r2_test = metrics.r2_score(y_test, pred_test, multioutput='variance_weighted')
    print_write('R2 score(test): ' + str(r2_test), op_file)

    op_file.close()


def print_write(line, f):
    """Print line and write line to file"""
    print(line)
    f.write(line + '\n')


# TODO: Generate figures for visual comparison of pred and actual values
# fig = plt.figure()
# plt.plot(range(y_test.shape[0]), pred_test[:, 0], '--')
# plt.plot(range(y_test.shape[0]), y_test.ix[:, 0], '--')
# plt.xlabel('Index')
# plt.ylabel('Direct Horizontal Irradiance (W/m^2)')
# plt.savefig("trial.png")
# plt.show()

if __name__ == "__main__":
    baseline_model()
    for city in config.cities:
        baseline_model(city)
