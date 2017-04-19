import data
import config
import metrics
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
    for city_ in config.cities:
        baseline_model(city_)

