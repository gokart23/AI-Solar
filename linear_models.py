import data
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as metrics

def baseline_model(savefile="baseline_model", city=None):
	""" Train a un-normalized linear regression model and obtain predictions on test-set """

	if city == None:
		X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset()
	else:
		X_train, y_train, X_val, y_val, X_test, y_test = data.load_dataset(config.datasets[city])

	

	regr_lm = linear_model.LinearRegression()

	print("Fitting model")
	regr_lm.fit(X_train, y_train)
	# print('Coefficients shape: \n', regr_lm.coef_.shape)

	print("Predicting values")
	pred_val = regr_lm.predict(X_val)
	pred_test = regr_lm.predict(X_test)

	op_file = open(config.output_directory + "models." + savefile + ".txt", "w+")
	op_file.write("Baseline Model - Un-normalized Linear Regression\n\n")


	# The root-mean-squared error (RMSE)
	print("Mean squared error(validation): ", np.sqrt(metrics.mean_squared_error(y_val, pred_val)) )
	op_file.write( ("Mean squared error(validation): " + str(np.sqrt(metrics.mean_squared_error(y_val, pred_val))) + "\n") )
	# R2 coefficient: 1 is perfect prediction
	print("R2 score(validation): " , metrics.r2_score(y_val, pred_val, multioutput='variance_weighted'))
	op_file.write( ("R2 score(validation): " + str(metrics.r2_score(y_val, pred_val,  multioutput='variance_weighted')) + "\n") )

	print("Mean squared error(test): ",
				 np.sqrt(metrics.mean_squared_error(y_test, pred_test)))
	op_file.write( ("Mean squared error(test): " +
				 str(np.sqrt(metrics.mean_squared_error(y_test, pred_test))) + "\n") )
	print("R2 score(test): " , metrics.r2_score(y_test, pred_test, multioutput='variance_weighted'))
	op_file.write( ("R2 score(test): " + str(metrics.r2_score(y_test, pred_test, multioutput='variance_weighted')) + "\n") )

	op_file.close()

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