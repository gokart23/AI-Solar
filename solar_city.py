import data
import config
import metrics
import scipy.stats
import numpy as np
from sklearn import naive_bayes
from sklearn import linear_model

def extract_dni_stats(savefile="dni_train_city_values.txt"):
    """ Extract mean and std_dev values for DNI"""
	means, stds = [], []
	for city in config.solar_power_cities:
		df = data.read_data(city)
		means.append(df[['DNI']].mean())
		stds.append(df[['DNI']].std())
	
	with open(config.output_directory + savefile, "w+") as f:
		for mean, std in zip(means, stds):
			f.write( str(mean) + "," + str(std) + "\n")
		tot_mean, tot_std = np.mean(means), np.mean(stds)
		f.write("\n" + str(tot_mean) + "," + str(tot_std) + "\n")

def baseline_comparison( city, mean=config.solar_mean, std=config.solar_std, header="", savefile="baseline_comparison.txt" ):
    """ Baseline model for calculating KL-divergence from Gaussian-fit training data """

	df = data.read_data(city)
	df_mean, df_std = df[['DNI']].mean()[0], df[['DNI']].std()[0]
	mean_prob = scipy.stats.norm(mean, std).pdf( df_mean )
	mean_confidence = 100.*(scipy.stats.norm(mean, std).cdf(df_mean) - scipy.stats.norm(mean, std).cdf(-df_mean))

	kl_div = np.log( df_std / std ) + ( ((std**2) + np.power( (df_mean - mean), 2)) / (2 * np.power(df_std, 2))  ) - 0.5
	
	with open(config.output_directory + savefile, "w+") as f:
		metrics.print_write( header + "\n" + str( (df_mean, df_std, mean_prob, mean_confidence, kl_div )  ) + "\n", f )

def gnb_model(test_city, header="", savefile="gnb_model.txt"):
    """ Gaussian Naive Bayes for predicting confidence as a solar power location """
    training_df, training_op = [], []
    for city in config.solar_power_cities:
        df = data.read_data(city)
        inp_df, _ = df, df #data.input_output_split(df)
        if len(training_df) == 0:
            training_df = inp_df
            training_op = np.ones( inp_df.shape[0], dtype=int )
        else:
            training_df = training_df.append(inp_df, ignore_index=True)
            training_op = np.hstack( (training_op, np.ones( inp_df.shape[0], dtype=int )) )
    for city in ['Guwahati', 'Guwahati', 'Delhi', 'Patna']:
        df = data.read_data(city)
        inp_df, _ = df, df #data.input_output_split(df)
        training_df = training_df.append(inp_df, ignore_index=True)
        training_op = np.hstack( (training_op, np.zeros( inp_df.shape[0], dtype=int )) )
    
    testing_df = data.read_data(test_city)
    # testing_df, _ = testing_df, testing_df# data.input_output_split(testing_df) 

    gnb = naive_bayes.GaussianNB()
    print ("Fitting GNB model for ", test_city)
    gnb.fit( training_df, training_op )

    print ("Predicting values")
    prob_pred = gnb.predict_proba( testing_df )

    with open(config.output_directory + savefile, "w+") as f:
        metrics.print_write( header + "\n" + str( np.mean( prob_pred[:, 1] )  ) + "\n", f )

def logistic_regression_model(test_city, header="", savefile="log_regression_model.txt"):
    """ Logistic Regression Model for predicting confidence as a solar power location"""
    training_df, training_op = [], []
    for city in config.solar_power_cities + ['Ahemdabad', 'Mithapur']:
        df = data.read_data(city)
        inp_df, _ = df, df #data.input_output_split(df)
        if len(training_df) == 0:
            training_df = inp_df
            training_op = np.ones( inp_df.shape[0], dtype=int )
        else:
            training_df = training_df.append(inp_df, ignore_index=True)
            training_op = np.hstack( (training_op, np.ones( inp_df.shape[0], dtype=int )) )
    for city in ['Guwahati', 'Guwahati', 'Guwahati', 'Delhi', 'Patna']:
        df = data.read_data(city)
        inp_df, _ = df, df #data.input_output_split(df)
        training_df = training_df.append(inp_df, ignore_index=True)
        training_op = np.hstack( (training_op, np.zeros( inp_df.shape[0], dtype=int )) )

    testing_df = data.read_data(test_city)

    log_regr = linear_model.LogisticRegression(C=1e5)
    print ("Fitting logistic regression for ", test_city)
    log_regr.fit( training_df, training_op )

    print ("Predicting values")
    prob_pred = log_regr.predict_proba( testing_df )

    with open(config.output_directory + savefile, "w+") as f:
        metrics.print_write( header + "\n" + str( np.mean( prob_pred[:, 1] )  ) + "\n", f )

if __name__ == "__main__":
	
	for city in config.solar_power_cities:
        preprocess(city)
		baseline_comparison(city, header="Baseline viability comparison for " + city, savefile="training_baseline_comparison.txt")
		preprocess(city)

	extract_dni_stats()

	for city in config.solar_test_cities:
		data.preprocess(city)
		baseline_comparison(city, header="Baseline viability comparison for " + city, savefile=city + "_baseline_comparison.txt")
		gnb_model(city, header="GNB Confidence Prediction for " + city, savefile=city + "_gnb_comparison.txt")
		logistic_regression_model(city, header="Logistic Regression Confidence Prediction for " + city, savefile=city + "_logistic_comparison.txt")