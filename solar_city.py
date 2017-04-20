import data
import config
import metrics
import scipy.stats
import numpy as np
from sklearn import naive_bayes

def extract_dni_stats(savefile="dni_train_city_values.txt"):
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
	df = data.read_data(city)
	df_mean, df_std = df[['DNI']].mean()[0], df[['DNI']].std()[0]
	mean_prob = scipy.stats.norm(mean, std).pdf( df_mean )
	mean_confidence = 100.*(scipy.stats.norm(mean, std).cdf(df_mean) - scipy.stats.norm(mean, std).cdf(-df_mean))

	kl_div = np.log( df_std / std ) + ( ((std**2) + np.power( (df_mean - mean), 2)) / (2 * np.power(df_std, 2))  ) - 0.5
	
	with open(config.output_directory + savefile, "w+") as f:
		metrics.print_write( header + "\n" + str( (df_mean, df_std, mean_prob, mean_confidence, kl_div )  ) + "\n", f )



if __name__ == "__main__":
	
	# for city in config.solar_power_cities:
	# 	preprocess(city)
	# extract_dni_stats()
	
	for city in config.solar_test_cities:
		data.preprocess(city)
		baseline_comparison(city, header="Baseline viability comparison for " + city, savefile=city + "_baseline_comparison.txt")
