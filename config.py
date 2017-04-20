# Paths to csv directories
directories = {
    'Bhopal': 'data/NREL-Bhopal-2000-2014',
    'Chennai': 'data/NREL-Chennai-2000-2014',
    'Delhi': 'data/NREL-Delhi-2000-2014',
    'Guwahati': 'data/NREL-Guwahati-2000-2014',
    'Mumbai': 'data/NREL-Mumbai-2000-2014',    

    'Charanka': 'data/NREL-Charanka-2000-2014', 
    'Kamuthi': 'data/NREL-Kamuthi-2000-2014', 
    'Neemuch': 'data/NREL-Neemuch-2000-2014', 
    'Rajgarh': 'data/NREL-Rajgarh-2000-2014',
    'Sakri': 'data/NREL-Sakri-2000-2014',

    'Ahemdabad': 'data/NREL-Ahemdabad-2000-2014',
    'Mithapur': 'data/NREL-Phalodi-2000-2014',
    'Phalodi': 'data/NREL-Phalodi-2000-2014',
    'Patna': 'data/NREL-Patna-2000-2014',
}

cities = ['Bhopal', 'Chennai', 'Delhi', 'Guwahati', 'Mumbai']

solar_power_cities = ['Charanka', 'Kamuthi', 'Neemuch', 'Rajgarh', 'Sakri', 'Patna']
solar_test_cities = cities + ['Ahemdabad', 'Mithapur', 'Phalodi']
solar_mean = 0.441375551253
solar_std = 0.310788037987

datasets = {
    'Bhopal': 'data/dataset_Bhopal.pkl',
    'Chennai': 'data/dataset_Chennai.pkl',
    'Delhi': 'data/dataset_Delhi.pkl',
    'Guwahati': 'data/dataset_Guwahati.pkl',
    'Mumbai': 'data/dataset_Mumbai.pkl',   
}

# Features to be dropped while reading csv data
dropped_features = [
    0,      # Year
    4,      # Minute
    8,      # ClearSky DHI
    9,      # Clearsky DNI
    10,     # Clearsky GHI
    17,     # Snow Depth
    18,     # Wind Direction
    20,     # Fill Flag
]

# Night-time hours to be dropped
dropped_hours = [0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23]

# Directory used for storing results
output_directory = "output/"

# Gradient boosting model parameters
gbm = {
    'n_estimators': 1200,
    'learning_rate': 0.1,
    'max_depth': 4,
}
