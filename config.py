# Paths to csv directories
directories = {
    'Bhopal': 'data/NREL-Bhopal-2000-2014',
    'Chennai': 'data/NREL-Chennai-2000-2014',
    'Delhi': 'data/NREL-Delhi-2000-2014',
    'Guwahati': 'data/NREL-Guwahati-2000-2014',
    'Mumbai': 'data/NREL-Mumbai-2000-2014',
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
