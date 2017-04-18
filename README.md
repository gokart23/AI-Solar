## Solar energy prediction using weather data

We intend to predict solar energy based on data from five cities:
Bhopal, Chennai, Delhi, Guwahati and Mumbai.

### Dependencies

We recommend using Python 3.5 or above. The PyPI dependencies are noted
in requirements.txt.

In addition, to run stationarity test in stats.py,
statsmodels requires numpy+mkl for which the wheel can be downloaded
[here](www.lfd.uci.edu/~gohlke/pythonlibs/#numpy). This package is a
replacement for numpy. So, previous versions of numpy should be
uninstalled before installing numpy+mkl.

The requirements `bottleneck` and `numexpr` are optional and are
recommended by pandas for achieving calculation speedups.

### Data

We used NREL database which can be accessed [here](https://maps.nrel.gov/nsrdb-viewer/#/?aL=UdPEX9%255Bv%255D%3Dt%268VWYIh%255Bv%255D%3Dt%268VWYIh%255Bd%255D%3D1&bL=groad&cE=0&lR=0&mC=21.820707853875017%2C92.28515625&zL=4).
To download,  select 'Point' under 'Download Data' section and put
a marker on required city in the map. We used data of all features from
2000-14.

### Configuration

In config.py, one can specify paths to data for different cities.
We noted our results for pre-processing methods under output folder.

Run `python data.py` to generate test, validation and train dataset from raw_data.
Run `python linear_models.py` to get baseline results from linear regression model.
