import config
import numpy as np
import sklearn.metrics as metrics


def generate_results(city, y_val, y_test, pred_val, pred_test, savefile, heading, skip_save=False):
    """Calculate rmse and r2 score for validation and testing"""
    op_path = config.output_directory + savefile
    op_path += '-' + city + '.txt' if city else '.txt'
    op_file = open(op_path, "w+")
    op_file.write(heading + '\n\n')

    # The root-mean-squared error (RMSE) on validation data
    rmse_val = np.sqrt(metrics.mean_squared_error(y_val, pred_val))
    if skip_save == True:
        print("Mean squared error(validation): " + str(rmse_val), op_file)
    else:
        print_write("Mean squared error(validation): " + str(rmse_val), op_file)

    # R2 coefficient on validation data: 1 is perfect prediction
    r2_val = metrics.r2_score(y_val, pred_val, multioutput='variance_weighted')
    if skip_save == True:
        print("R2 score(validation): " + str(r2_val), op_file)
    else:
        print_write("R2 score(validation): " + str(r2_val), op_file)

    # The root-mean-squared error (RMSE) on test data
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, pred_test))
    if skip_save == True:
        print("Mean squared error(test): " + str(rmse_test), op_file)
    else:
        print_write("Mean squared error(test): " + str(rmse_test), op_file)

    # R2 coefficient on test data: 1 is perfect prediction
    r2_test = metrics.r2_score(y_test, pred_test, multioutput='variance_weighted')
    if skip_save == True:
        print('R2 score(test): ' + str(r2_test), op_file)
    else:
        print_write('R2 score(test): ' + str(r2_test), op_file)

    op_file.close()

    return rmse_val, r2_val, rmse_test, r2_test

def get_model_results(model):
    """ Function to run the specified model on all default parameters on global and local datasets"""
    model()
    for city_ in config.cities:
        model(city_)


def print_write(line, f):
    """Print line and write line to file"""
    print(line)
    f.write(line + '\n')

