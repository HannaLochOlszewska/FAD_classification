import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

np.set_printoptions(precision=2)
# TODO: co to za rs?
rs = 42

def generate_model(simulation_folder, featured_model, test_version=""):
    """
    Function for generating model for given scenario and feature based model
    :param featured_model: "RF", "GB" or other id defined
    :return: model
    """

    Start = datetime.now()
    project_directory = os.path.dirname(os.getcwd())
    path_to_data = os.path.join(project_directory, "Data", "Synthetic data")
    path_to_characteristics_data = os.path.join(path_to_data, simulation_folder, 
                                                "Characteristics"+test_version)
    path_to_model = os.path.join(project_directory, "Models", featured_model, 
                                 simulation_folder, "Model"+test_version)
    path_to_hyperparameters = os.path.join(path_to_model, "hyperparameters.json")

    X_train = np.load(os.path.join(path_to_characteristics_data, "X_train.npy"))
    y_train = np.load(os.path.join(path_to_characteristics_data, "y_train.npy"))

    with open(path_to_hyperparameters, 'r') as f:
        param_data = json.load(f)
    if featured_model == "RF":
        model = RandomForestClassifier()
    elif featured_model == "GB":
        model = GradientBoostingClassifier()
    model.set_params(**param_data)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(path_to_model, 'model.sav'))
    End = datetime.now()
    ExecutedTime = End - Start
    df = pd.DataFrame({'ExecutedTime': [ExecutedTime]})
    df.to_csv(os.path.join(path_to_model, "time_for_modelling.csv"))
    print(ExecutedTime)

if __name__ == "__main__":

    generate_model(simulation_folder="Zero_free", featured_model='RF', test_version='_sta_10')
    generate_model(simulation_folder="Zero_free", featured_model='GB', test_version='_sta_10')
