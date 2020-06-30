import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

"""
Randomised search for random forest hyperparameters.
"""


def search_hyperparameters(simulation_folder, test_version=''):
    """
    Function for searching best hyperparameters for random forest algorithm
    :param simulation_folder: str, name of subfolder for given set simulation,
    i.e. the different proportion of types of motion or just a subset of motion is considered
    :param test_version: str, name of subsubfolder for given characteristic sets
    """

    Start = datetime.now()
    project_directory = os.path.dirname(os.getcwd())
    path_to_save = os.path.join(project_directory, "Data", "Synthetic data")
    path_to_characteristics_data = os.path.join(path_to_save, simulation_folder, "Characteristics" + test_version)
    path_to_hyperparameters = os.path.join(project_directory, "Models", "RF", simulation_folder, 'Model' + test_version)
    if not os.path.exists(path_to_hyperparameters):
        os.makedirs(path_to_hyperparameters)
    X_train = np.load(os.path.join(path_to_characteristics_data, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(path_to_characteristics_data, "y_train.npy"), allow_pickle=True)

    random_grid_forest = {'n_estimators': [int(x) for x in range(100, 1001, 100)],
                          'criterion': ["gini", "entropy"],
                          'max_features': ['log2', 'sqrt', None],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                          'min_samples_split': [2, 5, 10],
                          'min_samples_leaf': [1, 2, 4],
                          'bootstrap': [True, False]
                          }

    rf = RandomForestClassifier()
    # Random search of parameters, using 10 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid_forest, n_iter=100, cv=10,
                                   verbose=2, random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    print(rf_random.best_params_)
    with open(os.path.join(path_to_hyperparameters, "hyperparameters.json"), 'w') as fp:
        json.dump(rf_random.best_params_, fp)
    End = datetime.now()
    ExecutedTime = End - Start
    df = pd.DataFrame({'ExecutedTime': [ExecutedTime]})
    df.to_csv(os.path.join(path_to_hyperparameters, "time_for_searching.csv"))
    print(ExecutedTime)


if __name__ == "__main__":
    search_hyperparameters(simulation_folder="Base_corr", test_version='_sta_10')
