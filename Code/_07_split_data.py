import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""
Splitting data for train/test set.
"""

def split_data(simulation_folder, test_version=''):
    """
    Function for spliting data into test and train set
    :param simulation_folder: str, name of subfolder for given set simulation,
    i.e. the different proportion of types of motion or just a subset of motion is considered
    :param test_version: str, name of subsubfolder for given characteristic sets
    """

    project_directory = os.path.dirname(os.getcwd())
    path_to_save = os.path.join(project_directory, "Data", "Synthetic data")

    path_to_characteristics_data = os.path.join(path_to_save, simulation_folder, "Characteristics" + test_version)
    file_with_characteristics = os.path.join(path_to_characteristics_data, "characteristics.csv")
    characteristics_data = pd.read_csv(file_with_characteristics)
    characteristics_data = characteristics_data.drop(["file", "motion"], axis=1)
    X = characteristics_data.loc[:, characteristics_data.columns != 'diff_type']
    y = characteristics_data["diff_type"]
    y_for_split = characteristics_data["diff_type"].values
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    np.save(os.path.join(path_to_characteristics_data, 'classes.npy'), labelencoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y_for_split)

    np.save(os.path.join(path_to_characteristics_data, "X_data.npy"), X)
    np.save(os.path.join(path_to_characteristics_data, "y_data.npy"), y)
    np.save(os.path.join(path_to_characteristics_data, "X_train.npy"), X_train)
    np.save(os.path.join(path_to_characteristics_data, "X_test.npy"), X_test)
    np.save(os.path.join(path_to_characteristics_data, "y_train.npy"), y_train)
    np.save(os.path.join(path_to_characteristics_data, "y_test.npy"), y_test)


if __name__ == "__main__":

    split_data(simulation_folder="Base_corr", test_version='_sta_10')
