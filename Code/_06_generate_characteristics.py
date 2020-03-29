import os
import pandas as pd
from _05_characteristics import Characteristic, CharacteristicTwo

def get_characteristics(path_to_file, typ="", motion=""):
    data = pd.read_csv(path_to_file)
    x = data['x'].values
    y = data['y'].values
    ch = CharacteristicTwo(x=x, y=y, dt=1 / 30, typ=typ, motion=motion, file=path_to_file)
    data = ch.data
    return data

def generate_characteristics(simulation_folder, test_version=''):
    """
    Function for generating the characteristics file for given scenario
    - characteristics are needed for featured based classifiers
    """

    project_directory = os.path.dirname(os.getcwd())
    path_to_save = os.path.join(project_directory, "Data", "Synthetic data")

    path_to_trajectories = os.path.join(path_to_save, simulation_folder, "Trajectories")
    path_to_data = os.path.join(path_to_save, simulation_folder, "Trajectories_Stats")
    path_to_characteristics_data = os.path.join(path_to_save, simulation_folder, "Characteristics"+test_version)
    if not os.path.exists(path_to_characteristics_data):
        os.makedirs(path_to_characteristics_data)
    trajectories_lists = [file for file in os.listdir(path_to_trajectories)]
    initial_data = pd.read_csv(os.path.join(path_to_data, "all_data.csv"))
    characteristics_data = pd.DataFrame([])
    for trajectory in trajectories_lists:
        print(trajectory)
        typ = initial_data[initial_data.path == trajectory]["diff_type"].values[0]
        motion = initial_data[initial_data.path == trajectory]["process"].values[0] + "_" + typ
        d = get_characteristics(path_to_file=os.path.join(path_to_trajectories, trajectory), typ=typ, motion=motion)
        characteristics_data = pd.concat([characteristics_data, d], sort=False)
    characteristics_data.to_csv(os.path.join(path_to_characteristics_data, "characteristics.csv"), index=False)


if __name__ == "__main__":

    generate_characteristics(simulation_folder="Smaller", test_version='_best_old')
    generate_characteristics(simulation_folder="Base", test_version='_best_old')
