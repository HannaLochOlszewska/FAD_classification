import os
from enum import Enum
import pandas as pd
from _05_characteristics import Characteristic, CharacteristicTwo, CharacteristicThree
import multiprocessing as mp
from itertools import repeat

class CharSet(Enum):
    """Enum of diffusion type."""
    One = 1
    Two = 2
    Three = 3

def get_characteristics(char_set, path_to_file, dt, typ="", motion=""):
    data = pd.read_csv(path_to_file)
    x = data['x'].values
    y = data['y'].values
    if char_set == CharSet.One:
        ch = Characteristic(x=x, y=y, dt=dt, percentage_max_n=1, typ=typ, motion=motion, file=path_to_file)
    elif char_set == CharSet.Two:
        ch = CharacteristicTwo(x=x, y=y, dt=dt, percentage_max_n=0.1, typ=typ, motion=motion, file=path_to_file)
    elif char_set == CharSet.Three:
        ch = CharacteristicThree(x=x, y=y, dt=dt, percentage_max_n=0.1, typ=typ, motion=motion, file=path_to_file)
    data = ch.data
    return data

def get_characteristics_single(initial_data, path_to_trajectories, trajectory, char_set):
    """
    :param initial_data: dataframe, info about all generated data
    :param path_to_trajectories: str, path to folder with trajectories
    :param trajectory: str, trajectory name
    :return: dataframe with characteristics
    """
    print(trajectory)
    typ = initial_data[initial_data.path == trajectory]["diff_type"].values[0]
    motion = initial_data[initial_data.path == trajectory]["process"].values[0] + "_" + typ
    dt = initial_data[initial_data.path == trajectory]["dt"].values[0]
    d = get_characteristics(char_set, path_to_file=os.path.join(path_to_trajectories, trajectory), 
                            dt=dt, typ=typ, motion=motion)
    return d


def generate_characteristics(simulation_folder, test_version='', data_type="Synthetic data", char_set=CharSet.One):
    """
    Function for generating the characteristics file for given scenario
    - characteristics are needed for featured based classifiers
    """

    project_directory = os.path.dirname(os.getcwd())
    path_to_save = os.path.join(project_directory, "Data", data_type)

    path_to_trajectories = os.path.join(path_to_save, simulation_folder, "Trajectories")
    path_to_data = os.path.join(path_to_save, simulation_folder, "Trajectories_Stats")
    path_to_characteristics_data = os.path.join(path_to_save, simulation_folder, "Characteristics"+test_version)
    if not os.path.exists(path_to_characteristics_data):
        os.makedirs(path_to_characteristics_data)
    trajectories_lists = [file for file in os.listdir(path_to_trajectories)]
    initial_data = pd.read_csv(os.path.join(path_to_data, "all_data.csv"))
    
    characteristics_input = zip(repeat(initial_data), repeat(path_to_trajectories), trajectories_lists, repeat(char_set))
    pool = mp.Pool(processes=(mp.cpu_count() - 1))
    characteristics_data = pool.starmap(get_characteristics_single, characteristics_input)
    pool.close()                                                   
    pool.join()
    results = pd.concat(characteristics_data)
    results.to_csv(os.path.join(path_to_characteristics_data, "characteristics.csv"), index=False)

if __name__ == "__main__":
    
    generate_characteristics(simulation_folder="Zero_free", test_version='_sta_10', char_set=CharSet.Three)
