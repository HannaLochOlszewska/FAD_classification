import os
from enum import Enum
from numpy import finfo
import pandas as pd
from _02_generators import generate_fbm, generate_ou, generate_directed

EPS = finfo(float).eps

class DiffType(Enum):
    """Enum of diffusion type."""
    Free = "free"
    Sub = "sub"
    Super = "super"

def generate_trajectories(simulation_folder, N=5000):

    """
    Function for generating trajectories datasets
    :param simulation_folder: name of the folder to store results
    :param N: number of trajectories in one (of 6) datasets
    """

    project_directory = os.path.dirname(os.getcwd())
    path_to_save = os.path.join(project_directory, "Data", "Synthetic data")

    #distr_filename = "trajectories_lengths_distribution.npy"
    if not os.path.exists(os.path.join(path_to_save, simulation_folder, "Trajectories_Stats")):
        os.makedirs(os.path.join(path_to_save, simulation_folder, "Trajectories_Stats"))
    if not os.path.exists(os.path.join(path_to_save, simulation_folder, "Trajectories")):
        os.makedirs(os.path.join(path_to_save, simulation_folder, "Trajectories"))

    ### Generate fbm
    c = 0.1
    H_sub = [0.1, 0.5 - c - EPS]
    H_free = [0.5 - c, 0.5 + c]
    H_super = [0.5 + c + EPS, 0.9]

    # Subdiffusion
    generate_fbm(number_of_spt=N, path_to_save=path_to_save, simulation_folder=simulation_folder,
                 length_of_trajectory=0, diff_type=DiffType.Sub.value, H_range=H_sub,
                 is_distribution_of_selection_known=False, distribution_filename="")
    print("Subdiffusion via fBm generated.")

    # Free diffusion
    generate_fbm(number_of_spt=N, path_to_save=path_to_save, simulation_folder=simulation_folder,
                 length_of_trajectory=0, diff_type=DiffType.Free.value, H_range=H_free,
                 is_distribution_of_selection_known=False, distribution_filename="")
    print("Free diffusion via fBm generated.")

    # Superdiffusion
    generate_fbm(number_of_spt=N, path_to_save=path_to_save, simulation_folder=simulation_folder,
                 length_of_trajectory=0, diff_type=DiffType.Super.value, H_range=H_super,
                 is_distribution_of_selection_known=False, distribution_filename="")
    print("Superdiffusion via fBm generated.")

    ### Generate Ornstein-Uhlenbeck
    c = 0.1
    theta = [0, 0]
    lambda_sub = [0 + c + EPS, 1]
    lambda_free = [0, 0 + c]

    # Subdiffusion
    generate_ou(number_of_spt=N, path_to_save=path_to_save, simulation_folder=simulation_folder,
                length_of_trajectory=0, diff_type=DiffType.Sub.value, lambda_range=lambda_sub,
                theta_range=theta, is_distribution_of_selection_known=False, distribution_filename="")
    print("Subdiffusion via OU generated.")

    # Free diffusion
    # HINT: here we generate half of the set
    generate_ou(number_of_spt=int(N/2), path_to_save=path_to_save, simulation_folder=simulation_folder,
                length_of_trajectory=0, diff_type=DiffType.Free.value, lambda_range=lambda_free,
                theta_range=theta, is_distribution_of_selection_known=False, distribution_filename="")
    print("Free diffusion via OU generated.")
    
    ### Generate directed Brownian motion
    c = 0.1    
    # HINT: the range for v parameter is divided for two parts to allow the choice of 
    # superdiffusion parameter around 0 - negative or positive - we pass the list of list for easy choice.
    we_free = [[0 - c, 0 + c]]
    we_super = [[-1, 0 - c - EPS], [0 + c + EPS, 1]]

    # Free diffusion
    # HINT: here we generate half of the set
    generate_directed(number_of_spt=N-int(N/2), path_to_save=path_to_save, simulation_folder=simulation_folder,
                      length_of_trajectory=0, diff_type=DiffType.Free.value, we_range=we_free,
                      is_distribution_of_selection_known=False, distribution_filename="")
    print("Free diffusion via directed Bm generated.")

    # Superdiffusion
    generate_directed(number_of_spt=N, path_to_save=path_to_save, simulation_folder=simulation_folder,
                      length_of_trajectory=0, diff_type=DiffType.Super.value, we_range=we_super,
                      is_distribution_of_selection_known=False, distribution_filename="")
    print("Superdiffusion via directed Bm generated.")


def join_initial_datasets(simulation_folder):
    """
    Function for preparing datasets with stats for different trajectories in one dataset
    :param simulation_folder: name of the folder to store results
    """

    project_directory = os.path.dirname(os.getcwd())
    path_to_save = os.path.join(project_directory, "Data", "Synthetic data")
    path_to_data = os.path.join(path_to_save, simulation_folder, "Trajectories_Stats")
    paths_data_files = [os.path.join(path_to_data, folder) for folder in os.listdir(path_to_data) if
                        folder != "all_data.csv"]
    data = pd.DataFrame([])
    for file in paths_data_files:
        d = pd.read_csv(file, index_col=0)
        data = pd.concat([data, d], sort=False)
        data.reset_index()
    data.to_csv(os.path.join(path_to_data, "all_data.csv"), index=False)



if __name__ == "__main__":

    generate_trajectories(simulation_folder="Base_corr", N=20000)
    join_initial_datasets(simulation_folder="Base_corr")
