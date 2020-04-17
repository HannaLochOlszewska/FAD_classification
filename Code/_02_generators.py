import os
import random

import numpy as np
import pandas as pd

from _01_processes import fbm_generator, ou_generator, directed_bm_generator

dt = 1
sigma = 1
N = [50, 500]
noise_sigma = 0

col_for_stats = ["path", "process", "diff_type", "dt", "T", "N", "sigma", "H", "lambda", "theta", "v", "noise_sigma"]

def generate_fbm(number_of_spt, path_to_save, simulation_folder, length_of_trajectory,
                 diff_type, H_range, is_distribution_of_selection_known=False, 
                 distribution_filename=""):
    """
    Function to generate set of the normal trajectories using fractional Brownian motion
    :param number_of_spt: int, number of points in SPT
    :param path_to_save: str, path to destination directory
    :param simulation_folder: str, name of subfolder of given simulation
    :param length_of_trajectory: int, length of trajectory if 0 it randomly choose the number from uniform distribution
    :param diff_type: string, "free", "sub" or "super" - type of diffusion
    :param H_range: list of two elements, with lower and upper bound of H value
    :param is_distribution_of_selection_known: bool, type of distribution eg. should distribution of data be taken from
    distribution of experimental data
    :param distribution_filename: str, name of the numpy file with array of trajectories experimental data lengths
    :return: dataframe, with data of normal motions
    """
    stats_data = pd.DataFrame([], columns=col_for_stats)
    for i in range(number_of_spt):
        H = np.random.uniform(H_range[0], H_range[1])
        path_to_data = os.path.dirname(path_to_save)
        n = get_length_of_trajectory(is_distribution_of_selection_known, length_of_trajectory, 
                                     distribution_filename, path_to_data)
        T = (n - 1) * dt
        x_fbm, y_fbm = fbm_generator(n, H, sigma, dt)
        x_rand, y_rand = generate_noise(length=len(x_fbm), n_sigma=noise_sigma)
        x = x_fbm + x_rand
        y = y_fbm + y_rand
        trajectory_vector = pd.DataFrame({"x": x, "y": y})
        name_of_trajectory_file = diff_type + "_fbm_" + str(i) + ".txt"
        trajectory_vector.to_csv(os.path.join(path_to_save, simulation_folder, "Trajectories", name_of_trajectory_file))
        stats_data_mini = pd.DataFrame(
            [[name_of_trajectory_file, "fbm", diff_type, dt, T, n, sigma, H, None, None, None, noise_sigma]],
            columns=col_for_stats)
        stats_data = pd.concat([stats_data, stats_data_mini], ignore_index=True)
    stats_data.to_csv(os.path.join(path_to_save, simulation_folder, "Trajectories_Stats", diff_type+"_fbm.csv"))


def generate_ou(number_of_spt, path_to_save, simulation_folder, length_of_trajectory,
                diff_type, lambda_range, theta_range, is_distribution_of_selection_known=False, 
                distribution_filename=""):
    """
    Function to generate set of the normal trajectories using fractional Brownian motion
    :param number_of_spt: int, number of points in SPT
    :param path_to_save: str, path to destination directory
    :param simulation_folder: str, name of subfolder of given simulation
    :param length_of_trajectory: int, length of trajectory if 0 it randomly choose the number from uniform distribution
    :param diff_type: string, "free", "sub" or "super" - type of diffusion
    :param lambda_range: list of two elements, with lower and upper bound of lambda value (same in each direction (x,y))
    :param theta_range: list of two lists of two elements, with lower and upper bound of theta value in each direction (x,y)
    :param is_distribution_of_selection_known: bool, type of distribution eg. should distribution of data be taken from
    distribution of experimental data
    :param distribution_filename: str, name of the numpy file with array of trajectories experimental data lengths
    :return: dataframe, with data of normal motions
    """
    stats_data = pd.DataFrame([], columns=col_for_stats)
    for i in range(number_of_spt):
        lambda_x = np.random.uniform(lambda_range[0], lambda_range[1])
        lambda_y = np.random.uniform(lambda_range[0], lambda_range[1])
        theta_x = np.random.uniform(theta_range[0], theta_range[1])
        theta_y = np.random.uniform(theta_range[0], theta_range[1])
        path_to_data = os.path.dirname(path_to_save)
        n = get_length_of_trajectory(is_distribution_of_selection_known, length_of_trajectory, 
                                     distribution_filename, path_to_data)
        T = (n - 1) * dt
        x_ou, y_ou = ou_generator(n, (lambda_x, lambda_y), (theta_x, theta_y), sigma, dt)
        x_rand, y_rand = generate_noise(length=len(x_ou), n_sigma=noise_sigma)
        x = x_ou + x_rand
        y = y_ou + y_rand
        trajectory_vector = pd.DataFrame({"x": x, "y": y})
        name_of_trajectory_file = diff_type+"_ou_" + str(i) + ".txt"
        trajectory_vector.to_csv(os.path.join(path_to_save, simulation_folder, 
                                              "Trajectories", name_of_trajectory_file))
        stats_data_mini = pd.DataFrame([[name_of_trajectory_file, "ou", diff_type, dt, T, 
                                         n, sigma, None, (lambda_x, lambda_y), (theta_x, theta_y), None, noise_sigma]],
            columns=col_for_stats)
        stats_data = pd.concat([stats_data, stats_data_mini], ignore_index=True)
    stats_data.to_csv(os.path.join(path_to_save, simulation_folder, "Trajectories_Stats", 
                                   diff_type+"_ou.csv"))


def generate_directed(number_of_spt, path_to_save, simulation_folder, length_of_trajectory,
                      diff_type, we_range, is_distribution_of_selection_known=False, 
                      distribution_filename=""):
    """
    Function to generate set of the normal trajectories using fractional Brownian motion
    :param number_of_spt: int, number of points in SPT
    :param path_to_save: str, path to destination directory
    :param simulation_folder: str, name of subfolder of given simulation
    :param length_of_trajectory: int, length of trajectory if 0 it randomly choose the number from uniform distribution
    :param diff_type: string, "free", "sub" or "super" - type of diffusion
    :param we_range: list of two elements, with lower and upper bound of we value (same in each direction (x,y))
    :param is_distribution_of_selection_known: bool, type of distribution eg. should distribution of data be taken from
    distribution of experimental data
    :param distribution_filename: str, name of the numpy file with array of trajectories experimental data lengths
    :return: dataframe, with data of normal motions
    """
    stats_data = pd.DataFrame([], columns=col_for_stats)
    for i in range(number_of_spt):
        we_x_range = np.random.randint(0,len(we_range)) # 0 or 0,1
        we_x = np.random.uniform(we_range[we_x_range][0], we_range[we_x_range][1])
        we_y_range = np.random.randint(0,len(we_range)) # 0 or 0,1
        we_y = np.random.uniform(we_range[we_y_range][0], we_range[we_y_range][1])
        path_to_data = os.path.dirname(path_to_save)
        n = get_length_of_trajectory(is_distribution_of_selection_known, length_of_trajectory, 
                                     distribution_filename, path_to_data)
        T = (n - 1) * dt
        x_dir, y_dir = directed_bm_generator(n, (we_x, we_y), sigma, dt)
        x_rand, y_rand = generate_noise(length=len(x_dir), n_sigma=noise_sigma)
        x = x_dir + x_rand
        y = y_dir + y_rand
        trajectory_vector = pd.DataFrame({"x": x, "y": y})
        name_of_trajectory_file = diff_type+"_directed_" + str(i) + ".txt"
        trajectory_vector.to_csv(os.path.join(path_to_save, simulation_folder, 
                                              "Trajectories", name_of_trajectory_file))
        stats_data_mini = pd.DataFrame([[name_of_trajectory_file, "directed", diff_type, 
                                         dt, T, n, sigma, None, None, None, (we_x, we_y), noise_sigma]],
            columns=col_for_stats)
        stats_data = pd.concat([stats_data, stats_data_mini], ignore_index=True)
    stats_data.to_csv(os.path.join(path_to_save, simulation_folder, "Trajectories_Stats", 
                                   diff_type+"_directed.csv"))


def get_length_of_trajectory(is_distribution_of_selection_known, length_of_trajectory, 
                             distribution_filename, path_to_data):
    """
    Function gives the number with the length of generated trajectory
    :param is_distribution_of_selection_known: bool, type of distribution eg. should distribution of data be taken from
    distribution of experimental data
    :param length_of_trajectory: int, length of trajectory if 0 it randomly choose the number from uniform distribution
    :param distribution_filename: str, name of the numpy file with array of trajectories experimental data lengths
    :return: int, with the length of trajectory
    """
    if is_distribution_of_selection_known:
        my_dis = np.load(os.path.join(path_to_data, "Experimental data", "Data distributions", distribution_filename))
        n = random.choice(my_dis)
    else:
        if bool(length_of_trajectory):
            n = length_of_trajectory
        else:
            n = random.randint(N[0], N[1])
    return n


def generate_noise(length, n_sigma=1.0):
    """
    Function for generating vector of noises for simulated trajectories
    :param length: int, length of pure trajectory
    :param sigma: double, the std of noise
    :return: lists of noises
    """
    x_rand = np.random.normal(0, n_sigma, length)
    y_rand = np.random.normal(0, n_sigma, length)
    return x_rand, y_rand
