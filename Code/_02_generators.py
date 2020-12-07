import os
import random
import numpy as np
import pandas as pd

from _01_processes import fbm_generator, ou_generator, directed_bm_generator

"""This file contains generators of the specific motion sets. 
Each function saves trajectories and initial conditions as well as randomly selected parameters to the files"""
                                                            
dt = 1
sigma = 1  # np.sqrt(2*D)
N = [50, 500]

col_for_stats = ["path", "process", "diff_type", "dt", "T", "N", "sigma", "H", "lambda",
                   "theta", "v", "noise_sigma", "snr"]

def generate_fbm(number_of_spt, path_to_save, simulation_folder, length_of_trajectory,
                 diff_type, H_range, add_nonzero_noise, is_distribution_of_selection_known=False,
                 distribution_filename=""):
    """
    Function to generate set of the normal trajectories using fractional Brownian motion
    :param number_of_spt: int, number of points in SPT
    :param path_to_save: str, path to folder with synthetic data and its information
    :param simulation_folder: str, name of subfolder for given set simulation
    :param length_of_trajectory: int, length of trajectory
    if 0 it randomly choose the number from uniform distribution with parameters defined in the top of this file (N list)
    :param diff_type: string, "free", "sub" or "super" - type of diffusion
    :param H_range: list of two elements, with lower and upper bound of H value
    :param add_nonzero_noise: bool, information if noise should have randomly chosen sigma (not sigma=0)
    :param is_distribution_of_selection_known: bool, type of distribution
    eg. should distribution of data be taken from distribution of experimental data
    :param distribution_filename: str, name of the numpy file with array of experimental trajectories lengths
    :return: dataframe, with data of normal motions
    """
    stats_data = pd.DataFrame([], columns=col_for_stats)
    for i in range(number_of_spt):
        H = np.random.uniform(H_range[0], H_range[1])
        path_to_data = os.path.dirname(path_to_save)
        n = get_length_of_trajectory(is_distribution_of_selection_known, length_of_trajectory,
                                     distribution_filename, path_to_data)
        T = (n - 1) * dt
        #sigma = np.sqrt(2*np.random.uniform(1,9))
        x_fbm, y_fbm = fbm_generator(n, H, sigma, dt)
        x_rand, y_rand, noise_sigma, snr = generate_noise(length=len(x_fbm), D_sigma = sigma,
                                                          add_nonzero_noise=add_nonzero_noise)
        x = x_fbm + x_rand
        y = y_fbm + y_rand
        trajectory_vector = pd.DataFrame({"x": x, "y": y})
        name_of_trajectory_file = diff_type + "_fbm_" + str(i) + ".txt"
        trajectory_vector.to_csv(os.path.join(path_to_save, simulation_folder, "Trajectories", name_of_trajectory_file))
        stats_data_mini = pd.DataFrame(
            [[name_of_trajectory_file, "fbm", diff_type, dt, T, n, sigma, H, None, None, None, noise_sigma, snr]],
            columns=col_for_stats)
        stats_data = pd.concat([stats_data, stats_data_mini], ignore_index=True)
    stats_data.to_csv(os.path.join(path_to_save, simulation_folder, "Trajectories_Stats", diff_type + "_fbm.csv"))


def generate_ou(number_of_spt, path_to_save, simulation_folder, length_of_trajectory,
                diff_type, lambda_range, theta_range, add_nonzero_noise,
                is_distribution_of_selection_known=False, distribution_filename=""):
    """
    Function to generate set of the normal trajectories using fractional Brownian motion
    :param number_of_spt: int, number of points in SPT
    :param path_to_save: str, path to folder with synthetic data and its information
    :param simulation_folder: str, name of subfolder for given set simulation
    :param length_of_trajectory: int, length of trajectory
    if 0 it randomly choose the number from uniform distribution with parameters defined in the top of this file (N list)
    :param diff_type: string, "free", "sub" or "super" - type of diffusion
    :param lambda_range: list of two elements, with lower and upper bound of lambda value (same in each direction (x,y))
    :param theta_range: list of two lists of two elements, with lower and upper bound of theta value in each direction (x,y)
    :param add_nonzero_noise: bool, information if noise should have randomly chosen sigma (not sigma=0)
    :param is_distribution_of_selection_known: bool, type of distribution
    eg. should distribution of data be taken from distribution of experimental data
    :param distribution_filename: str, name of the numpy file with array of experimental trajectories lengths
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
        #sigma = np.sqrt(2*np.random.uniform(1,9))
        x_ou, y_ou = ou_generator(n, (lambda_x, lambda_y), (theta_x, theta_y), sigma, dt)
        x_rand, y_rand, noise_sigma, snr = generate_noise(length=len(x_ou), D_sigma = sigma, add_nonzero_noise=add_nonzero_noise)
        x = x_ou + x_rand
        y = y_ou + y_rand
        trajectory_vector = pd.DataFrame({"x": x, "y": y})
        name_of_trajectory_file = diff_type + "_ou_" + str(i) + ".txt"
        trajectory_vector.to_csv(os.path.join(path_to_save, simulation_folder,
                                              "Trajectories", name_of_trajectory_file))
        stats_data_mini = pd.DataFrame([[name_of_trajectory_file, "ou", diff_type, dt, T,
                                         n, sigma, None, (lambda_x, lambda_y), (theta_x, theta_y), None, noise_sigma, snr]],
                                         columns=col_for_stats)
        stats_data = pd.concat([stats_data, stats_data_mini], ignore_index=True)
    stats_data.to_csv(os.path.join(path_to_save, simulation_folder, "Trajectories_Stats",
                                   diff_type + "_ou.csv"))


def generate_directed(number_of_spt, path_to_save, simulation_folder, length_of_trajectory,
                      diff_type, we_range, add_nonzero_noise, is_distribution_of_selection_known=False,
                      distribution_filename=""):
    """
    Function to generate set of the normal trajectories using fractional Brownian motion
    :param number_of_spt: int, number of points in SPT
    :param path_to_save: str, path to folder with synthetic data and its information
    :param simulation_folder: str, name of subfolder for given set simulation
    :param length_of_trajectory: int, length of trajectory
    if 0 it randomly choose the number from uniform distribution with parameters defined in the top of this file (N list)
    :param diff_type: string, "free", "sub" or "super" - type of diffusion
    :param we_range: list of two elements, with lower and upper bound of we value (same in each direction (x,y))
    :param add_nonzero_noise: bool, information if noise should have randomly chosen sigma (not sigma=0)
    eg. should distribution of data be taken from distribution of experimental data
    :param distribution_filename: str, name of the numpy file with array of experimental trajectories lengths
    :return: dataframe, with data of normal motions
    """
    stats_data = pd.DataFrame([], columns=col_for_stats)
    for i in range(number_of_spt):
        we_x_range = np.random.randint(0, len(we_range))  # 0 or 0,1
        we_x = np.random.uniform(we_range[we_x_range][0], we_range[we_x_range][1])
        we_y_range = np.random.randint(0, len(we_range))  # 0 or 0,1
        we_y = np.random.uniform(we_range[we_y_range][0], we_range[we_y_range][1])
        path_to_data = os.path.dirname(path_to_save)
        n = get_length_of_trajectory(is_distribution_of_selection_known, length_of_trajectory,
                                     distribution_filename, path_to_data)
        T = (n - 1) * dt
        #sigma = np.sqrt(2*np.random.uniform(1,9))
        x_dir, y_dir = directed_bm_generator(n, (we_x, we_y), sigma, dt)
        v = np.sqrt(we_x ** 2 + we_y ** 2)
        x_rand, y_rand, noise_sigma, snr = generate_noise(length=len(x_dir), D_sigma = sigma, add_nonzero_noise=add_nonzero_noise,
                                                          is_direct_motion=True, v=v)
        x = x_dir + x_rand
        y = y_dir + y_rand
        trajectory_vector = pd.DataFrame({"x": x, "y": y})
        name_of_trajectory_file = diff_type + "_directed_" + str(i) + ".txt"
        trajectory_vector.to_csv(os.path.join(path_to_save, simulation_folder,
                                              "Trajectories", name_of_trajectory_file))
        stats_data_mini = pd.DataFrame([[name_of_trajectory_file, "directed", diff_type,
                                         dt, T, n, sigma, None, None, None, (we_x, we_y), noise_sigma, snr]],
                                         columns=col_for_stats)
        stats_data = pd.concat([stats_data, stats_data_mini], ignore_index=True)
    stats_data.to_csv(os.path.join(path_to_save, simulation_folder, "Trajectories_Stats",
                                   diff_type + "_directed.csv"))


def get_length_of_trajectory(is_distribution_of_selection_known, length_of_trajectory, 
                             distribution_filename, path_to_data):
    """
    Function gives the number of the generated trajectory length
    :param is_distribution_of_selection_known: bool, type of distribution
    eg. should distribution of data be taken from distribution of experimental data
    :param length_of_trajectory: int, length of trajectory
    if 0 it randomly choose the number from uniform distribution
    :param distribution_filename: str, name of the numpy file with array of experimental trajectories lengths
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


def generate_noise(length, D_sigma, add_nonzero_noise=True, is_direct_motion=False, v=0):
    """
    Function for generating vector of noises for simulated trajectories
    :param length: int, length of pure trajectory
    :param add_nonzero_noise: bool, information if noise should have randomly chosen sigma (not sigma=0)
    :param is_direct_motion: bool, information if noise is added to direct motion
    :param v, velocity for direct motion used in generating of sigma in randomly chosen noise scenario
    :return: 2D vector of noises, sigma noise and SNR parameters
    """
    SNR = [1, 9]
    snr = np.random.uniform(SNR[0], SNR[1])
    D = D_sigma ** 2 / 2
    if add_nonzero_noise:
        if not is_direct_motion:
            n_sigma = np.sqrt(D * dt) / snr
        else:
            n_sigma = np.sqrt(D * dt + v ** 2 * dt ** 2) / snr
    else:
        n_sigma = 0
    x_rand = np.random.normal(0, n_sigma, length)
    y_rand = np.random.normal(0, n_sigma, length)
    return x_rand, y_rand, n_sigma, snr
