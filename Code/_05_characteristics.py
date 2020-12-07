import math
import numpy as np
import pandas as pd
from numpy import log, mean, sqrt, where, std, exp, sign
from scipy import linalg as LA
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from _04_msd import generate_theoretical_msd_normal, generate_empirical_msd, \
                    generate_theoretical_msd_anomalous_log, generate_empirical_pvariation, \
                    generate_empirical_velocity_autocorrelation
                    
"""
Classes of characteristics sets.
"""

class CharacteristicBase:
    """
    Class representing base characteristics of given trajectory, based on T. Wagner et al.
    "Classification and Segmentation of Nanoparticle Diffusion Trajectories in Cellular Micro Environments"
    PLoS ONE 12(1), (2017).    
    """

    def __init__(self, x, y, dt, percentage_max_n, typ="", motion="", file=""):
        """
        :param x: list, x coordinates
        :param y: list, y coordinates
        :param dt: float, time between steps
        :param typ: str, type of diffusion i.e sub, super, rand
        :param motion: str, mode of diffusion eg. normal, directed
        :param file: str, path to trajectory
        :param percentage_max_n: float, percentage of length of the trajectory for msd generating
        """
        self.x = x
        self.y = y
        self.dt = dt
        self.percentage_max_n = percentage_max_n
        self.type = typ
        self.motion = motion
        self.file = file

        self.N = self.get_length_of_trajectory()
        self.T = self.get_duration_of_trajectory()
        self.max_number_of_points_in_msd = self.get_max_number_of_points_in_msd()
        self.n_list = self.get_range_for_msd()
        self.empirical_msd = generate_empirical_msd(self.x, self.y, self.n_list)
        self.displacements = self.get_displacements()
        self.d = self.get_max_displacement()
        self.L = self.get_total_length_of_path()

    def get_length_of_trajectory(self):
        """
        :return: int, length of trajectory represented by N parameter
        """
        return len(self.x)

    def get_duration_of_trajectory(self):
        """
        :return: int, duration of the trajectory life represented by T parameter
        """
        return int((self.N - 1) * self.dt)

    def get_max_number_of_points_in_msd(self):
        """
        :return: int, maximal number which can be used to generate msd
        """
        if self.percentage_max_n != None:
            return math.floor(self.percentage_max_n * self.N)
        else:
            return self.N if self.N <= 100 else 101

    def get_range_for_msd(self):
        """
        :return: array, range of steps in msd function
        """
        return np.array(range(1, self.max_number_of_points_in_msd))

    def get_displacements(self):
        """
        :return: array, list of displacements between x and y coordinates
        """
        return np.array(
            [self.get_displacement(self.x[i], self.y[i], self.x[i - 1], self.y[i - 1]) for i in range(1, self.N - 1)])

    @staticmethod
    def get_displacement(x1, y1, x2, y2):
        """
        :param x1: float, first x coordinate
        :param y1: float, first y coordinate
        :param x2: float, second x coordinate
        :param y2: float, second y coordinate
        :return: float, displacement between two points
        """
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_total_length_of_path(self):
        """
        :return: int, total length of path represented by L parameter
        """
        return sum(self.displacements)

    def get_max_displacement(self):
        """
        :return: float, maximum displacement represented by d in all displacement list
        """
        return max(self.displacements)


class Characteristic(CharacteristicBase):
    """
    Class representing characteristics of given trajectory
    """

    def __init__(self, x, y, dt, percentage_max_n=1, typ="", motion="", file=""):
        """
        :param x: list, x coordinates
        :param y: list, y coordinates
        :param dt: float, time between steps
        :param typ: str, type of diffusion i.e sub, super, rand
        :param motion: str, mode of diffusion eg. normal, directed
        :param file: str, path to trajectory
        :param percentage_max_n: float, percentage of length of the trajectory for msd generating
        """

        CharacteristicBase.__init__(self,x, y, dt, percentage_max_n, typ, motion, file)

        self.D = self.get_diffusion_coef()
        self.alpha = self.get_exponent_alpha()
        self.radius_gyration_tensor = self.get_tensor()
        self.eigenvalues, self.eigenvectors = LA.eig(self.radius_gyration_tensor)
        self.asymmetry = self.get_asymmetry()
        self.efficiency = self.get_efficiency()
        self.trappedness = self.get_trappedness()
        self.diff_kurtosis = self.get_kurtosis_corrected()
        self.fractal_dimension = self.get_fractal_dimension()
        self.gaussianity = self.get_gaussianity()
        self.mean_gaussianity = self.get_mean_gaussianity()
        self.spec_gaussianity = self.get_point_of_gaussianity()
        self.mean_squared_displacement_ratio = self.get_mean_squared_displacement_ratio()
        self.straightness = self.get_straightness()
        self.values = [self.file, self.type, self.motion, self.D, self.alpha, self.asymmetry, self.efficiency,
                       self.fractal_dimension, self.mean_gaussianity, self.diff_kurtosis,
                       self.mean_squared_displacement_ratio, self.straightness, self.trappedness]
        self.columns = ["file", "diff_type", "motion", "D", "alpha", "asymmetry", "efficiency", "fractal_dimension",
                        "mean_gaussianity", "diff_kurtosis", "mean_squared_displacement_ratio", "straightness",
                        "trappedness"]
        self.data = pd.DataFrame([self.values], columns=self.columns)

    def get_diffusion_coef(self):
        """
        :return: float, diffusion coefficient represented by D parameter;
        estimated based on curve fitting of empirical and normal theoretical diffusion.
        """
        popt, _ = curve_fit(lambda x, d: generate_theoretical_msd_normal(x, d, self.dt), self.n_list,
                            self.empirical_msd)
        D = popt[0]
        return D

    def get_exponent_alpha(self):
        """
        :return: float, exponential anomalous parameter by alpha parameter;
        estimated based on curve fitting of empirical and normal anomalous diffusion.
        Modification of this function can also estimate D parameter
        """
        popt, _ = curve_fit(
            lambda x, log_D, a: generate_theoretical_msd_anomalous_log(log(self.dt * self.n_list), log_D, a),
            log(self.dt * self.n_list), log(self.empirical_msd), bounds=((-np.inf, 0), (np.inf, 2)))
        alpha = popt[1]
        return alpha

    def get_tensor(self):
        """
        :return: matrix, the tensor T for given trajectory
        """
        a = sum((self.x - mean(self.x)) ** 2) / len(self.x)
        c = sum((self.y - mean(self.y)) ** 2) / len(self.y)
        b = sum((self.x - mean(self.x)) * (self.y - mean(self.y))) / len(self.x)
        return np.array([[a, b], [b, c]])

    def get_asymmetry(self):
        """
        The asymmetry of a trajectory can be used to detect directed motion.
        :return: float, asymmetry parameter - only real part of
        """
        lambda1 = self.eigenvalues[0]
        lambda2 = self.eigenvalues[1]
        a = -1 * log(1 - (lambda1 - lambda2) ** 2 / (2 * (lambda1 + lambda2) ** 2))
        return a.real

    def get_efficiency(self):
        """
        Efficiency relates the net squared displacement of a particle to the sum of squared step lengths
        :return: float, efficiency parameter
        """
        upper = self.get_displacement(self.x[self.N - 2], self.y[self.N - 2], self.x[0], self.y[0]) ** 2
        displacements_to_squere = self.displacements ** 2
        lower = (self.N - 1) * sum(displacements_to_squere)
        E = upper / lower
        return E

    def get_trappedness(self, n=3):
        """
        Trappedness is the probability that a diffusing particle with the diffusion coefficient D
        and traced for a time interval t is trapped in a bounded region with radius r0.
        :param n: int, given point of trappedness
        :return: float, probability of trappedness in point n
        """
        t = self.n_list * self.dt
        popt, _ = curve_fit(lambda x, d: generate_theoretical_msd_normal(self.n_list[:2], d, self.dt),
                            self.n_list[:2], self.empirical_msd[:2])
        d = popt[0]
        p = 1 - exp(0.2048 - 0.25117 * ((d * t) / (self.d / 2) ** 2))
        p = np.array([i if i > 0 else 0 for i in p])[n]
        return p

    def get_kurtosis_corrected(self):
        """
        Kurtosis measures the asymmetry and peakedness of the distribution of points within a trajectory
        :return: float, kurtosis for trajectory
        """
        index = where(self.eigenvalues == max(self.eigenvalues))[0][0]
        dominant_eigenvector = self.eigenvectors[index]
        a_prod_b = np.array([sum(np.array([self.x[i], self.y[i]]) * dominant_eigenvector) for i in range(len(self.x))])
        K = 1 / self.N * sum((a_prod_b - mean(a_prod_b)) ** 4 / std(a_prod_b) ** 4) - 3
        return K

    def get_fractal_dimension(self):
        """
        The fractal dimension is a measure of the space-filling capacity of a pattern.
        :return: float, fractional dimension parameter
        """
        upper = log(self.N)
        lower = log(self.N * self.L ** (-1) * self.d)
        D = upper / lower
        return D

    def get_gaussianity(self):
        """
        A trajectory’s Gaussianity checks the Gaussian statistics on increments
        :return: array, list of gaussianity points
        """
        r4 = generate_empirical_msd(self.x, self.y, self.n_list, 4)
        r2 = generate_empirical_msd(self.x, self.y, self.n_list, 2)
        g = r4 / (2 * r2 ** 2)
        g = -1 + 2 * r4 / (3 * r2 ** 2)
        return g

    def get_mean_gaussianity(self):
        """
        :return: float, mean of gaussianity points
        """
        return mean(self.gaussianity)

    def get_point_of_gaussianity(self, n=3):
        """
        :param n: int, point
        :return: float, point in gaussianity lists of points
        """
        return self.gaussianity[n]

    def get_mean_squared_displacement_ratio(self):
        """
        The mean square displacement ratio characterizes the shape of the MSD curve.
        :return: float, mean squared displacement ratio parameter
        """
        n1 = np.array(range(1, self.max_number_of_points_in_msd - 1))
        n2 = np.array(range(2, self.max_number_of_points_in_msd))
        r_n1 = self.empirical_msd[0:self.max_number_of_points_in_msd - 2]
        r_n2 = self.empirical_msd[1:self.max_number_of_points_in_msd]
        r = mean(r_n1 / r_n2 - n1 / n2)
        return r

    def get_straightness(self):
        """
        Straightness is a measure of the average direction change between subsequent steps.
        :return: float, straing
        """
        upper = self.get_displacement(self.x[self.N - 2], self.y[self.N - 2], self.x[0], self.y[0])
        displacements = np.array(
            [self.get_displacement(self.x[i], self.y[i], self.x[i - 1], self.y[i - 1]) for i in range(1, self.N - 1)])
        lower = sum(displacements)
        S = upper / lower
        return S


class CharacteristicThree(Characteristic):
    """
    Characteristics corresponding to the article of A. Weron et al.
    "Statistical testing approach for fractional anomalous diffusion classification"
    PRE 99, 042149 (2019).
    """

    def __init__(self, x, y, dt, percentage_max_n=0.1, typ="", motion="", file=""):
        """
        :param x: list, x coordinates
        :param y: list, y coordinates
        :param dt: float, time between steps
        :param typ: str, type of diffusion i.e sub, super, rand
        :param motion: str, mode of diffusion eg. normal, directed
        :param file: str, path to trajectory
        :param percentage_max_n: float, percentage of length of the trajectory for msd generating
        """

        Characteristic.__init__(self, x, y, dt, percentage_max_n, typ, motion, file)

        self.D_new = self.estimate_diffusion_coef()
        self.p_variations, self.p_variation_names = self.get_pvariation_test(p_list=np.arange(1, 6))
        self.maximum_ts = self.get_maximum_test_statistic()

        self.values = [self.file, self.type, self.motion, self.D_new, self.alpha,
                       self.maximum_ts] + list(self.p_variations)
        self.columns = ["file", "diff_type", "motion", "D", "alpha",
                        "max_ts"] + self.p_variation_names

        self.data = pd.DataFrame([self.values], columns=self.columns)

    def estimate_diffusion_coef(self):
        """
        :return: float, exponential anomalous parameter by alpha parameter;
        estimated based on curve fitting to log of empirical and normal anomalous diffusion.
        Modification of this function can also estimate alpha parameter
        """
        log_msd = np.log(self.empirical_msd)
        tau = np.log((self.dt * self.n_list)).reshape((-1, 1))
        model = LinearRegression().fit(tau, log_msd)
        log_d = model.intercept_
        D = math.exp(log_d) / 4
        return D

    def get_pvariation_test(self, p_list):
        """
        :param p_list: list, p-values for calculation of p-variation
        :return: tuple with list of values and list of strings,
        the list of powers fitted to the calculated p-variations
        and list of the corresponding feature names
        """
        max_m = int(max(0.01 * self.N, 5))
        m_list = np.arange(1, max_m + 1)

        test_values = []
        p_var = generate_empirical_pvariation(self.x, self.y, p_list, m_list)
        for i in range(len(p_list)):
            pv = p_var[i]
            gamma_power_fit = LinearRegression().fit(np.log(m_list).reshape(-1, 1), np.log(pv))
            gamma = gamma_power_fit.coef_[0]
            test_values.append(gamma)

        feature_names = ['p_var_' + str(p) for p in p_list]

        return test_values, feature_names

    def get_maximum_test_statistic(self):
        """
        :return: float, the value of the maximum test statistics
        """
        distance = np.array(
            [self.get_displacement(self.x[i], self.y[i], self.x[0], self.y[0]) for i in range(1, self.N)])
        d_max = np.max(distance)
        # TODO: The sigma estimator can be improved (Briane et al., 2018)
        sigma_2 = 1 / (2 * (self.N - 1) * self.dt) * np.sum(self.displacements ** 2)
        ts = d_max / np.sqrt(sigma_2 * self.T)

        return ts
    
    
class CharacteristicTwo(CharacteristicThree):
    """
    A new set of characteristics, tailor-made for the classification of the movements
    belonging to one of the 
    The description and analysis to appear in: H. Loch-Olszewska, J. Szwabiński,
    "Impact of feature choice on machine learning classification of fractional 
    anomalous diffusion", Entropy.
    """

    def __init__(self, x, y, dt, percentage_max_n=0.1, typ="", motion="", file=""):
        """
        :param x: list, x coordinates
        :param y: list, y coordinates
        :param dt: float, time between steps
        :param typ: str, type of diffusion i.e sub, super, rand
        :param motion: str, mode of diffusion eg. normal, directed
        :param file: str, path to trajectory
        :param percentage_max_n: float, percentage of length of the trajectory for msd generating
        """

        CharacteristicThree.__init__(self, x, y, dt, percentage_max_n, typ, motion, file)

        #self.D_new = self.estimate_diffusion_coef()
        self.velocity_autocorrelation, self.velocity_autocorrelation_names = self.get_velocity_autocorrelation([1])
        self.p_variation = self.get_feature_from_pvariation()
        self.max_excursion_normalised = self.get_max_excursion()

        self.values = [self.file, self.type, self.motion, self.D_new, self.alpha,
                       self.efficiency, self.mean_squared_displacement_ratio, self.straightness,
                       self.p_variation, self.max_excursion_normalised] + list(self.velocity_autocorrelation)
        self.columns = ["file", "diff_type", "motion", "D", "alpha",
                        "efficiency", "mean_squared_displacement_ratio", "straightness",
                        "p-variation", "max_excursion_normalised"] + self.velocity_autocorrelation_names

        self.data = pd.DataFrame([self.values], columns=self.columns)

    def get_velocity_autocorrelation(self, hc_lag_list):
        """
        Calculate the velocity autocorrelation
        :return: float, the empirical autocorrelation for lag 1.
        """
        # hc_lag_list = [1,2,3,4,5]
        titles = ["vac_lag_" + str(x) for x in hc_lag_list]
        autocorr = generate_empirical_velocity_autocorrelation(self.x, self.y, hc_lag_list, self.dt, delta=1)
        return autocorr, titles

    def get_feature_from_pvariation(self):
        """
        Calculate p_variation with preset p and m choice and return the 
        """
        p_list = [1/H for H in np.arange(0.1, 1.0, 0.1)]
        m_list = list(range(1, 11))
        p_var_matrix = generate_empirical_pvariation(self.x, self.y, p_list, m_list)

        m_array = np.array(m_list).reshape(-1, 1)
        p_var_d = [LinearRegression().fit(m_array, p_var_matrix[p_index]).coef_[0] for p_index in range(len(p_list))]
        signs_p = np.nonzero(np.diff([sign(val) for val in p_var_d]))

        if len(signs_p[0]) > 0:
            p_var_info = signs_p[0][0] * sign(p_var_d[0])
        else:
            p_var_info = 0

        return p_var_info

    def get_total_displacement(self):
        """
        The total displacement of the trajectory
        :return: float, the total displacement of a trajectory
        """
        total_displacement = self.get_displacement(self.x[self.N - 1], self.y[self.N - 1], self.x[0], self.y[0])
        return total_displacement

    def get_max_excursion(self):
        """
        The maximal excursion of the particle, normalised to its total displacement (range of movement)
        :return: float, max excursion
        """
        excursion = self.d / self.get_total_displacement()
        return excursion    

