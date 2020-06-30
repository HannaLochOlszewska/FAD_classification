import numpy as np
from fbm import FBM

from stochastic.diffusion import OrnsteinUhlenbeckProcess

"""This file contains a functions which generates 2D vectors with different type of motions"""

def fbm_generator(N, H, sigma=1.0, dt=1.0):
    """
    Function generates the 2D vector of fractional Brownian motion
    :param N: int, number of steps
    :param H: float, Hurst parameter
    :param sigma: float, scale parameter
    :param dt: float, time between steps
    :return: 2D vector of fBm
    """
    T = N * dt
    f = FBM(n=int(N), hurst=H, length=T, method='daviesharte')
    x = f.fbm() * sigma
    y = f.fbm() * sigma
    return x[:N], y[:N]


def ou_generator(N, lmbd, theta, sigma=1.0, dt=1.0):
    """
    Function generates the 2D vector of Ornstein-Uhlenbeck process
    :param N: int, number of steps
    :param lbmd: tuple, the lambda parameter for each coordinate (reversion speed)
    :param theta: tuple, the theta parameter for each coordinate (mean of the process)
    :param sigma: float, scale parameter
    :param dt: float, time between steps
    :return: 2D vector of Ornstein-Uhlenbeck process
    """
    T = N * dt
    s_x = OrnsteinUhlenbeckProcess(speed=lmbd[0], mean=theta[0], vol=sigma, t=T)
    s_y = OrnsteinUhlenbeckProcess(speed=lmbd[1], mean=theta[1], vol=sigma, t=T)
    x = s_x.sample(N)
    y = s_y.sample(N)
    return x[:N], y[:N]


def directed_bm_generator(N, we, sigma=1.0, dt=1.0):
    """
    Function generates the 2D vector of Ornstein-Uhlenbeck process
    :param N: int, number of steps
    :param we: tuple, the constant velocity of drift on each coordinate
    :param sigma: float, scale parameter
    :param dt: float, time between steps
    :return: 2D vector of anomalous diffusion
    """
    T = N * dt
    f = FBM(n=int(N), hurst=0.5, length=T, method='daviesharte')
    x = f.fbm() * sigma + np.linspace(0, T, N + 1) * we[0]
    y = f.fbm() * sigma + np.linspace(0, T, N + 1) * we[1]
    return x[:N], y[:N]
