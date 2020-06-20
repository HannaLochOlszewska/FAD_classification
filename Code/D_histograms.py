# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:49:51 2020

@author: Hania
"""

import pandas as pd
import matplotlib.pyplot as plt

char_Wagner_a2AR = "C:\Hania\Praca\Python\FAD_classification\Data\Empirical data\Beethoven_AWJJEB\Characteristics_a2AR_basal_Wagner_10\characteristics.csv"
char_Wagner_Gi = "C:\Hania\Praca\Python\FAD_classification\Data\Empirical data\Beethoven_AWJJEB\Characteristics_Gi_basal_Wagner_10\characteristics.csv"
char_best_old_a2AR = "C:\Hania\Praca\Python\FAD_classification\Data\Empirical data\Beethoven_AWJJEB\Characteristics_a2AR_basal_best_old\characteristics.csv"
char_best_old_Gi = "C:\Hania\Praca\Python\FAD_classification\Data\Empirical data\Beethoven_AWJJEB\Characteristics_Gi_basal_best_old\characteristics.csv"

ch_W_a = pd.read_csv(char_Wagner_a2AR)["D"].values
ch_W_G = pd.read_csv(char_Wagner_Gi)["D"].values
ch_bo_a = pd.read_csv(char_best_old_a2AR)["D"].values
ch_bo_G = pd.read_csv(char_best_old_Gi)["D"].values


fig, ax = plt.subplots(figsize=(8,6))
plt.subplot(221)
plt.hist(ch_W_a, bins='fd', density=True)
plt.xlim([0,0.004])
plt.ylabel("Characteristic")
plt.xlabel("D")
plt.title("Receptors")
plt.subplot(222)
plt.hist(ch_W_G, bins='fd', density=True)
plt.xlim([0,0.004])
plt.title("G-proteins")
plt.xlabel("D")
plt.subplot(223)
plt.hist(ch_bo_a, bins='fd', density=True)
plt.xlim([0,0.004])
plt.ylabel("CharacteristicTwo/Three")
plt.xlabel("D")
plt.subplot(224)
plt.hist(ch_bo_G, bins='fd', density=True)
plt.xlim([0,0.004])
plt.xlabel("D")
fig.tight_layout()
plt.savefig("C:\Hania\Praca\TeX\\Notatki-klasyfikatory\D_estimation_comparison.png", dpi=200)


import matplotlib
matplotlib.rc('font', **{'family':'sans-serif','sans-serif':['Arial']})#,'size':20})
font = 20

fig, ax = plt.subplots(figsize=(6,3))
plt.subplot(121)
plt.hist(ch_bo_a, bins='fd')
plt.title("Receptors")
plt.xlim([0,0.01])
plt.ylabel("Number of trajectories")
plt.xlabel("D")
#tick_marks = np.linspace(0.0,1.0,5)
#plt.xticks(tick_marks, rotation=45)
plt.subplot(122)
plt.hist(ch_bo_G, bins='fd')
plt.xlim([0,0.01])
plt.ylabel("Number of trajectories")
plt.title("G-proteins")
plt.xlabel("D")
fig.tight_layout()
plt.savefig("C:\Hania\Praca\TeX\\Notatki-klasyfikatory\D_estimation.pdf", dpi=200)