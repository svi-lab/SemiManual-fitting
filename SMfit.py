#!/usr/bin/env python
# coding: utf-8

# This script should permit you to fit your curve with pseudo-voigt peaks.
# Your curve should be baseline-corrected.
# This procedure allows you to manually select reasonable initial parameters
# to pass on to scipy optimize module.
#
# **Instructions:**
# 1. Left-click on the graph to add one pseudo-voigt profile at a time.
#     -  Once the profile added, you can modify its width by scrolling
# 2. Repeat the procedure as many times as needed to account for all the peaks.
# - Right-clicking any time, will draw the sum of all the present profiles
# - Left clicking on the top of existing peak erases it.

# In[1]:
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import utilities as ut

fitting_function = ut.multi_pV
plt.style.use('bmh')

# del mfit
figsize = (12, 10)
# In[2]:
# Creating some data...
# You should replace this whole cell with loading your own data
# You should provide x and y as numpy arrays of shape ("length of data", )
# For example: >>> x, y = np.loadtxt("your_file_containing_data.txt")

dummy_params = [51, 200, 85, 0.3,
                10, 272, 37, 0.8,
                2.7, 317, 39, 0.52,
                3.9, 471, 62, 0.25]
# A, X, w, GL
x = np.arange(0, 584, 1.34)
y = fitting_function(x, *dummy_params)
# Add some noise to the data:.
y += np.random.random(len(x))*np.mean(y)/5


# In[3]:
# check the docstring for the fitonclick class:
mfit = ut.fitonclick(y, x, scrolling_speed=4, figsize=figsize,
                     # These bounds will be passed to curve_fit inside the class
                     A_bounds=(0.5, 1.4, "multiply"),
                     x_bounds=(-2*ut.set_size(x), 2*ut.set_size(x), "add"),
                     w_bounds=(0.5, 16, "multiply"),
                     gl_bounds=(0, 1, "absolute"),
                     plot_results=True)
# %%
# print the parameters as found by the fitting procedure:
parametar_names = ['Height', 'Center', 'FWMH', 'Gauss/Lorenz']
if len(dummy_params) == len(fitted_params):
    rpm = "Real params"
    real_parameter_values = [f"   ::   {dp:6.1f}" for dp in dummy_params]
else:
    rpm = ""
    real_parameter_values = [""]*len(mfit.fitted_params)
print(f"{'Your initial guess':>30s}{'After fitting':>26s}{rpm:>16s}\n")
for i in range(len(mfit.fitted_params)):
    print(f"Peak {i//4}|   {parametar_names[i%4]:<13s}: "
          f" {mfit.manualfit_params.flatten()[i]:8.2f}   ->   "
          f" {mfit.fitted_params[i]:6.2f} \U000000B1 {mfit.fitting_err[i]:4.2f}"
          f"{real_parameter_values[i]}")
# %%
# Deleting the class instance, in case you want to start over
# del mfit
# del mf_params
