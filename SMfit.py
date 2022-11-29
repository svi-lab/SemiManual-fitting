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
mfit = ut.fitonclick(x, y, scrolling_speed=4, figsize=figsize)
# Important:
while mfit.block:
    plt.waitforbuttonpress(timeout=-1)
plt.close()  # You can also leave the plot on if you want
# In[7]:
x_size = mfit.x_size
peaks_present: int = mfit.peak_counter
pic = mfit.pic
# creating the list of initial parameters from your manual input:
# (as a list of lists)
mf_params = mfit.manualfit_params


# the sum of manually created peaks:
manualfit = mfit.manualfit_spectra
# if len(mfit.sum_peak) == 1:
#     manualfit = mfit.sum_peak[0][0].get_ydata()
# else:
#     manualfit = fitting_function(x, *mf_params)

# In[7]:
# Setting the bounds based on your input
# (you can play with this part if you feel like it,
# but leaving it as it is should be ok for basic usage)
bounds = ut.set_bounds(mf_params.reshape(-1, 4),
                       A=(0.5, 1.4, "multiply"),
                       x=(-2*x_size, 2*x_size, "add"),
                       w=(0.5, 16, "multiply"),
                       gl=(0, 1, "absolute"))


# In[7]:
# The curve-fitting part:
fitted_params, b = curve_fit(fitting_function, x, y, method='trf',
                             p0=mf_params,
                             absolute_sigma=False, bounds=bounds)
#%%
fitting_err = np.sqrt(np.diag(b))

y_fitted = fitting_function(x, *fitted_params)
# %%
# Plotting the results of the optimization:
figg, axx, = plt.subplots(figsize=figsize)
figg.subplots_adjust(bottom=0.25)
errax = figg.add_axes([0.125, 0.1, 0.775, 0.1])
errax.set_facecolor('w')

axx.plot(x, manualfit,
         '--g', alpha=0.5, label='initial manual fit')

axx.plot(x, y, linestyle='none', marker='o', c='k',
         alpha=0.3, label='original data')

axx.plot(x, y_fitted,
         '--r', lw=4, alpha=0.6, label='after optimization')

axx.legend()

errax.plot(x, y-y_fitted, linestyle='none', marker='o')
errax.set_ylabel('error\n(data - fit)')
errax.set_xlabel(f'fitting error = '
                 f'{np.sum(fitting_err/np.ceil(fitted_params))/peaks_present:.3f}'
                 f'\n\u03A3(\u0394param/param) /n_peaks')
errax2 = errax.twinx()
errax2.set_yticks([])
errax2.set_ylabel(f'\u03A3(\u0394y) = {np.sum(y-y_fitted):.2f}',
                  fontsize='small')
axx.set_title('After fitting')
plt.show(block=False)

# In[8]:
# Plotting the individual peaks after fitting
pfig, pax = plt.subplots(figsize=figsize)

pax.plot(x, y, linestyle='none', marker='o', ms=4, c='k', alpha=0.3)
pax.plot(x, y_fitted,
         '--k', lw=2, alpha=0.6, label='fit')
par_nam = ['h', 'x0', 'w', 'G/L']
for i in range(peaks_present):
    fit_res = list(zip(par_nam, fitted_params[i*4:i*4+4],
                       fitting_err[i*4:i*4+4]))
    label = [f"{P}={v:.2f}\U000000B1{e:.1f}" for P, v, e in fit_res]
    yy_i = ut.pV(x, *fitted_params[i*4:i*4+4])
    peak_i, = pax.plot(x, yy_i, alpha=0.5, label=label)
    pax.fill_between(x, yy_i, facecolor=peak_i.get_color(), alpha=0.3)
pax.legend()
pax.set_title('Showing the individual peaks as found by fitting procedure')

pfig.show()
# %%
# print the parameters as found by the fitting procedure:
parametar_names = ['Height', 'Center', 'FWMH', 'Gauss/Lorenz']
if len(dummy_params) == len(fitted_params):
    rpm = "Real params"
    real_parameter_values = [f"   ::   {dp:6.1f}" for dp in dummy_params]
else:
    rpm = ""
    real_parameter_values = [""]*peaks_present*4
print(f"{'Your initial guess':>30s}{'After fitting':>26s}{rpm:>16s}\n")
for i in range(len(fitted_params)):
    print(f"Peak {i//4}|   {parametar_names[i%4]:<13s}: "
          f" {mf_params[i]:8.2f}   ->   "
          f" {fitted_params[i]:6.2f} \U000000B1 {fitting_err[i]:4.2f}"
          f"{real_parameter_values[i]}")
# %%
# Deleting the class instance, in case you want to start over
del mfit
del mf_params
