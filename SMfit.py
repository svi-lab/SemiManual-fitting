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

# In[4]:
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from itertools import chain
from scipy.optimize import curve_fit
from utilities import fitonclick, pV
from utilities import multi_peak as fitting_function
plt.style.use('bmh')

figsize = (12, 10)
# In[5]:
# Creating some data...
# You should replace this whole cell with loading your own data
# You should provide x and y as numpy arrays of shape ("length of data", )
# For example: >>> x, y = np.loadtxt("your_file_containing_data.txt")

dummy_params = [51, 200, 85, 0.3,
                10, 272, 37, 0.8,
                2.7, 317, 39, 0.52,
                3.9, 471, 62, 0.25]

dummy_x = np.arange(0, 584, 1.34)
dummy_y = fitting_function(dummy_x, *dummy_params)
dummy_y += np.random.random(len(dummy_x))*np.mean(dummy_y)/5


x = dummy_x
y = dummy_y

# In[6]:
# check the docstring for the fitonclick class:
mfit = fitonclick(x, y, scrolling_speed=4, figsize=figsize)
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
manualfit_components_params = copy(list(map(list, zip(
                            pic['h'], pic['x0'], pic['w'], pic['GL']))))
# to transform the list of lists into one single list:
manualfit_components_params = list(chain(*manualfit_components_params))

# the sum of manually created peaks:
assert len(mfit.sum_peak) > 0, 'No peaks initiated'
manualfit = mfit.sum_peak[0][0].get_ydata()

# In[7]:
# Setting the bounds based on your input
# (you can play with this part if you feel like it,
# but leaving it as it is should be ok for basic usage)

# set the initial bounds as infinities:
upper_bounds = np.ones(len(manualfit_components_params))*np.inf
lower_bounds = np.ones(len(manualfit_components_params))*(-np.inf)

# setting reasonable bounds for the peak amplitude
# as a portion to your initial manual estimate
upper_bounds[0::4] = [A*1.4 for A in manualfit_components_params[0::4]]
lower_bounds[0::4] = [A*0.7 for A in manualfit_components_params[0::4]]

# setting reasonable bounds for the peak position
# as a shift in regard to your initial manual position
upper_bounds[1::4] = \
    [shift + 2*x_size for shift in manualfit_components_params[1::4]]
lower_bounds[1::4] = \
    [shift - 2*x_size for shift in manualfit_components_params[1::4]]

# setting the bounds for the widths
upper_bounds[2::4] = \
    [width*16 for width in manualfit_components_params[2::4]]
lower_bounds[2::4] = \
    [width*0.5 for width in manualfit_components_params[2::4]]


# setting the bounds for the lorentz/gauss ratio
upper_bounds[3::4] = 1
lower_bounds[3::4] = 0


bounds = (lower_bounds, upper_bounds)


# In[7]:

# The curve-fitting part:
fitted_params, b = curve_fit(fitting_function, x, y,
                             p0=manualfit_components_params,
                             absolute_sigma=False, bounds=bounds)
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
    yy_i = pV(x, *fitted_params[i*4:i*4+4])
    peak_i, = pax.plot(x, yy_i, alpha=0.5, label=label)
    pax.fill_between(x, yy_i, facecolor=peak_i.get_color(), alpha=0.3)
pax.legend()
pax.set_title('Showing the individual peaks as found by fitting procedure')

pfig.show()
parametar_names = ['Height', 'Center', 'FWMH', 'Ratio Gauss/Lorenz']
print(f"{'Your initial guess':>47s}{'After fitting':>19s}\n")
for i in range(len(fitted_params)):
    print(f"Peak {i//4}|   {parametar_names[i%4]:<20s}: "
          f" {manualfit_components_params[i]:8.2f}     ->    "
          f" {fitted_params[i]:6.2f} \U000000B1 {fitting_err[i]:4.2f}")
# %%
# Deleting the class instance, in case you want to start over
del mfit
