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
from matplotlib.patches import Ellipse
#from matplotlib.artist import ArtistInspector
from copy import copy
from itertools import chain
from scipy.optimize import curve_fit
plt.style.use('bmh')

# initial parameters:
scrolling_speed: float = 1
initial_width: float = 5
initial_GaussToLoretnz_ratio = 0.5
figsize = (12, 8)


def pV(x, h=30, x0=0, w=10, factor=0.5):
    '''Manualy created pseudo-Voigt profile
    Parameters:
    ------------
    x: Independent variable
    h: height
    x0: The position of the peak on the x-axis
    w: FWHM
    factor: the ratio of lorentz vs gauss in the peak
    Returns:
    y-array of the same shape as the input x-array
    '''

    def Gauss(x, w):
        return((2/w) * np.sqrt(np.log(2)/np.pi) * np.exp(
                -(4*np.log(2)/w**2) * (x - x0)**2))

    def Lorentz(x, w):
        return((1/np.pi)*(w/2) / (
                (x - x0)**2 + (w/2)**2))

    intensity = h * np.pi * (w/2) / (
                    1 + factor * (np.sqrt(np.pi*np.log(2)) - 1))

    return(intensity * (factor * Gauss(x, w)
                        + (1-factor) * Lorentz(x, w)))


def fitting_function(x, *params):
    '''
    The function giving the sum of the pseudo-Voigt peaks.
    Parameters:
    *params: is a list of parameters. Its length is = 4 * "number of peaks",
    where 4 is the number of parameters in the "pV" function.
    Look in the docstring of pV function for more info on theese.
    '''
    result = np.zeros_like(x, dtype=np.float)
    for i in range(0, len(params), 4):
        result += pV(x, *params[i:i+4])  # h, x0, w, r)
    return result


# In[5]:
# Creating some data...
# You should replace this whole cell with loading your own data
# You should provide x and y as numpy arrays of shape ("length of data", )
# For example: >>> x, y = np.loadtxt("your_file_containing_data.txt")

dummy_params = [51, 200, 85, 0.7,
                10, 272, 37, 0.8,
                2.7, 317, 39, 0.52,
                3.9, 471, 62, 0.25]

dummy_x = np.arange(0, 584, 1.34)
dummy_y = fitting_function(dummy_x, *dummy_params)
dummy_y += np.random.random(len(dummy_x))*np.mean(dummy_y)/5


x = dummy_x
y = dummy_y

# In[6]:

# Setting some sensible values to be used afterwards,
# impacting for example the elipse size and initial width:
def set_size(variable, rapport=70):
    return (variable.max() - variable.min())/rapport


x_size = set_size(x)
y_size = 2*set_size(y)
# %% Setting up the plot:
fig, ax = plt.subplots(figsize=figsize)
ax.plot(x, y, linestyle='none', marker='o', c='k', ms=4, alpha=0.5)
ax.set_title('Left-click to add/remove peaks,'
             'Scroll to adjust width, \nRight-click to draw sum,'
             'Double-Right-Click when done')
plt.show()
# %%

# Initiating variables to which we will atribute peak caractéristics:
pic = {}
pic['line'] = []  # List containing matplotlib.Line2D object for each peak
pic['h'] = []  # List that will contain heights of each peak
pic['x0'] = []  # List that will contain central positions of each peak
pic['w'] = []  # List containing widths

# List of cumulated graphs
# (used later for updating while removing previous one)
sum_peak = []

cid3 = []
scroll_count = 0  # counter to store the cumulative values of scrolling
artists = []  # will be used to store the elipses on tops of the peaks

plt.ioff()


def onclick(event, x=x, x_size=x_size, y_size=y_size,
            scrolling_speed=scrolling_speed,
            initial_width=initial_width,
            GL=initial_GaussToLoretnz_ratio):
    global pic, sum_peak, artists, scroll_count, block

    cum_graph_present: int = len(sum_peak)  # actually only 0 or 1
    peak_counter: int = len(pic['line'])  # number of peaks on the graph

    def _add_peak(artists, pic):
        h = event.ydata
        x0 = event.xdata
        yy = pV(x=x, h=h, x0=x0, w=x_size*initial_width, factor=GL)
        one_elipsis = ax.add_artist(
                        Ellipse((x0, h),
                                x_size, y_size, alpha=0.5,
                                gid=str(peak_counter)))
        artists.append(one_elipsis)
        pic['line'].append(ax.plot(x, yy, alpha=0.75, lw=2.5,
                           picker=5))
        # ax.set_ylim(auto=True)
        pic['h'].append(h)
        pic['x0'].append(x0)
        pic['w'].append(x_size)
        fig.canvas.draw_idle()
        return(artists, pic)

    def _remove_peak(clicked_indice, artists, pic):
        artists[clicked_indice].remove()
        artists.pop(clicked_indice)
        ax.lines.remove(pic['line'][clicked_indice][0])
        pic['line'].pop(clicked_indice)
        pic['x0'].pop(clicked_indice)
        pic['h'].pop(clicked_indice)
        pic['w'].pop(clicked_indice)
        fig.canvas.draw_idle()
        return(artists, pic)

    def _draw_peak_sum(cum_graph_present, sum_peak, pic):
        def _remove_sum(sum_peak):
            ax.lines.remove(sum_peak[-1][0])
            sum_peak.pop()
            return sum_peak
        def _add_sum(sum_peak):
            sum_peak.append(ax.plot(x, sumy,
                                    '--',
                                    color='lightgreen',
                                    lw=3, alpha=0.6))
            return sum_peak
        # Sum all the y values from all the peaks:
        sumy = np.sum(np.asarray(
                [pic['line'][i][0].get_ydata() for i in range(peak_counter)]),
                axis=0)
        # Check if there is already a cumulated graph plotted:
        if cum_graph_present == 1:
            # Check if the sum of present peaks correponds to the cumulated graph
            if np.array_equal(sum_peak[-1][0].get_ydata(), sumy):
                pass
            else:  # if not, remove the last cumulated graph from the figure:
                sum_peak = _remove_sum(sum_peak)
                cum_graph_present -= 1
                # and then plot the new cumulated graph:
                if sumy.shape == x.shape:
                    sum_peak = _add_sum(sum_peak)
                    cum_graph_present += 1
        # No cumulated graph present:
        elif cum_graph_present == 0:
            # plot the new cumulated graph:
            if sumy.shape == x.shape:
                sum_peak = _add_sum(sum_peak)
                cum_graph_present += 1
        else:
            raise("WTF?")
        fig.canvas.draw_idle()
        return(cum_graph_present, sum_peak)

    def _adjust_peak_width(pic, peak_identifier=-1, scroll_count=scroll_count):
        scroll_count += x_size*np.sign(event.step)*scrolling_speed/10

        if scroll_count > -x_size*initial_width*0.999:
            w2 = x_size*initial_width + scroll_count
        else:
            w2 = x_size*initial_width/1000
            # This doesn't allow you to sroll to negative values
            # (basic width is x_size)
            scroll_count = -x_size*initital_width*0.999

        center2 = pic['x0'][peak_identifier]
        h2 = pic['h'][peak_identifier]
        pic['w'][peak_identifier] = w2
        yy = pV(x=x, x0=center2, h=h2, w=w2, factor=GL)
        active_line = pic['line'][peak_identifier][0]
        # This updates the values on the peak identified
        active_line.set_ydata(yy)
        ax.draw_artist(active_line)
        fig.canvas.draw_idle()
        return(scroll_count, pic)

    # Now let's put everything together:peak_counter:int = 0
    if event.inaxes == ax:  # if you click inside the plot
        if event.button == 1:  # left click
            # Create list of all elipses and check if the click on one of them:
            click_in_artist = [artist.contains(event)[0] for artist in artists]
            if any(click_in_artist):  # if the click was on one of the elipses
                clicked_indice = click_in_artist.index(True) # identify the one
                artists, pic = _remove_peak(clicked_indice, artists, pic)
                peak_counter -= 1

            else:  # if click was not on any of the already drawn elipsis
                peak_counter += 1
                artists, pic = _add_peak(artists, pic)

        elif event.button == 3 and not event.step:
            # On some computers middle and right click have both the value 3
            cum_graph_present, sum_peak = _draw_peak_sum(cum_graph_present,
                                                         sum_peak, pic)

        elif event.step != 0:
            if peak_counter:
                scroll_count, pic = _adjust_peak_width(pic, peak_identifier=-1)
                # -1 means that scrolling will only affect the last plotted peak

        if event.button != 1 and event.dblclick:
            block = True
            fig.canvas.mpl_disconnect(cid)
            fig.canvas.mpl_disconnect(cid2)
            plt.close()
            return



cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid2 = fig.canvas.mpl_connect('scroll_event', onclick)

while not 'block' in locals():
    plt.waitforbuttonpress(timeout=-1)

# In[7]:
peak_counter: int = len(pic['line'])

# creating the list of initial parameters from your manual input:
# (as a list of lists)
manualfit_components_params = copy(list(map(list, zip(
                            pic['h'], pic['x0'], pic['w'],
                            [initial_GaussToLoretnz_ratio]*peak_counter
                            ))))
# to transform the list of lists into one single list:
manualfit_components_params = list(chain(*manualfit_components_params))

# the sum of manually created peaks:
manualfit = sum_peak[0][0].get_data()[1]

# In[7]:

# Setting the bounds based on your input
# (you can play with this part if you feel like it,
# but leaving it as it is should be ok for basic usage)

# set the initial bounds as infinities:
upper_bounds = np.ones(len(manualfit_components_params))*np.inf
lower_bounds = np.ones(len(manualfit_components_params))*(-np.inf)

# setting reasonable bounds for the peak amplitude
# as a portion to your initial manual estimate
upper_bounds[0::4] = [A*1.32 for A in manualfit_components_params[0::4]]
lower_bounds[0::4] = [A*0.75 for A in manualfit_components_params[0::4]]

# setting reasonable bounds for the peak position
# as a shift in regard to your initial manual position
upper_bounds[1::4] = \
    [shift + 2*x_size for shift in manualfit_components_params[1::4]]
lower_bounds[1::4] = \
    [shift - 2*x_size for shift in manualfit_components_params[1::4]]

# setting the bounds for the widths
upper_bounds[2::4] = \
    [width*10 for width in manualfit_components_params[2::4]]
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
                 f'{np.sum(fitting_err/np.ceil(fitted_params))/peak_counter:.3f}'
                 f'\n\u03A3(\u0394param/param) /n_peaks')
axx.set_title('After fitting')
plt.show(block=False)

# In[8]:

plt.figure(figsize=figsize)
plt.plot(x, y, linestyle='none', marker='o', ms=4, c='k', alpha=0.3)
plt.plot(x, fitting_function(x, *fitted_params),
         '--k', lw=2, alpha=0.6, label='fit')
par_nam = ['h', 'x0', 'w', 'G/L']
# plt.plot(manualfit, '--y', lw=3, alpha=0.6)
for i in range(peak_counter):
    fit_res = list(zip(par_nam, fitted_params[i*4:i*4+4],
                       fitting_err[i*4:i*4+4]))
    label = [f"{P}={v:.2f}\U000000B1{e:.1f}" for P, v, e in fit_res]
    plt.plot(x, pV(x, *fitted_params[i*4:i*4+4]), alpha=0.5, label=label)
plt.legend()
plt.title('Showing the individual peaks as found by fitting procedure')

plt.show(block=False)
parametar_names = ['Height', 'Center', 'FWMH', 'Ratio Gauss/Lorenz']
print(f"{'Your initial guess':>47s}{'After fitting':>19s}\n")
for i in range(len(fitted_params)):
    print(f"Peak {i//4}|   {parametar_names[i%4]:<20s}: "
          f" {manualfit_components_params[i]:8.2f}     ->    "
          f" {fitted_params[i]:6.2f} \U000000B1 {fitting_err[i]:4.2f}")

