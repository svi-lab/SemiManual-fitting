#!/usr/bin/env python
'''
# ## This script should permit you to fit your curve with pseudo-voigt peaks.
# Your curve should be baseline-corrected.##
# This procedure allows you to manually select reasonable initial parameters
# to pass on to scipy optimize module.
#
# **Instructions:**
# 1. Left-click on the graph to add one pseudo-voigt profile at a time.
#     -  Once the profile added, you can modify its width by scrolling the
# mouse wheel.
# 2. Repeat the procedure as many times as needed to account for all the peaks.
# - Right-clicking any time, will draw the sum of all the present profiles
# you added up to that moment.
# - Left clicking on the top of existing peak erases it.
# from IPython import get_ipython
# get_ipython().magic('reset -sf')
'''
from copy import copy
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from cycler import cycler

def simple_pseudo_voigt(x, amp=30, center=0, w=10, offset=0, factor=0.5):
    '''Manualy created pseudo-Voigt profile
    Parameters:
    ------------
    x: Independent variable
    amp: Amplitude of the profile (in this case it coincides with the area of
       the profile to retreive the height, you need to do some arithmethics
        height = (((1-factor)*amp)/(w*sqrt(pi/log(2)))+(factor*amp)/(pi*w))
        or (0.3989423*amp/w) for purely gaussian profile
    center: The position of the peak on the x-axis
    w: The measure of the width of the profile, usually called sigma.
       To retreive the width at half-height, you should try: height = 2.0*w,
       or 2.3548200*w for purely gaussian profile
    offset: well, the offset
           (note that it isn't really usefull here,
           since your data is considered to be baseline-substracted)
    factor: the ratio of lorentz vs gauss in the peak
    Returns:
    ------------
    y-array of the same shape as the input x-array
    '''
    return(offset + amp*(factor*((2/np.pi)*w/(4*(x - center)**2 + w**2))
                         + (1 - factor)*np.sqrt(4*np.log(2))/(w*np.sqrt(np.pi))
                         * np.exp(-4*np.log(2)/(w**2) * (x-center)**2)))

def fitting_function(x, *params):
    '''
    The function giving the sum of the pseudo-Voigt peaks.
    Parameters:
    *params: is a list of parameters. Its length is = "number of peaks" * 5,
    where 5 is the number of parameters in the "pv" function.
    Look in the docstring of pv function for more info on theese.
    '''
    result = np.zeros_like(X, dtype=np.float)
    for ii in range(0, len(params), 5):
        result += simple_pseudo_voigt(x, *params[ii:ii+5]) # A, c, w, o, r)
    return result


# In[5]:


# Creating some data...
# You should replace this whole cell with loading your own data
# Yyou should provide x and y as numpy arrays of shape ("length of data", )

# =============================================================================
# dummy_params = [51, 200, 85, 0, 0.7,
#  4, 272, 37, 0, 0.8,
#  2.7, 317, 39, 0, 0.52,
#  3.9, 471, 62, 0, 0.25]
# =============================================================================

# =============================================================================
# dummy_x = np.arange(0,584, 1.34)
# dummy_y = fitting_function(dummy_x, *dummy_params)
# dummy_y += np.random.random(len(dummy_x))*np.mean(dummy_y)/5
#
#
# x = dummy_x
# y = dummy_y
# =============================================================================

DATA = np.load('../c0.npy')
Y = DATA[0][440:-30]
X = np.arange(len(Y))

# In[6]:


# Setting some sensible values to be used afterwards,
# for example for point size (normally, no need to change theese):
N_X = 80
RR = 100


# Setting up the plot:
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(X, Y, 'k', lw=4) # initial plot of your data
ax.set_title('Left-click to add/remove peaks,'
             'Scroll to adjust width,'
             'Right-click to draw sum,'
             'Double-Right-Click when done')

x_size = (ax.get_xlim()[1] - ax.get_xlim()[0])/RR
y_size = 2*(ax.get_ylim()[1] - ax.get_ylim()[0])/RR

plt.rcParams["axes.prop_cycle"] = cycler('color',
                                         ['#332288', '#CC6677', '#DDCC77',
                                          '#117733', '#88CCEE', '#AA4499',
                                          '#44AA99', '#999933', '#882255',
                                          '#661100', '#6699CC', '#AA4466'])
# the above sets up the color palette to be used for plotting lines
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Initiating variables to which we will atribute peak caractéristics
#block = False
pic = {}
pic['line'] = [] # List containing matplotlib.Line2D object for each peak
pic['A'] = [] # List that will contain amplitudes of each peak
pic['center'] = [] # List that will contain central positions of each peak
pic['w'] = [] # List containing widths
pic['fill'] = []
it = 0 # Iterator used normally for counting right clicks
# (each right click launches the plot of the cumulative curbe)
sum_peak = [] # List of cumulated graphs
# (used later for updating while removing previous one)
peaks_present = 0
cid3 = []
scroll_count = 0 # counter to store the cumulative values of scrolling
indice = 0
artists = []
clicked_indice = -1

plt.ioff()

def onclick(event):
    global it, peaks_present, scroll_count, indice, x_size, y_size, clicked_indice, X, block

    if event.inaxes == ax:
        if event.button == 1:
            click_in_artist = [artist.contains(event)[0] for artist in artists]
            if not any(click_in_artist):
                indice += 1
                peaks_present += 1
                artists.append(ax.add_artist(Ellipse((event.xdata, event.ydata),
                                                     x_size, y_size, alpha=0.5,
                                                     picker=max(x_size, y_size),
                                                     gid=indice)))
                amplitude = 1.27*x_size*event.ydata
                center = event.xdata
                yy = simple_pseudo_voigt(x=X, amp=amplitude, center=center, w=x_size)
                #ax.set_ylim(auto=True)
                pic['A'].append(amplitude)
                pic['center'].append(center)
                pic['line'].append(ax.plot(X, yy, alpha=0.75, lw=2.5, picker=5))
                pic['w'].append(x_size)
                #ax.fill_between(x, yy.min(), yy, alpha=0.3, color=cycle[peaks_present])
                fig.canvas.draw_idle()

            else:
                clicked_indice = click_in_artist.index(True)
                artists[clicked_indice].remove()
                artists.pop(clicked_indice)
                ax.lines[clicked_indice+1].remove()
                pic['line'].pop(clicked_indice)
                pic['center'].pop(clicked_indice)
                pic['A'].pop(clicked_indice)
                pic['w'].pop(clicked_indice)
                fig.canvas.draw_idle()

        elif event.button == 3 and not event.step:
            if it > 0: # Checks if there is already a cumulated graph plotted
                ax.lines.remove(sum_peak[-1][0]) # remove the last cumulated graph from the figure
                sum_peak.pop()
            # Sum all the y values from all the peaks:
            sumy = np.sum(np.asarray([pic['line'][i][0].get_ydata()
                                      for i in range(len(pic['line'][:]))]),
                          axis=0)
            if sumy.shape == X.shape:
                # Added this condition for the case where you removed all peaks,
                # but the cumulated graph is left,
                # then right-clicking need to remove that one
                sum_peak.append(ax.plot(X, sumy, '--',
                                        color='lightgreen',
                                        lw=3, alpha=0.6)) # plot the cumulated graph
                it += 1 # One cumulated graph added
            else:
                it -= 1 # if you right clicked on the graph with no peaks,
                # then you remove the cumulated graph as well
            fig.canvas.draw_idle()

        elif event.step != 0:
            if peaks_present:
                peak_identifier = -1 # means that scrolling will only affect
                # the last plotted peak
                scroll_count += x_size*event.step/15 # This adjust the "speed"
                # of width change with scrolling (event.step is always +-1)

                if scroll_count > -x_size+0.01:
                    w2 = x_size + scroll_count
                else:
                    w2 = 0.01
                    scroll_count = -x_size+0.01 # This doesn't allow you
                    # to sroll to negative values (basic width is x_size)

                center2 = pic['center'][peak_identifier]
                amp2 = pic['A'][peak_identifier]*w2/x_size
                # In order to keep the Height unchanged
                # (A (from amplitude), as defined here, is proportional to the area)
                pic['w'][peak_identifier] = w2
                yy = simple_pseudo_voigt(x=X, center=center2, amp=amp2, w=w2)
                active_line = pic['line'][peak_identifier][0]
                active_line.set_ydata(yy) # This updates the values on the peak
                #identified by "peak_identifier" (last peak for -1).
                ax.draw_artist(active_line)
                fig.canvas.draw_idle()
# =============================================================================
#                 if peak_identifier > -1:
#                     cycle_indice = peak_identifier
#                 else:
#                     cycle_indice = indice
#                 pic['fill'].append(ax.fill_between(x, 0, yy, alpha=0.3,
#                                    color=cycle[cycle_indice]))
#                 fig.canvas.draw_idle()
# =============================================================================
        if event.button != 1 and event.dblclick:
            block = True
            fig.canvas.mpl_disconnect(cid)
            fig.canvas.mpl_disconnect(cid2)
            plt.close()
            return


cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid2 = fig.canvas.mpl_connect('scroll_event', onclick)#scroll)

while not 'block' in locals():
    plt.waitforbuttonpress(timeout=-1)
del block
# these last 3 lines had to be added so to block the execution until the user
# finishes with "manual fitting"


# In[7]:

manualfit_components = [copy(line[0].get_data()[1]) for line in pic['line']]
# recovering the y values of each peak (not really useful)

N_peaks = len(manualfit_components)

manualfit_components_params = copy(list(
    map(list, zip(
        [A*w/x_size for A, w in list(zip(pic['A'], pic['w']))],
        pic['center'],
        pic['w'],
        [0]*N_peaks,
        [0.5]*N_peaks))))
# creating the list of initial parameters from your manual input (as a list of lists)

manualfit_components_params = list(chain(*manualfit_components_params))
# to transform the list of lists into one single list
manualfit = np.sum(np.asarray(manualfit_components), axis=0)
# the sum of manually created peaks


# In[7]:
# Setting the bounds based on your input
# (you can play with this part if you feel like it, but leaving it as it is should be ok)

# set the initial bounds as infinities:
upper_bounds = np.ones(len(manualfit_components_params))*np.inf
lower_bounds = np.ones(len(manualfit_components_params))*(-np.inf)

# setting reasonable bounds for the peak amplitude as a portion to your manual estimate
upper_bounds[0::5] = [A*1.2 for A in manualfit_components_params[0::5]]
lower_bounds[0::5] = [A*0.8 for A in manualfit_components_params[0::5]]

# setting reasonable bounds for the peak position as a shift to your manual position
upper_bounds[1::5] = [shift + x_size for shift in manualfit_components_params[1::5]]
lower_bounds[1::5] = [shift - x_size for shift in manualfit_components_params[1::5]]

# setting the bounds for the widths
upper_bounds[2::5] = [width*1.10 for width in manualfit_components_params[2::5]]
lower_bounds[2::5] = [width*0.9 for width in manualfit_components_params[2::5]]

# setting the bounds for the offsets
upper_bounds[3::5] = y_size * 5
lower_bounds[3::5] = 0

# setting the bounds for the lorentz/gauss ratio
upper_bounds[4::5] = 1
lower_bounds[4::5] = 0

bounds = (lower_bounds, upper_bounds)


# In[7]:
# The fitting part:
a, b = curve_fit(fitting_function, X, Y, p0=manualfit_components_params, bounds=bounds)
# Strangely enough, adding the bounds does not improve significantly
# the efficiency of the procedure.
# Note however that since the offset parametar here doesn't really play
# any role, not using the bounds will give you some strange values for it.

figg, axx, = plt.subplots(figsize=(16, 12))
figg.subplots_adjust(bottom=0.2)
err_ax = figg.add_axes([0.125, 0.05, 0.775, 0.1])
err_ax.set_facecolor('w')

axx.plot(X, fitting_function(X, *manualfit_components_params), '--r',
         alpha=0.5, label='initial manual fit')
axx.plot(X, Y, 'k', label='original data')
axx.plot(X, fitting_function(X, *a), '--g', lw=4, alpha=0.6,
         label='after optimization')
axx.legend()
err_values = (Y-fitting_function(X, *a)) / Y.mean()
err_ax.scatter(X, err_values)
err_ax.set_ylim((err_values.min()*1.25, err_values.max()*1.25))
err_ax.set_ylabel('error')
axx.set_title('After fitting')
plt.show(block=False)

# In[8]:

SDT_ERR = np.sqrt(np.diag(b))
plt.figure(figsize=(16, 12))
plt.plot(X, Y, lw=4, c='k')
#plt.plot(manualfit, '--y', lw=3, alpha=0.6)
for i in range(len(manualfit_components)):
    plt.plot(X, simple_pseudo_voigt(X, *a[i*5:i*5+5])-a[i*5+3], alpha=0.5,
             label=[f"{prd:.2f}\U000000B1{krk:.1f}"
                    for prd, krk in list(zip(a[i*5:i*5+5], SDT_ERR[i*5:i*5+5]))])
plt.legend()
plt.title('Showing the individual peaks as found by fitting procedure')

plt.show(block=False)
parametar_names = ['Amplitude', 'Center', 'Width', 'Offset', 'Ratio Lorenz/Gauss']
for i, aa in enumerate(a):
    print(f"Peak_{i//5} - Manually found  {parametar_names[i%5]:<24s}:"
          f"{manualfit_components_params[i]:8.2f}"
          f"  ------->  After fitting :   {aa:8.2f} \U000000B1 {SDT_ERR[i]:8.2f}")
          #f"::::::: dummy params : {dummy_params[i]:8.2f}")
          