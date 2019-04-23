#!/usr/bin/env python
# coding: utf-8

# ## This script should permit you to fit your curve with pseudo-voigt peaks. Your curve should be baseline-corrected.##
# This procedure allows you to manually select reasonable initial parameters to pass on to scipy optimize module.
# 
# **Instructions:** 
# 1. Left-click on the graph to add one pseudo-voigt profile at a time.
#     -  Once the profile added, you can modify its width by scrolling the mouse wheel.
# 2. Repeat the procedure as many times as needed to account for all the peaks.
# - Right-clicking any time, will draw the sum of all the present profiles you added up to that moment.
# - Left clicking on the top of existing peak erases it.

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'widget')
plt.style.use('bmh')
from matplotlib.patches import Ellipse
from cycler import cycler
from copy import copy
from itertools import chain
from scipy.optimize import curve_fit


def pv(x, A=30, center=0, w=10, offset=0, factor=0.5):
    '''Manualy created pseudo-Voigt profile
    Parameters:
    ------------
    x: Independent variable
    A: Amplitude of the profile (in this case it coincides with the area of the profile to retreive the height, you need to do some arithmethics
        height = (((1-factor)*A)/(w*sqrt(pi/log(2)))+(factor*A)/(pi*w))
        or (0.3989423*A/w) for purely gaussian profile
    center: The position of the peak on the x-axis
    w: The measure of the width of the profile, usually called sigma. To retreive the width at half-height, you should try: height = 2.0*w, or 2.3548200*w for purely gaussian profile
    offset: well, the offset (note that it isn't really usefull here, since your data is considered to be baseline-substracted)
    factor: the ratio of lorentz vs gauss in the peak
    Returns:
    y-array of the same shape as the input x-array
    '''
    return(offset + A*(factor*((2/np.pi)*w/(4*(x - center)**2 + w**2)) + (1 - factor)*np.sqrt(4*np.log(2))/(w*np.sqrt(np.pi))*np.exp(-4*np.log(2)/(w**2) * (x-center)**2)))

def fitting_function(x, *params):
    '''
    The function giving the sum of the pseudo-Voigt peaks.
    Parameters:
    *params: is a list of parameters. Its length is = "number of peaks" * 5, where 5 is the number of parameters in the "pv" function. Look in the docstring of pv function for more info on theese.
    '''
    result = np.zeros_like(x, dtype=np.float)   
    for i in range(0, len(params), 5):
        result += pv(x, *params[i:i+5]) # A, c, w, o, r)
    return result


# In[5]:


# Creating some data...
# You should replace this whole cell with loading your own data 
# Yyou should provide x and y as numpy arrays of shape ("length of data", )


dummy_params = [51, 200, 85, 0, 0.7, 
 4, 272, 37, 0, 0.8, 
 2.7, 317, 39, 0, 0.52, 
 3.9, 471, 62, 0, 0.25]

dummy_x = np.arange(0,584, 1.34)
dummy_y = fitting_function(dummy_x, *dummy_params)
dummy_y += np.random.random(len(dummy_x))*np.mean(dummy_y)/5


x = dummy_x
y = dummy_y


# In[6]:


# Setting some sensible values to be used afterwards, for example for point size (normally, no need to change theese):
nx = 80
rapport = 100


# Setting up the plot:
fig, ax = plt.subplots(figsize=(16,8))
ax.plot(x, y, 'k', lw=4) # initial plot of your data


x_size = (ax.get_xlim()[1] - ax.get_xlim()[0])/rapport
y_size = 2*(ax.get_ylim()[1] - ax.get_ylim()[0])/rapport







plt.rcParams["axes.prop_cycle"] = cycler('color', 
                    ['#332288', '#CC6677', '#DDCC77', '#117733', '#88CCEE', '#AA4499', 
                     '#44AA99', '#999933', '#882255', '#661100', '#6699CC', '#AA4466']) # this sets up the color palette to be used for plotting lines

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']



# Initiating variables to which we will atribute peak caractéristics
pic = {}
pic['line']=[] # List containing matplotlib.Line2D object for each peak
pic['A']=[] # List that will contain amplitudes of each peak
pic['center']=[] # List that will contain central positions of each peak
pic['w']=[] # List containing widths
pic['fill']=[]
it=0 # Iterator used normally for counting right clicks (each right click launches the plot of the cumulative curbe)
sum_peak=[] # List of cumulated graphs (used later for updating while removing previous one)
peaks_present = 0
cid3=[]
scroll_count=0 # counter to store the cumulative values of scrolling
indice = 0
artists = []
clicked_indice = -1


plt.ioff()

def onclick(event):
    global it, peaks_present, scroll_count, indice, x_size, y_size, clicked_indice, x, block
    if event.inaxes == ax:
        if event.button==1:
            click_in_artist = [artist.contains(event)[0] for artist in artists]
            if not any(click_in_artist):
                indice += 1
                peaks_present += 1
                artists.append(ax.add_artist(Ellipse((event.xdata, event.ydata), x_size, y_size, alpha=0.5, picker=max(x_size, y_size), gid=indice)))
                A = 1.27*x_size*event.ydata
                center=event.xdata
                yy = pv(x=x, A=A, center=center, w=x_size)
                #ax.set_ylim(auto=True)
                pic['A'].append(A)
                pic['center'].append(center)
                pic['line'].append(ax.plot(x, yy, alpha=0.75, lw=2.5, picker=5))
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
                #print(ax.artists)




        elif event.button == 3 and not event.step: # On my laptop middle click and right click have the same values (?!)
            if it>0: # Checks if there is already a cumulated graph plotted
                ax.lines.remove(sum_peak[-1][0]) # remove the last cumulated graph from the figure
                sum_peak.pop()
            # Sum all the y values from all the peaks:
            sumy = np.sum(np.asarray([pic['line'][i][0].get_ydata() for i in range(len(pic['line'][:]))]), axis=0)
            if sumy.shape == x.shape: # Added this condition for the case where you removed all peaks, but the cumulated graph is left, then right-clicking need to remove that one
                sum_peak.append(ax.plot(x, sumy, '--', color='lightgreen', lw=3, alpha=0.6)) # plot the cumulated graph 
                it+=1 # One cumulated graph added
            else:
                it-=1 # if you right clicked on the graph with no peaks, you removed the cumulated graph as well
            fig.canvas.draw_idle()

        if event.step != 0:
            if peaks_present:
                peak_identifier = -1 # means that scrolling will only affect the last plotted peak 
                '''(this may change in the future so to permit the user to modify whatewer peak's widht he wishes to adjust)
                This however turns out to be a bit too difficult to acheive. For now, I'll settle with this version, where, if you want to readjust some previously placed peak, you need in fact to repace it with a new one.
                (you can first add the new one on the position that you think is better, adjust it's width, and then remove the one you didn't like by clicking on it's top)'''


                scroll_count += x_size*event.step/15 # This adjust the "speed" of width change with scrolling (event.step is always +-1)

                #ax.collections.pop(peak_identifier)

                if scroll_count > -x_size+0.01:
                    w2 = x_size + scroll_count
                else:
                    w2 = 0.01
                    scroll_count = -x_size+0.01 # This doesn't allow you to sroll to negative values (basic width is x_size)

                center2=pic['center'][peak_identifier]
                A2 = pic['A'][peak_identifier]*w2/x_size # In order to keep the Height unchanged (A (from amplitude), as defined here, is proportional to the area)
                
                pic['w'][peak_identifier]= w2
                yy = pv(x=x, center=center2, A=A2, w=w2)
                active_line = pic['line'][peak_identifier][0]
                active_line.set_ydata(yy) # This updates the values on the peak identified by "peak_identifier" (last peak for -1). 
                ax.draw_artist(active_line)
                if peak_identifier > -1:
                    cycle_indice = peak_identifier
                else:
                    cycle_indice = indice
                #pic['fill'].append(ax.fill_between(x, 0, yy, alpha=0.3, color=cycle[cycle_indice]))
                fig.canvas.draw_idle()
        #print(f'axartists = {ax.artists}\n artists = {artists}\n\n{pic}\n\n axlines = {ax.lines}\n{"kraj":=<60s}\n\n')
        #print(pic)
        if event.button != 1 and event.dblclick:
            block = True
            fig.canvas.mpl_disconnect(cid)
            fig.canvas.mpl_disconnect(cid2)
            plt.close()
            return
ax.set_title('Left-click to add/remove peaks, Scroll to adjust width, Right-click to draw sum, Double-Right-Click when done')
cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid2 = fig.canvas.mpl_connect('scroll_event', onclick)#scroll)
#plt.show(block=True)
while not 'block' in locals():
    plt.waitforbuttonpress(timeout=-1)
#display(output)


# In[7]:



manualfit_components = [copy(line[0].get_data()[1]) for line in pic['line']] # recovering the y values of each peak (not really useful)

N_peaks = len(manualfit_components)

manualfit_components_params =  copy(list(map(list, zip([A*w/x_size for A, w in list(zip(pic['A'], pic['w']))], pic['center'], pic['w'], [0]*N_peaks, [0.5]*N_peaks)))) # creating the list of initial parameters from your manual input (as a list of lists)
manualfit_components_params = list(chain(*manualfit_components_params)) # to transform the list of lists into one single list

manualfit = np.sum(np.asarray(manualfit_components), axis=0) # the sum of manually created peaks

    


# In[7]:


# Setting the bounds based on your input (you can play with this part if you feel like it, but leaving it as it is should be ok) 

# set the initial bounds as infinities:
upper_bounds = np.ones(len(manualfit_components_params))*np.inf
lower_bounds = np.ones(len(manualfit_components_params))*(-np.inf)

# setting reasonable bounds for the peak amplitude as a portion to your manual estimate 
upper_bounds[0::5]=[A*1.2 for A in manualfit_components_params[0::5]]
lower_bounds[0::5]=[A*0.8 for A in manualfit_components_params[0::5]]

# setting reasonable bounds for the peak position as a shift to your manual position 
upper_bounds[1::5]=[shift + x_size for shift in manualfit_components_params[1::5]]
lower_bounds[1::5]=[shift - x_size for shift in manualfit_components_params[1::5]]

# setting the bounds for the widths
upper_bounds[2::5]=[width*1.10 for width in manualfit_components_params[2::5]]
lower_bounds[2::5]=[width*0.9 for width in manualfit_components_params[2::5]]

# setting the bounds for the offsets
upper_bounds[3::5]=y_size*5
lower_bounds[3::5]=0

# setting the bounds for the lorentz/gauss ratio
upper_bounds[4::5]=1
lower_bounds[4::5]=0


bounds = (lower_bounds, upper_bounds)


# In[7]:


# The fitting part:    
a, b = curve_fit(fitting_function, x, y, p0=manualfit_components_params, bounds=bounds)
# Strangely enough, adding the bounds does not improve significantly the efficiency of the procedure. 
# Note however that since the offset parametar here doesn't really play any role, not using the bounds will give you some strange values for it.


figg, axx, = plt.subplots(figsize=(16, 12))
figg.subplots_adjust(bottom=0.2)
donja = figg.add_axes([0.125, 0.05, 0.775, 0.1])
donja.set_facecolor('w')

axx.plot(x, fitting_function(x, *manualfit_components_params), '--r', alpha=0.5, label='initial manual fit')
axx.plot(x, y, 'k', label='original data')
axx.plot(x, fitting_function(x, *a), '--g', lw=4, alpha=0.6, label='after optimization')
axx.legend()

donja.scatter(x, y-fitting_function(x, *a))
donja.set_ylabel('error')
axx.set_title('After fitting')
plt.show(block=False)

# In[8]:


greska = np.sqrt(np.diag(b))
plt.figure(figsize=(16,12))
plt.plot(x, y, lw=4, c='k')
#plt.plot(manualfit, '--y', lw=3, alpha=0.6)
for i in range(len(manualfit_components)):
    plt.plot(x, pv(x, *a[i*5:i*5+5])-a[i*5+3], alpha=0.5, label=[f"{prd:.2f}\U000000B1{krk:.1f}" for prd, krk in list(zip(a[i*5:i*5+5], greska[i*5:i*5+5]))]) # offset is offset :)
plt.legend()
plt.title('Showing the individual peaks as found by fitting procedure')

plt.show(block=False)
parametar_names = ['Amplitude', 'Center', 'Width', 'Offset', 'Ratio Lorenz/Gauss']
for i in range(len(a)):
    print(f"Peak_{i//5} - Manually found  {parametar_names[i%5]:<24s}: {manualfit_components_params[i]:8.2f}  ------->  After fitting :   {a[i]:8.2f} \U000000B1 {greska[i]:8.2f} ::::::: dummy params : {dummy_params[i]:8.2f}")


