#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Tue Jun 11 15:28:47 2019

@author: dejan
"""
import numpy as np
from joblib import Parallel, delayed
from warnings import warn
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Ellipse
from scipy import sparse
from scipy.optimize import minimize_scalar
from skimage import io, transform


def find_barycentre(x, y, method="trapz_minimize"):
    '''Calculates the index of the barycentre value.
        Parameters:
        ----------
        x:1D ndarray: ndarray containing your raman shifts
        y:1D ndarray: Ndarray containing your intensity (counts) values
        method:string: only "trapz_minimize" for now
        Returns:
        ---------
        (x_value, y_value): the coordinates of the barycentre
        '''
    assert(method in ['trapz_minimize'])
    half = np.trapz(y, x=x)/2
    if method in 'trapz_minimize':
        def find_y(Y0, xx=x, yy=y, method=method):
            '''Internal function to minimize
            depending on the method chosen'''
            # Calculate the area of the curve above the Y0 value:
            part_up = np.trapz(yy[yy >= Y0] - Y0, x=xx[yy >= Y0])
            # Calculate the area below Y0:
            part_down = np.trapz(yy[yy <= Y0], x=xx[yy <= Y0])
            # for the two parts to be the same
            to_minimize_ud = np.abs(part_up - part_down)
            # fto make the other part be close to half
            to_minimize_uh = np.abs(part_up - half)
            # to make the other part be close to half
            to_minimize_dh = np.abs(part_down - half)
            return to_minimize_ud**2 + to_minimize_uh + to_minimize_dh

        def find_x(X0, xx=x, yy=y, method=method):
            part_left = np.trapz(yy[xx <= X0], x=xx[xx <= X0])
            part_right = np.trapz(yy[xx >= X0], x=xx[xx >= X0])
            to_minimize_lr = np.abs(part_left - part_right)
            to_minimize_lh = np.abs(part_left - half)
            to_minimize_rh = np.abs(part_right - half)
            return to_minimize_lr**2 + to_minimize_lh + to_minimize_rh

        minimized_y = minimize_scalar(find_y, method='Bounded',
                                      bounds=(np.quantile(y, 0.01),
                                              np.quantile(y, 0.99)))
        minimized_x = minimize_scalar(find_x, method='Bounded',
                                      bounds=(np.quantile(x, 0.01),
                                              np.quantile(x, 0.99)))
        y_value = minimized_y.x
        x_value = minimized_x.x

    elif method == "list_minimize":
        yy = y
        xx = x
        ys = np.sort(yy)
        z2 = np.asarray(
            [np.abs(np.trapz(yy[yy<=y_val], x=xx[yy<=y_val]) -\
                    np.trapz(yy[yy>=y_val]-y_val, x=xx[yy>=y_val]))\
             for y_val in ys])
        y_value = ys[np.argmin(z2)]
        x_ind = np.argmin(np.abs(np.cumsum(yy) - np.sum(yy)/2)) + 1
        x_value = xx[x_ind]

    return x_value, y_value


def baseline_als(y, lam=1e5, p=5e-5, niter=12):
    '''Adapted from:
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    To get the feel on how the algorithm works, you can think of it as
    if the rolling ball which comes from beneath the spectrum and thus sets
    the baseline.

    Then, further following the image, schematic explanaton of the params would be:

    Params:
    ----------
        y: 1D or 2D ndarray of floats
            the spectra on which to find the baseline

        lam: float
            Can be viewed as the radius of the ball.
            As a rule of thumb, this value should be something like
            ten times the width of the broadest feature you want to keep
            (width is to be measured in number of points, since
            for the moment no x values are taken into account
            in this algorithm)

        p: float
            Can be viewed as the measure of how much the ball
            can penetrate into the spectra from below

        niter: int
            number of iterations
           (the resulting baseline should stabilize after
            some number of iterations)

    Returns:
    -----------
        b_line:ndarray: the baseline (same shape as y)

    Note:
    ----------
        It takes around 2-3 sec per 1000 spectra with 10 iterations
        on i7 4cores(8threads) @1,9GHz

    '''
    def _one_bl(yi, lam=lam, p=p, niter=niter, z=None):
        if z is None:
            L = yi.shape[-1]
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
            w = np.ones(L)
            W = sparse.spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w)  # Do not create a new matrix, just update diagonal values
            Z = W + D
            z = sparse.linalg.spsolve(Z, w*yi)
            w = p * (yi > z) + (1-p) * (yi < z)
        return z

    if y.ndim == 1:
        b_line = _one_bl(y)
    elif y.ndim == 2:
        b_line = np.asarray(Parallel(n_jobs=-1)(delayed(_one_bl)(y[i])
                                                for i in range(y.shape[0])))
    else:
        warn("This only works for 1D or 2D arrays")

    return b_line


def slice_lr(spectra, sigma=None, pos_left=None, pos_right=None):
    '''
    Several reasons may make you want to apply the slicing.

    a) Your spectra might have been recorded with the dead pixels included.
    It is normaly a parameter which should had been set at the spectrometer
    configuration (Contact your spectros's manufacturer for assistance)
    b) You might want to isolate only a part of the spectra which
    interests you.
    c) You might have made a poor choice of the spectral range at the
       moment of recording the spectra.

    Parameters:
    ---------------
    spectra: N-D ndarray: your spectra. The last dimension is corresponds
                          to one spectrum recorded at given position
    sigma: 1D ndarray: your Raman shifts. Default is None, meaning
                       that the slicing will be applied based on the
                       indices of spectra, not Raman shift values
    pos_left :int or float: position from which to start the slice. If sigma
                      is given, pos_left is the lower Raman shift value,
                      if not, it's the lower index of the spectra.
    pos-right:int or float: same as for pos_left, but on the right side.
                            It can be negative (means you count from the end)

    Returns:
    ---------------
    spectra_kept: N-D ndarray: your spectra containing only the zone of
                              interest.
                              spectra_kept.shape[:-1] = spectra_shape[:-1]
                              spectra_kept.shape[-1] <= spectra.shape[-1]
    sigma_kept: 1D ndarray: if sigma is given: your Raman shift values for the
                            isolated zone.
                            len(sigma_kept)=spectra_kept.shape[-1] <=
                            len(sigma)=spectra.shape[-1]
                            if sigma is not given: indices of the zone of
                            interest.
    '''

    if sigma is None:
        sigma = np.arange(spectra.shape[-1])

    # If you pass a negative number as the right position:
    if isinstance(pos_right, (int, float)):
        if pos_right < 0:
            pos_right = sigma[pos_right]

    if pos_left is None:
        pos_left = sigma.min()
    if pos_right is None:
        pos_right = sigma.max()

    assert pos_left <= pos_right, "Check your initialization Slices!"
    _condition = (sigma >= pos_left) & (sigma <= pos_right)
    sigma_kept = sigma[_condition]  # add np.copy if needed
    spectra_kept = np.asarray(spectra[..., _condition], order='C')

    return spectra_kept, sigma_kept


def pV(x: np.ndarray, h: float, x0: float = None,
       w: float = None, factor: float = 0.5):
    '''Creates an pseudo-Voigt profile.

    Parameters:
    ------------
    x : 1D ndarray
        Independent variable (Raman shift for ex.)
    h : float
        height of the peak
    x0 : float
        The position of the peak on the x-axis.
        Default value is at the middle of the x
    w : float
        FWHM - The width
        Default value is 1/3 of the x
    factor : float
        The ratio of Gauss vs Lorentz in the peak
        Default value is 0.5

    Returns:
    --------------
    y : np.ndarray :
        1D-array of the same length as the input x-array
    ***************************

    Example :
    --------------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(150, 1300, 1015)
    >>> plt.plot(x, pV(x, 200))
    '''

    def Gauss(x, w):
        return((2/w) * np.sqrt(np.log(2)/np.pi) * np.exp(
                -(4*np.log(2)/w**2) * (x - x0)**2))

    def Lorentz(x, w):
        return((1/np.pi)*(w/2) / (
                (x - x0)**2 + (w/2)**2))

    if x0 is None:
        x0 = x[int(len(x)/2)]
    if w is None:
        w = (x.max() - x.min()) / 3

    intensity = h * np.pi * (w/2) /\
        (1 + factor * (np.sqrt(np.pi*np.log(2)) - 1))

    return(intensity * (factor * Gauss(x, w)
                        + (1-factor) * Lorentz(x, w)))


def multi_pV(x, *params, peak_function=pV):
    '''
    This function returns the spectra as the sum of the pseudo-Voigt peaks,
    given the independent variable `x` and a set of parameters for each peak.
    (one sublist for each Pseudo-Voigt peak).

    Parameters :
    -----------------
    x : np.ndarray
        1D ndarray - independent variable.
    *params : list[list[float]]
        The list of lists containing the peak parameters. For each infividual
        peak to be created there should be a sublist of parameters to be
        passed to the pV function. So that `params` list finally contains
        one of these sublists for each Pseudo-Voigt peak to be created.
        Look in the docstring of pV function for more info on theese params.

    Returns :
    -----------------
    y : np.ndarray
        1D ndarray of the same length as the input x-array

    Example :
    -----------------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(150, 1300, 1015) # Create 1015 equally spaced points
    >>> mpar = [[40, 220, 100], [122, 440, 80], [164, 550, 160], [40, 480, 340]]
    >>> plt.plot(x, multi_pV(x, *mpar))
    '''
    result = np.zeros_like(x, dtype=np.float)
    n_peaks = int((len(params)+0.1)/4)  # Number of peaks
    ipp = np.asarray(params).reshape(n_peaks, 4)
    for pp in ipp:
        result += peak_function(x, *pp)  # h, x0, w, r)
    return result


def create_multiple_spectra(x: np.ndarray, *initial_peak_params: list,
                            N=10000, noise: float = 0.02,
                            spectrum_function=multi_pV,
                            noise_bias='linea', funny_peak='random'):
    """Creates N different spectra using mutli_pV function.

    Params:
    ----------------
        x : np.ndarray
            1D ndarray - independent variable
        initial_peak_params: list
            The list of sublists containing individual peak parameters as
            demanded by the `spectrum_function`.
        defaults: list, optional
            Default params for the sublists where not all params are set.
            The function will try to come up with something if the defaults
            are not provided.
        N : int, optional
            The number of spectra to create. Defaults to 1024 (32x32 :)
        noise : float
            Noisiness and how much you want the spectra to differ between them.
        spectrum_function : function
            The default is multi_pV.
            You should be able to provide something else, but this is not yet
            tested.
        noise_bias: None or 'smiley' or 'linea'
            Default is 'linea'.
            The underlaying pattern of the differences between the spectra.
        funny_peak: int or list of ints or 'random' or 'all'
            Only applicable if `noise_bias` is 'smiley'.
            Designates the peak on which you want the bias to appear.
            If 'random', one peak is chosen randomly.

    Returns:
    -----------------
    y : np.ndarray :
        2D ndarray of the shape (N, len(x))

    Example:
    --------------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(150, 1300, 1015)  # Create 1015 equally spaced points
    >>> mpar = [[40, 220, 100], [122, 440, 80], [164, 550, 160], [40, 480, 340]]
    >>> my_spectra = create_multiple_spectra(x, mpar)
    """

    def binarization_load(f, shape=(132, 132)):
        '''May be used if "linea" mode is active'''
        im = io.imread(f, as_gray=True)
        return transform.resize(im, shape, anti_aliasing=True)

    n_peaks = int((len(initial_peak_params) + 0.1) / 4)  # Number of peaks
    ipp = initial_peak_params.reshape(n_peaks, 4)
    ponderation = 1 + (np.random.rand(N, n_peaks, 1) - 0.5) * noise
    peaks_params = ponderation * ipp
    # -------- The funny part ----------------------------------
    if noise_bias == 'smiley':
        smile = io.imread('./misc/bibi.jpg')
        x_dim = int(np.sqrt(N))
        y_dim = N//x_dim
        print(f"You'll end up with {x_dim}*{y_dim} = {x_dim*y_dim} points"
              f"instead of initial {N}")
        N = x_dim * y_dim
        smile_resized = transform.resize(smile, (x_dim, y_dim))
        noise_bias = smile_resized.ravel()
        if funny_peak == 'random':
            funny_peak = np.random.randint(0, n_peaks+1)
        elif funny_peak == 'all':
            funny_peak = list(range(n_peaks))
        peaks_params[:, funny_peak, 0] *= noise_bias
    elif noise_bias == 'linea':
        x_dim = int(np.sqrt(N))
        y_dim = N//x_dim
        images = './misc/linea/*.jpg'
        coll_all = io.ImageCollection(images, load_func=binarization_load,
                                      shape=(x_dim, y_dim))
        print(f"You'll end up with {x_dim}*{y_dim} = {x_dim*y_dim} points"
              f"instead of initial {N}")
        N = x_dim * y_dim
    # -------- The End of the funny part ------------------------
    additive_noise = peaks_params[:, :, 0].mean() *\
        (0.5 + np.random.rand(len(x))) / 5
    spectra = np.asarray(
               [multi_pV(x, peaks_params[i]) +
                additive_noise[np.random.permutation(len(x))]
                for i in range(N)])
    if isinstance(noise_bias, str) and noise_bias == 'linea':
        noise_bias = coll_all.concatenate().reshape(110, -1)
        spectra[:, -110:] *= noise_bias.T

    return spectra.reshape(N, -1)


class AllMaps(object):
    '''
    Allows one to rapidly visualize maps of Raman spectra.
    You can also choose to visualize the map and plot the
    corresponding component side by side if you set the
    "components" parameter.

    Parameters:
        map_spectra:3D ndarray
            the spectra shaped as (n_lines, n_columns, n_wavenumbers)
        sigma:1D ndarray : an array of wavenumbers (len(sigma)=n_wavenumbers)
        components: 2D ndarray
            The most evident use-case would be to
            help visualize the decomposition results from PCA or NMF.
            In this case, the function will plot the component with the
            corresponding map visualization of the given components'
            presence in each of the points in the map.
            So, in this case, your map_spectra would be for example
            the matrix of components' contributions in each spectrum,
            while the "components" array will be your actual components.
            In this case you can ommit your sigma values or set them to
            something like np.arange(n_components)
        components_sigma: 1D ndarray
            in the case explained above, this would be the
            actual wavenumbers
        **kwargs: dict
            can only take 'title' as a key for the moment

        Returns: The interactive visualization (you can scroll through
            sigma values with a slider, or using left/right keyboard arrows)
    '''

    def __init__(self, map_spectra, sigma=None, components=None,
                 components_sigma=None, **kwargs):
        self.map_spectra = map_spectra
        if sigma is None:
            self.sigma = np.arange(map_spectra.shape[-1])
        else:
            assert map_spectra.shape[-1] == len(sigma), "Check your Ramans shifts array"
            self.sigma = sigma
        self.first_frame = 0
        self.last_frame = len(self.sigma)-1
        if components is not None:
            #assert len(components) == map_spectra.shape[-1], "Check your components"
            self.components = components
            if components_sigma is None:
                self.components_sigma = np.arange(components.shape[-1])
            else:
                self.components_sigma = components_sigma
        else:
            self.components = None
        if components is not None:
            self.fig, (self.ax2, self.ax, self.cbax) =\
                plt.subplots(ncols=3, gridspec_kw={'width_ratios': [40, 40, 1]})
            self.cbax.set_box_aspect(40 * self.map_spectra.shape[0] /
                                     self.map_spectra.shape[1])
        else:
            self.fig, (self.ax, self.cbax) =\
                plt.subplots(ncols=2, gridspec_kw={'width_ratios': [40, 1]})
            self.cbax.set_box_aspect(40 * self.map_spectra.shape[0] /
                                     self.map_spectra.shape[1])
            #self.cbax = self.fig.add_axes([0.92, 0.3, 0.03, 0.48])
        # Create some space for the slider:
        self.fig.subplots_adjust(bottom=0.19, right=0.89)
        self.title = kwargs.get('title', None)

        self.im = self.ax.imshow(self.map_spectra[:, :, 0])
        self.im.set_clim(np.percentile(self.map_spectra[:, :, 0], [1, 99]))
        if self.components is not None:
            self.line, = self.ax2.plot(self.components_sigma, self.components[0])
            self.ax2.set_box_aspect(self.map_spectra.shape[0] /
                                    self.map_spectra.shape[1])
            self.ax2.set_title(f"Component {0}")
        self.titled(0)
        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = self.fig.add_axes([0.15, 0.1, 0.7, 0.03],
                                         facecolor=self.axcolor)

        self.sframe = Slider(self.axframe, 'Frame',
                             self.first_frame, self.last_frame,
                             valinit=self.first_frame, valfmt='%d', valstep=1)

        self.my_cbar = mpl.colorbar.colorbar_factory(self.cbax, self.im)
        # calls the "update" function when changing the slider position:
        self.sframe.on_changed(self.update)
        # Calling the "press" function on keypress event
        # (only arrow keys left and right work)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.show()

    def titled(self, frame):
        if self.components is None:
            if self.title is None:
                self.ax.set_title(f"Raman shift = {self.sigma[frame]:.1f}cm⁻¹")
            else:
                self.ax.set_title(f"{self.title} n°{frame}")
        else:
            self.ax2.set_title(f"Component {frame}")
            if self.title is None:
                self.ax.set_title(f"Component n°{frame} contribution")
            else:
                self.ax.set_title(f"{self.title} n°{frame}")

    def update(self, val):
        '''This function is for using the slider to scroll through frames'''
        frame = int(self.sframe.val)
        img = self.map_spectra[:, :, frame]
        self.im.set_data(img)
        self.im.set_clim(np.percentile(img, [1, 99]))
        if self.components is not None:
            self.line.set_ydata(self.components[frame])
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.titled(frame)
        self.fig.canvas.draw_idle()

    def press(self, event):
        '''This function is to use arrow keys left and right to scroll
        through frames one by one'''
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < len(self.sigma)-1:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        img = self.map_spectra[:, :, new_frame]
        self.im.set_data(img)
        self.im.set_clim(np.percentile(img, [1, 99]))
        self.titled(new_frame)
        if self.components is not None:
            self.line.set_ydata(self.components[new_frame])
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.fig.canvas.draw_idle()


# %%
def set_size(variable, rapport=70):
    return (variable.max() - variable.min())/rapport


class fitonclick(object):
    '''This class is used to interactively draw pseudo-voigt (or other type)
    peaks, on top of your data.
    It was originaly created to help defining initial fit parameters to
    pass on to SciPy CurveFit.
    IMPORTANT! See the Example below, to see how to use the class
    Parameters:
        x: independent variable
        y: your data
        initial_GaussToLorentz_ratio:float between 0 and 1, default=0.5
            Pseudo-Voigt peak is composed of a Gaussian and of a Laurentzian
            part. This ratio defines the proportion of those parts.
        scrolling_speed: float>0, default=1
            defines how quickly your scroling widens peaks
        initial_width: float>0, default=5
            defines initial width of peaks
        **kwargs: dictionary, for exemple {'figsize':(9,9)}
            whatever you want to pass to plt.subplots(**kwargs)
    Returns:
        Nothing, but you can access the atributes using class instance, like
        fitonclick.pic: a dict containing the parameters of each peak added
        fitonclick.sum_peak: list containing cumulated graph line
            to get the y-values, use sum_peak[-1][0].get_ydata()
        fitonclick.peak_counter: int giving the number of peaks present
        etc.

    Example:
        >>>my_class_instance = fitonclick(x, y)
        >>>while my_class_instance.block:
        >>>    plt.waitforbuttonpress(timeout=-1)

    '''

    def __init__(self, x, y,
                 initial_GaussToLoretnz_ratio=0.5,
                 scrolling_speed=1,
                 initial_width=5,
                 **kwargs):
        plt.ioff()
        self.x = x
        self.y = y
        self.GL = initial_GaussToLoretnz_ratio
        self.scrolling_speed = scrolling_speed
        self.initial_width = initial_width
        self.manualfit_spectra = None
        # Initiating variables to which we will atribute peak caractéristics:
        self.pic = {}
        self.pic['line'] = []  # List containing matplotlib.Line2D object for each peak
        self.pic['h'] = []  # List that will contain heights of each peak
        self.pic['x0'] = []  # List that will contain central positions of each peak
        self.pic['w'] = []  # List containing widths
        self.pic['GL'] = []  # List containing the Gauss / Lorentz ratios
        # List of cumulated graphs
        # (used later for updating while removing previous one)
        self.sum_peak = []
        self.peak_counter: int = 0  # number of peaks on the graph
        self.cum_graph_present: int = 0  # only 0 or 1
        self.scroll_count = 0.  # counter to store the cumulative values of scrolling
        self.artists = []  # will be used to store the elipses on tops of the peaks
        self.block = True

        # Setting up the plot:
        self.fig, self.ax = plt.subplots(**kwargs)
        self.ax.plot(self.x, self.y,
                     linestyle='none', marker='o', c='k', ms=4, alpha=0.5)
        self.ax.set_title('Left-click to add/remove peaks; '
                          'Scroll to adjust width, \nRight-click to draw sum,'
                          ' Press "Enter" when done')
        self.x_size = set_size(self.x)
        self.y_size = 2 * set_size(self.y)
        self.cid = self.fig.canvas.mpl_connect('button_press_event',
                                               self.onclick)
        self.cid2 = self.fig.canvas.mpl_connect('scroll_event', self.onclick)
        self.cid3 = self.fig.canvas.mpl_connect("key_press_event", self.end_i)
        plt.show()

    def _add_peak(self, event):
        self.peak_counter += 1
        h = event.ydata
        x0 = event.xdata
        yy = pV(x=self.x, h=h, x0=x0,
                w=self.x_size*self.initial_width, factor=self.GL)
        one_elipsis = self.ax.add_artist(
                        Ellipse((x0, h),
                                self.x_size, self.y_size, alpha=0.5,
                                gid=str(self.peak_counter)))
        self.artists.append(one_elipsis)
        self.pic['line'].append(self.ax.plot(self.x, yy,
                                alpha=0.75, lw=2.5))
        self.pic['line'][-1][0].set_pickradius(5)
        # Is the above line necessary? (picker shoud work only on artists?)
        # ax.set_ylim(auto=True)
        self.pic['h'].append(h)
        self.pic['x0'].append(x0)
        self.pic['w'].append(self.x_size*self.initial_width)
        self.fig.canvas.draw_idle()
#        return(self.artists, self.pic)

    def _adjust_peak_width(self, event, peak_identifier=-1):
        """Adjust the peak width by scrolling the mouse."""

        self.scroll_count += self.x_size * np.sign(event.step) *\
                             self.scrolling_speed/10
        if self.scroll_count > -self.x_size*self.initial_width*0.999:
            w2 = self.x_size*self.initial_width + self.scroll_count
        else:
            w2 = self.x_size * self.initial_width / 1000
            # This doesn't allow you to sroll to negative values
            # (basic width is x_size)
            self.scroll_count = -self.x_size * self.initial_width * 0.999

        center2 = self.pic['x0'][peak_identifier]
        h2 = self.pic['h'][peak_identifier]
        self.pic['w'][peak_identifier] = w2
        yy = pV(x=self.x, x0=center2, h=h2, w=w2, factor=self.GL)
        active_line = self.pic['line'][peak_identifier][0]
        # This updates the values on the peak identified
        active_line.set_ydata(yy)
        self.ax.draw_artist(active_line)
        self.fig.canvas.draw_idle()
#        return(scroll_count, pic)

    def _remove_peak(self, clicked_indice):
        self.artists[clicked_indice].remove()  # remove as mpl artist
        self.artists.pop(clicked_indice)  # remove from the list
        self.ax.lines.remove(self.pic['line'][clicked_indice][0])
        self.pic['line'].pop(clicked_indice)
        self.pic['x0'].pop(clicked_indice)
        self.pic['h'].pop(clicked_indice)
        self.pic['w'].pop(clicked_indice)
        self.fig.canvas.draw_idle()
        self.peak_counter -= 1
#        return(artists, pic)

    def _draw_peak_sum(self):
        if self.peak_counter < 1:
            return

        def _remove_sum(self):
            assert self.cum_graph_present == 1, "no sum drawn, nothing to remove"
            self.ax.lines.remove(self.sum_peak[-1][0])
            self.sum_peak.pop()
            self.cum_graph_present -= 1
#            return sum_peak

        def _add_sum(self, sumy):
            assert sumy.shape == self.x.shape, "something's wrong with your data"
            self.sum_peak.append(self.ax.plot(self.x, sumy, '--',
                                              color='lightgreen',
                                              lw=3, alpha=0.6))
            self.cum_graph_present += 1
#            return sum_peak

        # Sum all the y values from all the peaks:
        self.manualfit_spectra = np.sum(np.asarray(
                                          [self.pic['line'][i][0].get_ydata()
                                           for i in range(self.peak_counter)]),
                                        axis=0)
        # Check if there is already a cumulated graph plotted:
        if self.cum_graph_present == 1:
            # Check if the sum of present peaks correponds to the cumulated graph
            if not np.array_equal(self.sum_peak[-1][0].get_ydata(),
                                  self.manualfit_spectra):
                # if not, remove the last cumulated graph from the figure:
                _remove_sum(self)
                # and then plot the new cumulated graph:
                _add_sum(self, sumy=self.manualfit_spectra)
        # No cumulated graph present:
        elif self.cum_graph_present == 0:
            # plot the new cumulated graph
            _add_sum(self, sumy=self.manualfit_spectra)

        else:
            raise("WTF?")
        self.fig.canvas.draw_idle()
#        return(cum_graph_present, sum_peak)

    def onclick(self, event):
        if event.inaxes == self.ax:  # if you click inside the plot
            if event.button == 1:  # left click
                # Create list of all elipes and check if the click was inside:
                click_in_artist = [art.contains(event)[0]
                                   for art in self.artists]
                if any(click_in_artist):  # if click was on any of the elipsis
                    # identify the one we clicked
                    clicked_indice = click_in_artist.index(True)
                    self._remove_peak(clicked_indice=clicked_indice)
                else:  # if click was not on any of the already drawn elipsis
                    self._add_peak(event)
            elif event.step:  # if it's a mouse scroll event
                if self.peak_counter:  # if there are any peaks
                    self._adjust_peak_width(event, peak_identifier=-1)
                    # peak_identifier = -1 means that scrolling will
                    # only affect the last plotted peak

            elif event.button != 1 and not event.step:
                # So, basically, right or middle click both draw the sum:
                self._draw_peak_sum()

                if event.dblclick:  # double, not left, click
                    # ATTENTION:
                    # doubleclick seems not to work with certain backends (?)
                    # Double Middle (or Right?) click ends the show
                    event.key = "enter"
                    self.end_i(event)

    def end_i(self, event):
        if event.key == "enter":
            self.pic['GL'] = [self.GL] * self.peak_counter
            self.manualfit_params = np.array(list(self.pic['h']) +
                                             list(self.pic['x0']) +
                                             list(self.pic['w']) +
                                             list(self.pic['GL'])
                                             )
            self.manualfit_params = self.manualfit_params.reshape(-1,
                                                                  self.peak_counter).T
            self.manualfit_params = self.manualfit_params.ravel()
            if self.manualfit_spectra is None:  # even if not drawn
                self.manualfit_spectra = np.sum(np.asarray(
                                          [self.pic['line'][i][0].get_ydata()
                                           for i in range(self.peak_counter)]),
                                        axis=0)
            self.fig.canvas.mpl_disconnect(self.cid)
            self.fig.canvas.mpl_disconnect(self.cid2)
            self.fig.canvas.mpl_disconnect(self.cid3)
            self.block = False


def set_bounds(initial_params,
               A=(0.5, 1.4, "multiply"),
               x=(-20, 20, "add"),
               w=(0.5, 16, "multiply"),
               gl=(0, 1, "absolute")):
    """Define the bounds based on the initial parameters.

    Parameters:
    -----------
    initial_params: list of lists or a 2D numpy array
        ! Warning: the order is important
        for the pseudo-voigt as defined in utilities.py `pV` function it is:
        [[A_1, x_1, w_1, gl_1],
         [A_2, x_2, ....],
         ...
         [A_N, x_N, w_N, gl_N]]
        where N is the number of peaks, and for each peak:
            A: amplitude
            x: position
            w: width (at mid-height)
            gl: gaussian to lorentzian ratio (from 0 to 1)
    A, x, w, gl: tuples of form (lower: float, upper: float, method: str)
        Bounds relative to the initial_params.
        method can be one of ["multiply", "add", "absolute"]
        "multiply": the bounds for the given parameter are in the range
                    (param * lower, param * upper)
        "add":      the bounds for the given parameter are in the range
                    (param - lower, param + upper)
        "absolute": the bounds for the given parameter are in the range
                    (lower, upper)

    Returns:
    --------
    bounds: tuple of type (lower_bounds, upper_bounds)
        where lower_bounds and upper_bounds are 1D numpy arrays of lengths
        equal to N * n_params.
            N: number of peaks
            n_params: number of parameters needed to define one peak
        """
    # set the initial bounds as infinities:
    upper_bounds = np.ones_like(initial_params)*np.inf
    lower_bounds = np.ones_like(initial_params)*(-np.inf)
    for i, p in enumerate([A, x, w, gl]):
        if p[-1] == "multiply":
            func = np.multiply
        elif p[-1] == "add":
            func = np.add
        else:
            def func(mp, p):
                return p
        lower_bounds[:, i] = np.asarray([func(mp[i], p[0]) for mp in initial_params])
        upper_bounds[:, i] = np.asarray([func(mp[i], p[1]) for mp in initial_params])

    return (lower_bounds.ravel(), upper_bounds.ravel())


def long_correction(sigma, lambda_laser, T=30, T0=0):
    """Williams' functions for Raman spectra:
    Function computing the Long correction factor according to Long
    1977. This function can operate on numpy.ndarrays as well as on
    simple numbers.

    Parameters
    ----------
    sigma : numpy.ndarray
        Wavenumber in cm-1
    lambda_inc : float
        Laser wavelength in nm.
    T : float
        Actual temperature in °C
    T0 : float
        The temperature to which to make the correction in °C
    Returns:
    ----------
    lcorr: numpy.ndarray of the same shape as sigma

    Examples
    --------
    >>> sigma, spectra_i = deconvolution.acquire_data('my_raman_file.CSV')
    >>> corrected_spectra = spectra_i * long_correction(sigma)
    """
    c = 2.998e10                          # cm/s
    lambda_inc = lambda_laser * 1e-7      # cm
    sigma_inc = 1. / lambda_inc           # cm-1
    h = 6.63e-34                          # J.s
    T_K = 273.0 + T                        # K
    T0_K = 273.0 + T0                      # K
    kB = 1.38e-23                         # J/K
    ss = sigma_inc / sigma
    cc = h*c/kB
    return (ss ** 3 / (ss - 1) ** 4
            * (1 - np.exp(cc * sigma * (1/T_K - 1/T0_K))))


def rolling_window(trt, window_size, ax=0):
    '''
    NOTE: Due to usage of as_strided function from numpy.stride_tricks,
          the results are sometimes unpredictible.
          You have been warned :)

    Function to create the 1D rolling window of the given size, on the
    given axis. The "window" is added as the new dimension to the input array,
    this new dimension is set as the first (0) axis of the resulting array.
    Parameters:
        trt:ndarray: input array
        window_size:int: the size of the window, must be odd
        ax:int: the axis you want to roll the window on
    Returns:
        ndarray of the shape (window_size,)+trt.shape
    Example:
        test = (np.arange(90)**2).reshape(9,10)
    '''
    assert window_size % 2 != 0, "Window size must be odd integer!"
    ee = window_size//2
    arr_shape = np.asarray(trt.shape)
    # If we want the result to be of the same shape as input array,
    # we have to expand the edges.
    # Here, we just duplicate the edge values ee times
    to_prepend = np.asarray([np.take(trt, 0, axis=ax).tolist()]*ee)
    to_append = np.asarray([np.take(trt, -1, axis=ax).tolist()]*ee)
    # Then we need to reshape so that the concatanation works well:
    concat_shape = arr_shape
    concat_shape[ax] = 1
    to_prepend = to_prepend.reshape(tuple(concat_shape))
    to_append = to_append.reshape(tuple(concat_shape))
    # Concatenate:
    a = np.concatenate((to_prepend, trt, to_append), axis=ax)
    # Final shape (we are adding one new dimension at the beggining)
    shape = (window_size,) + trt.shape
    # that new axis will cycle trough the same values as the axis given with
    # the ax parameter
    strides = (trt.strides[ax],) + trt.strides
    # Return thus created array:
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)
