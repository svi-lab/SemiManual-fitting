# SemiManual-fitting
_Allows you to visually set initial parameters to pass on to SciPy.optimize.curve_fit_

If you have an (experimental) spectra containing multiple peaks (interlapping or not), and you want to fit each peak with pseudo-voigt functions (weighted sum of gaussian and lorentzian), then this might interest you.

At the moment the script presupposes that your curve has already had the baseline substracted.

*Please note the part of the script that creates the dummy data. You should replace that part by loading your own data.*
