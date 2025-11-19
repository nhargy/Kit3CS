############################
# Name  : Kit3CS/SigTools
# Author: Nadav Hargittai
###########################

import numpy as np
from numpy import ndarray
import os
from scipy.optimize import curve_fit


# --- Fitting Functions ---
def gaussian(x, a, s, m):
    return a * np.exp(-((x-m)**2)/(2*(s**2)))

def double_gaussian(x, a1, s1, m1, a2, s2, m2):
    return gaussian(x, a1, s1, m1) + gaussian(x, a2, s2, m2)


# --- Basic Signal Tools ---
def read_spectrum(
        filepath : str, 
        cut      : int = 31
        )       -> tuple[ndarray, ndarray]:
    """
    Read the x (wavelength) and y (PE count) data from data textfile.
    """
    with open(filepath) as f:
        lines = f.readlines()[cut:]

    x = []
    y = []

    for line in lines:
        parts = line.strip().split()
        x_val = float(parts[0])
        y_val = float(parts[-1])

        x.append(x_val)
        y.append(y_val)

    return np.array(x), np.array(y)


def read_acquisition_data(
        filepath : str, 
        cut      : int  = 31,
        showkeys : bool = False
        )       -> dict: 
    """
    Read ANDOR SOLIS acquisition data from data textfile.
    """
    with open(filepath) as f:
        lines = f.readlines()[:cut]

        acq_dict = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                acq_dict[key.strip()] = value.strip()

                if showkeys == True:
                    print(key)

    return acq_dict


def get_wavelengths(
        dirpath : str
        )      -> ndarray:
    """
    Get the wavelength array of a directory of iterations
    """
    itpath = dirpath / "it_0.txt"
    x, _ = read_spectrum(itpath)

    return x


def generate_iteration_matrix(
        dirpath : str
        )      -> ndarray:
    """
    Places all the iterations from a collection into 1 matrix, where each
    row represents the PE counts from the exposure.
    """
    iteration_matrix = []
    i = 0
    while True:
        try:
            filepath = os.path.join(dirpath, f'it_{i}.txt')
            _, y     = read_spectrum(filepath)
            iteration_matrix.append(y)
            i+=1
        except:
            break

    return np.array(iteration_matrix)


def combine_iterations(
        iteration_matrix : ndarray, 
        method           : str ='mean-weighted'
        )               -> ndarray:
    """
    Combine the spectra from all the iterations from a single sample
    of a single collection, via simple averaging or weighted averaging.
    """

    # === MEAN-WEIGHTED METHOD ===
    if method == 'mean-weighted':

        # Set up empty array to collect values
        combined_array = []

        # Array of means on each column (wavelength bin)
        means_arr = np.mean(iteration_matrix, axis=0)
        std_arr   = np.std(iteration_matrix, axis=0)

        # Loop through the values
        cols = iteration_matrix.T
        for idx, col in enumerate(cols):
            w_sum = 0
            w_mean = 0
            for elm in col:
                d = abs(elm - means_arr[idx])
                w = np.exp(-d/std_arr[idx])
                w_mean+=elm * w
                w_sum+=w
            
            try:
                combined_array.append(w_mean/w_sum)
            except:
                combined_array.append(means_arr[idx])

    # === SIMPLE-AVERAGE METHOD ===
    elif method == 'simple-average':
        combined_array = np.mean(iteration_matrix, axis=0)

    return np.array(combined_array)


def scale_array(
        array  : ndarray, 
        factor : float
        )     -> ndarray:
    """
    Divide array by some factor.
    """
    scaled_array = np.divide(array, factor)
    return scaled_array


def subtract_const(
        array : ndarray,
        const : float,
        )     -> ndarray:
    """
    Subtract a constant from an array.
    """
    sub_array = np.subtract(array, const)
    return np.array(sub_array)


def subtract_array(
        array1 : ndarray,
        array2 : ndarray,
        )     -> ndarray:
    """
    Subtract one array from another.
    """
    sub_array = np.subtract(array1, array2)
    return np.array(sub_array)


def get_value_index(
        array     : ndarray,
        value     : float,
        tolerance : float = 0.6
    )            -> int:
    """
    Gets the array index of closest to a value in an array.
    """
    index = None
    for idx, elm in enumerate(array):
        if (value - tolerance) < elm < (value + tolerance):
            index = idx
            break

    return index


def calculate_signal(
        x      : ndarray, 
        y      : ndarray, 
        centre : float, 
        window : float = 40,
        p0     : tuple[float, float, float] = (150, 50, 600)
        )     -> dict:
    """
    Get the signal of the peak via simple cut and gaussian fit.
    """
    idx_left  = get_value_index(x, centre-window/2)
    idx_right = get_value_index(x, centre+window/2)

    x_cut = x[idx_left:idx_right]
    y_cut = y[idx_left:idx_right]

    cut_signal = np.trapezoid(y_cut, x_cut)

    try:
        popt, pcov = curve_fit(gaussian, x_cut, y_cut, p0=p0)
        a, s, _ = popt
        fit_signal = a * s * np.sqrt(2*np.pi)
    except:
        fit_signal = None

    signal_dict = {
        "cuts"       : [idx_left, idx_right],
        "popt"       : popt,
        "cut_signal" : cut_signal,
        "fit_signal" : fit_signal
    }

    return signal_dict