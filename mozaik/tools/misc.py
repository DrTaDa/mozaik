r"""
Various helper functions.
"""

import hashlib
import re

import numpy
import numpy.random
from numpy import pi, sqrt, exp, power


def sample_from_bin_distribution(bins, number_of_samples, seed):
    r"""
    Samples from a distribution defined by a vector the sum in. The vector doesn't have to add up to one
    it will be automatically normalized.


    Parameters
    ----------

    bins : ndarray
        The returned samples correspond to the bins in `bins` - the numpy array defining the bin distribution

    number_of_samples : int
        Number of samples to generate.

    seed : int
        The seed to use to generate the RandomState

    """
    if len(bins) == 0:
        return []

    bins = bins / numpy.sum(bins)
    if not numpy.all(bins >= 0):
        raise ValueError("probabilities are not non-negative")
    # This is exactly what numpy.random.RandomState(seed).choice(len(bins), number_of_samples, p=bins)
    # does, but without the per-call cost of creating a RandomState and of converting range() to an array.
    cdf = bins.cumsum()
    cdf /= cdf[-1]
    uniform_samples = seeded_random_state(seed).random_sample(number_of_samples)
    return cdf.searchsorted(uniform_samples, side='right')


_seeded_random_state = None


def seeded_random_state(seed):
    r"""
    Returns a numpy RandomState in the same state as numpy.random.RandomState(seed).

    The returned object is shared and re-seeded at each call (which is much cheaper
    than creating a new RandomState), so it has to be used before the next call.
    """
    global _seeded_random_state
    if _seeded_random_state is None:
        _seeded_random_state = numpy.random.RandomState(seed)
    else:
        _seeded_random_state.seed(seed)
    return _seeded_random_state


def counts_in_first_occurrence_order(samples):
    r"""
    Returns the distinct values in `samples` and the number of times each occurs,
    both in order of first occurrence, i.e. the same as
    list(Counter(samples).keys()), list(Counter(samples).values()).
    """
    values, first_index, counts = numpy.unique(samples, return_index=True, return_counts=True)
    order = numpy.argsort(first_index, kind='stable')
    return values[order], counts[order]


_normal_function_sqertofpi = sqrt(2 * pi)


def normal_function(x, mean=0, sigma=1.0):
    r"""
    Returns the value of probability density of normal distribution N(mean,sigma) at point `x`.
    """
    return numpy.exp(-numpy.power((x - mean) / sigma, 2) / 2) / (
        sigma * _normal_function_sqertofpi
    )


def find_neuron(which, positions):
    r"""
    Finds a neuron depending on which:
    'center' - the most central neuron in the sheet
    'top_right' - the top_right neuron in the sheet
    'top_left' - the top_left neuron in the sheet
    'bottom_left' - the bottom_left neuron in the sheet
    'bottom_right' - the bottom_right neuron in the sheet

    """
    minx = numpy.min(positions[0, :])
    maxx = numpy.max(positions[0, :])
    miny = numpy.min(positions[1, :])
    maxy = numpy.max(positions[1, :])

    def closest(x, y, positions):
        return numpy.argmin(
            numpy.sqrt(
                numpy.power(positions[0, :].flatten() - x, 2)
                + numpy.power(positions[1, :].flatten() - y, 2)
            )
        )

    if which == "center":
        cl = closest(minx + (maxx - minx) / 2, miny + (maxy - miny) / 2, positions)
    elif which == "top_right":
        cl = closest(maxx, maxy, positions)
    elif which == "top_left":
        cl = closest(minx, maxy, positions)
    elif which == "bottom_left":
        cl = closest(minx, miny, positions)
    elif which == "bottom_right":
        cl = closest(maxx, miny, positions)

    return cl


def _sanitize_result_directory_component(value):
    r"""
    Convert a parameter key or value into a filesystem-safe label.

    This preserves the original value for metadata files, but removes path
    separators and other characters that would otherwise fragment result
    directories into nested paths.
    """
    sanitized = re.sub(r"[\\/]+", "_", str(value))
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = re.sub(r"[^A-Za-z0-9._=-]+", "_", sanitized)
    return sanitized.strip("_")


def _shorten_result_directory_component(value, max_length=48):
    r"""
    Keep filesystem labels compact enough for nested result paths.

    Long values retain a readable prefix and append a short hash so different
    parameter values still map to distinct directory names.
    """
    sanitized = _sanitize_result_directory_component(value)
    if len(sanitized) <= max_length:
        return sanitized

    digest = hashlib.sha1(sanitized.encode("utf-8")).hexdigest()[:10]
    prefix_length = max_length - len(digest) - 1
    return sanitized[:prefix_length].rstrip("_") + "_" + digest


def result_directory_name(simulation_run_name, simulation_name, modified_parameters):
    modified_params_str = "_".join(
        [
            _shorten_result_directory_component(k, max_length=24)
            + ":"
            + _shorten_result_directory_component(modified_parameters[k])
            for k in sorted(modified_parameters.keys())
            if k != "results_dir"
        ]
    )
    if len(modified_params_str) > 100:
        modified_params_str = "_".join(
            [
                _shorten_result_directory_component(
                    str(k).split(".")[-1], max_length=20
                )
                + ":"
                + _shorten_result_directory_component(
                    modified_parameters[k], max_length=20
                )
                for k in sorted(modified_parameters.keys())
                if k != "results_dir"
            ]
        )

    return simulation_name + "_" + simulation_run_name + "_____" + modified_params_str
