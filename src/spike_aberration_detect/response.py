import numpy as np
import scipy.interpolate as sp_itp
from psfsim.polychrom import PolychromaticPSF

import spike_aberration_detect.spike_finder as spikes


def build_response_matrix(
    scanum: np.int_,
    scax: np.float64,
    scay: np.float64,
    wl_band: np.ndarray,
    wl_band_name: str,
    flux: np.float64,
    aberrations,
    ideal_spike_angles: np.ndarray,
    ps_size: np.int_,
    dense_ps_size: np.int_,
    seed: np.int_,
    step: np.float64,
    bound: np.int_,
    center,
    **kwargs,
):
    """
    Builds a response matrix from a set of aberrations.

    Parameters
    ----------
    scanum : int
        Roman SCA index (1 to 18).
    scax : float
        X position on the SCA in mm.
    scay : float
        Y position on the SCA in mm.
    wl_band : float array
        array of wavelengths to be used to generate the PSFs.
    wl_band_name : str
        Name of the desired wavelength band passed to psfsim.
    flux : float
        Average total number of electrons in the image, used for resampling.
    aberrations : tuple or array of float
        The aberrations to use in the creation of the response matrix. The function
        will use them one at a time. It is highly recommended that the values are all
        not None.
    ideal_spike_angles : array of float
        Since the difference in spike angle from ideal is what matters when fitting
        for Zernike coefficients the function also requires the 12 ideal spike angles.
    ps_size : int
        Postage stamp size of the desired PSF, in native pixels.
    dense_ps_size : int
        Postage stamp size of the desired PSF after interpolation.
    seed : int
        Seed for numpy random number generation.
    step : float
        The increment angle between successive lines, in degrees.
    bound : int, optional
        The length of lines to be drawn.
    center : tuple or array of int, optional
        The point at which each line begins.
    **kwargs : other
        Extra arguments to be passed to spike_aberration_detect.spike_finder.find_spikes.
        Includes borders and threshold. This function will not work if verbose=True is
        passed.

    Returns
    -------
    response_matrix : 2D array of float
        A 12x5 response matrix that can be used to estimate aberrations given an angle change
        vector.
    """
    resp_matrix = np.zeros((12, 5))
    for i in np.arange(aberrations.size):
        extra_aberrations = [None, None, None, None, None]
        extra_aberrations[i] = aberrations[i]

        psf_lin = generate_model_psf(
            scanum,
            scax,
            scay,
            flux,
            ps_size,
            dense_ps_size,
            wl_band,
            wl_band_name,
            seed,
            extra_aberrations=extra_aberrations,
        )

        psf = np.arcsinh(psf_lin)

        spike_list = spikes.find_spikes(psf, step, bound, center, **kwargs)
        resp_matrix[:, i] = (spike_list - ideal_spike_angles) / aberrations[i]

    return resp_matrix


def generate_model_psf(
    scanum: np.int_,
    scax: np.float64,
    scay: np.float64,
    flux: np.float64,
    ps_size: np.int_,
    dense_ps_size: np.int_,
    wl_band: np.ndarray,
    wl_band_name: str,
    seed: np.int_,
    ovsamp: np.int_ = 8,
    extra_aberrations=None,
    postprocess=True,
):
    """
    Generate a PSF at a point on the detector with a flux and aberrations.

    Parameters
    ----------
    scanum : int
        Roman SCA index (1 to 18).
    scax : float
        X position on the SCA in mm.
    scay : float
        Y position on the SCA in mm.
    flux : float
        Average total number of electrons in the image, used for resampling.
    ps_size : int
        Postage stamp size of the desired PSF, in native pixels.
    dense_ps_size : int
        Postage stamp size of the desired PSF after interpolation.
    wl_band : float array
        array of wavelengths to be used to generate the PSFs.
    wl_band_name : str
        Name of the desired wavelength band passed to psfsim.
    seed : int
        Seed for numpy random number generation.
    ovsamp : int, optional
        Number of samples per native pixel in psfsim. Must be greater than 1.
    extra_aberrations : tuple or array of float
        Parameters corresponding to zernike polynomials for introducing aberrations that
        add to the optical path length and produce different aberrations. Supports up to
        5 parameters (Z2, Z3, Z4, Z5, and Z6 in that order). The effects of each polynomial
        are as follows:
        Z2: horizontal centering
        Z3: vertical centering
        Z4: focus
        Z5: astigmatism
        Z6: also astigmatism
    postprocess : boolean, optional
        Whether to postprocess the generated PSF by interpolating and poisson resampling it. Defaults
        to true. If set to false, dense_ps_size and seed are not used while making the PSF.

    Returns
    -------
    psf : 2D array of float
        The simulated point spread function.
    """
    obj = PolychromaticPSF(scanum, scax, scay, wl_band)

    pixel_size = ovsamp
    obj.compute_poly_psf(
        optical_psf_only=True,
        use_postage_stamp_size=ps_size,
        ovsamp=ovsamp,
        use_filter=wl_band_name,
        extra_aberrations=extra_aberrations,
    )

    psf = spikes.downsample_2d_image(np.abs(obj.chromatic_psf), pixel_size)

    if postprocess:
        psf = spikes.interpolate_image(
            spikes.poisson_resample_image(psf, flux, seed),
            dense_ps_size,
        )
    else:
        psf *= flux

    return psf


def patch_image_holes(image, hole_val):
    """
    Patches over holes in an image via linear interpolation. Holes (for now) must be strictly equal to hole_val.

    Parameters
    ----------
    image : 2D array of int or float
        The image in which to patch holes.
    hole_val : int or float
        The value which will be interpreted as a hole. Values in the image must match this value exactly.

    Returns
    -------
    image_noholes : 2D array of float
        The image except all the holes are filled in by interpolating from the edges of each hole.

    """
    row = np.arange(image.shape[0])
    col = np.arange(image.shape[1])
    reg_rows, reg_cols = np.meshgrid(row, col, indexing="ij")

    good_rows, good_cols = np.where(image != hole_val)
    points = np.vstack((good_rows, good_cols)).T
    interp = sp_itp.LinearNDInterpolator(points, image[good_rows, good_cols])

    image_noholes = interp(reg_rows, reg_cols)
    return image_noholes


def poisson_chisq(data, means):
    """
    Calculates the chi-square value between expected and observed data when the observed data follows a
    poisson distribution.

    Parameters
    ----------
    data : Array of int or float
        The observed data.
    means : Array of int or float
        The expected data. The two arrays must be at least broadcastable.

    Returns
    -------
    chisq : float
        The chi square value calculated using a poisson likelihood function:
        chisq = sum_i(f(d_i, t_i)) and f(d_i, t_i) = 2 * (t_i - d_i - (d_i * ln(t_i)) + (d_i*ln(d_i))
    """
    f_arr = 2 * (means - data - data * np.log(means) + data * np.log(data))
    return np.sum(f_arr)


def convert_ab_to_jansky(mag):
    """
    Converts an AB magnitude to Janskys.

    Parameters
    ----------
    mag : float
        The AB magnitude.

    Returns
    -------
    mag_jy : float
        The magnitude in Janskys.
    """
    return 10 ** ((mag - 8.9) / -2.5)


def disc_coords(radius, center=(0, 0)):  # returns xy coords
    """
    Finds the image coordinates that represent a solid disc of a specified radius centered somewhere.

    Parameters
    ----------
    radius : int or float
        The radius of the disc.
    center : tuple or aray of int
        The center of the disc. Since this function calculates image coordinates these values must be
        integers.

    Returns
    -------
    circ_x : 1D array of int
        The x-coordinates of all points inside the disc.
    circ_y : 1D array of int
        The y-coordinates of all points inside the disc. Note the use of xy indexing for the returns of
        this function.
    """
    rad_int = np.int_(np.ceil(radius))
    side = np.arange(-rad_int, rad_int + 1)
    stamp_x, stamp_y = np.meshgrid(side, side, indexing="xy")
    radii = stamp_x**2 + stamp_y**2
    circ_x, circ_y = np.where(radii < radius**2)

    circ_x += center[0] - radius
    circ_y += center[1] - radius

    return circ_x, circ_y


def find_first_pair(arr_a, arr_b):
    """
    Given two arrays, find the value of the first pair of elements.

    Parameters
    ----------
    arr_a : 1D array
        The first array.
    arr_b : 1D array
        The second array.

    Returns
    -------
    first_pair_val : type(arr_a[0])
        The element of the first pair within the arrays.
    """
    sz = np.min((arr_a.size, arr_b.size))
    idx = np.arange(sz)

    candidate_idx_dict = {arr_b[i]: idx[i] for i in idx}
    pair_sum_list = np.array([candidate_idx_dict.get(i, np.iinfo(np.int_).max) for i in arr_a]) + idx

    first_pair_val = arr_a[np.argmin(pair_sum_list)]
    return first_pair_val


def find_object_in_catalog(
    obj_coord_1, obj_coord_2, cat_coords_1, cat_coords_2, return_index=True, target_catalog=None
):
    """
    Given a catalog (two arrays of coordinates) and the coordinates of an object, find the index or value of the closest match of the object
    in a specified catalog.

    Parameters
    ----------
    obj_coord_1 : int or float
        The first coordinate of the desired object.
    obj_coord_2 : int or float
        The second coordinate of the desired object.
    cat_coords_1 : 1D array of int or float
        The first coordinate of all objects in the catalog.
    cat_coords_2 : 1D array of int or float
        The second coordinate of all objects in the catalog.
    return_index : boolean, optional
        Whether to return the index of the object in the catalog, or a value at that index in target_catalog.
    target_catalog : 1D array, optional
        If return_index is true, the element in this array at the desired object's index will be returned.

    Returns
    -------
    object_index (if return_index is false) : int
        The index of the object in the catalog.
    target_catalog_object (if return_index is true): type(target_catalog[0])
        The element in target_catalog at the index of the object in the given catalog.
    """
    candidates_coord_1 = np.argsort(np.abs(cat_coords_1 - obj_coord_1))
    candidates_coord_2 = np.argsort(np.abs(cat_coords_2 - obj_coord_2))

    object_index = find_first_pair(candidates_coord_1, candidates_coord_2)

    if return_index:
        return object_index
    else:
        assert target_catalog is not None
        return target_catalog[object_index]
