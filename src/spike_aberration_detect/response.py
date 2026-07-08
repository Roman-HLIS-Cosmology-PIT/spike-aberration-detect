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
    psf = spikes.interpolate_image(
        spikes.poisson_resample_image(
            spikes.downsample_2d_image(np.abs(obj.chromatic_psf), pixel_size), flux, seed
        ),
        dense_ps_size,
    )

    return psf


def patch_image_holes(image, hole_val):
    """
    Patches over holes in an image via linear interpolation. Holes (for now) must be strictly equal to hole_val.
    """
    row = np.arange(image.shape[0])
    col = np.arange(image.shape[1])
    reg_rows, reg_cols = np.meshgrid(row, col, indexing="ij")

    good_rows, good_cols = np.where(image != hole_val)
    points = np.vstack((good_rows, good_cols)).T
    interp = sp_itp.LinearNDInterpolator(points, image[good_rows, good_cols])

    image_noholes = interp(reg_rows, reg_cols)
    return image_noholes
