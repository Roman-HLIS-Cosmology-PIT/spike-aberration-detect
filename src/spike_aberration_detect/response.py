import numpy as np
from psfsim.polychrom import PolychromaticPSF

import spike_aberration_detect.spike_finder as spikes


def build_response_matrix(
    aberrations,
    ideal_spike_angles,
    psf_object: PolychromaticPSF,
    dense_ps_size,
    step: np.float64,
    bound,
    center,
    **kwargs,
):
    """
    Build a response matrix from a set of aberrations.
    """
    resp_matrix = np.zeros((12, 5))
    for i in np.arange(aberrations.size):
        extra_aberrations = [None, None, None, None, None]
        extra_aberrations[i] = aberrations[i]

        psf_object.compute_poly_psf(
            postage_stamp_size=32,
            optical_psf_only=True,
            use_postage_stamp_size=96,
            ovsamp=8,
            extra_aberrations=extra_aberrations,
        )

        psf_ovsamp = np.log10(np.abs(psf_object.chromatic_psf))
        psf_dsamp = spikes.downsample_2d_image(psf_ovsamp, pixel_size=8)
        psf = spikes.interpolate_image(psf_dsamp, dense_ps_size)

        spike_list = spikes.find_spikes(psf, step, bound, center, **kwargs)
        resp_matrix[:, i] = (spike_list - ideal_spike_angles) / aberrations[i]

    return resp_matrix


def generate_model_psf(
    scanum,
    scax,
    scay,
    flux,
    ps_size,
    dense_ps_size,
    wl_band,
    wl_band_name: str,
    seed,
    ovsamp=8,
    extra_aberrations=None,
):
    """
    Generate a PSF at a point on the detector with a flux and aberrations.
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
        np.arcsinh(
            spikes.poisson_resample_image(
                spikes.downsample_2d_image(np.abs(obj.chromatic_psf), pixel_size), flux, seed
            )
        ),
        dense_ps_size,
    )

    return psf
