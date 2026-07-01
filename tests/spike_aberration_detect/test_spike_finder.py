import numpy as np
from psfsim.polychrom import PolychromaticPSF
from spike_aberration_detect.spike_finder import (
    downsample_2d_image,
    find_spikes,
    interpolate_image,
    line_coords,
    poisson_resample_image,
)


def test_line_coords():
    """
    Testing line coordinates at some angles.
    """
    angle = 0
    bound = 48
    zero_degrees_answer = np.zeros((2, bound))
    zero_degrees_answer[0] = np.arange(bound)
    zero_degrees = line_coords(angle, bound)
    assert np.allclose(zero_degrees, zero_degrees_answer)

    angles = np.array((90.0, 180.0, 270.0))
    center = np.array((30, 30))

    answer = np.zeros((3, 2, bound))
    answer[0, 1] = np.arange(0, -bound, -1)
    answer[1, 0] = np.arange(0, -bound, -1)
    answer[2, 1] = np.arange(0, bound)
    answer += center.reshape((2, 1))[np.newaxis, :, :]
    assert np.allclose(answer.astype(np.int_), line_coords(angles, bound, center=center))

    return


def test_find_spikes():
    """
    Testing the spike finder.
    """
    j129 = np.linspace(1.131, 1.454, 5)  # microns

    ps_size = 96
    ovsamp = pixel_size = 8
    mag = 1e10  # electrons from CCDs, use a large number (above 1e9) to get rid of any noticeable poisson effect
    step = 0.1  # degrees
    borders = (0.2, 1.0)
    seed = 727

    bound = ps_size // 2
    center = np.array((ps_size // 2, ps_size // 2))
    flux_factor = pixel_size**2 * mag

    j129_obj = PolychromaticPSF(9, 0, 0, j129)

    j129_obj.compute_poly_psf(
        optical_psf_only=True, use_postage_stamp_size=ps_size, ovsamp=ovsamp, use_filter="J129"
    )
    psf_ideal = np.arcsinh(
        poisson_resample_image(
            downsample_2d_image(np.abs(j129_obj.chromatic_psf), pixel_size), flux_factor, seed
        )
    )

    ideal_sums, _, _, ideal_spikes = find_spikes(
        psf_ideal, step, bound, center, borders=borders, verbose=True
    )

    assert ideal_spikes.size == 12
    assert np.all(np.diff(ideal_spikes) > 5.0)

    return


def test_poisson_resample():
    """
    Testing poisson resampling of an image.
    """
    poisson_means = np.array(((3, 10), (4e8, 300)))
    resamp = poisson_resample_image(poisson_means, 2, 727)
    resamp_answer = np.array(((4, 14), (799985336, 573)))

    assert np.allclose(resamp, resamp_answer)
    return


def test_downsample_2d_image():
    """
    Testing the image downsampler.
    """
    left, right = np.meshgrid(np.arange(8), np.arange(8))
    image = left + right

    dsamp_img = downsample_2d_image(image, 2)
    dsamp_l, dsamp_r = np.meshgrid(np.arange(1, 8, 2), np.arange(1, 8, 2))
    dsamp_img_answer = dsamp_l + dsamp_r - 1

    assert np.allclose(dsamp_img, dsamp_img_answer)
    return


def test_interpolate_image():
    """
    Testing linear image interpolation.
    """
    left, right = np.meshgrid(np.arange(3), np.arange(3))
    image = left + right
    dense_image = interpolate_image(image, 9)

    left_d, right_d = np.meshgrid(np.linspace(0, 2, 9), np.linspace(0, 2, 9))
    dense_image_answer = left_d + right_d

    assert np.allclose(dense_image, dense_image_answer)
    return


test_downsample_2d_image()
