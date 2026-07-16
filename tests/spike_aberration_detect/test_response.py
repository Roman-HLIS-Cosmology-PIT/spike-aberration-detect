# from spike_aberration_detect.response import build_response_matrix, generate_model_psf
import numpy as np
from spike_aberration_detect.response import (
    disc_coords,
    find_first_pair,
    find_object_in_catalog,
    patch_image_holes,
    poisson_chisq,
)

# import spike_aberration_detect.spike_finder as spikes


def test_build_response_matrix():
    """
    Testing the naive response matrix builder.
    """
    return


def test_generate_model_psf():
    """
    Testing model PSF generation.
    """
    return


def test_patch_image_holes():
    """
    Testing the image hole patcher.
    """
    image = np.array(
        (
            (
                5,
                5,
                5,
                5,
                5,
            ),
            (
                5,
                0,
                0,
                0,
                5,
            ),
            (
                5,
                0,
                0,
                0,
                5,
            ),
            (
                5,
                0,
                0,
                0,
                5,
            ),
            (
                5,
                5,
                5,
                5,
                5,
            ),
        )
    )
    patch = patch_image_holes(image, 0)
    assert np.allclose(patch, 5.0)

    return


def test_poisson_chisq():
    """
    Testing the poisson distribution chi square calculator.
    """
    rng = np.random.default_rng(1000)
    means = np.full(500, 500)
    data = rng.poisson(means, means.size)
    assert poisson_chisq(data, means) > 0.8
    return


def test_convert_ab_to_jansky():
    """
    Testing the conversion from AB mag to Jy.
    """
    return


def test_disc_coords():
    """
    Testing disc coordinate generation.
    """
    rad = 5.1
    center = (3, 0)

    disc_x, disc_y = disc_coords(rad, center)
    disc_pts = np.vstack((disc_x, disc_y)).T
    assert disc_pts.shape == (89, 2)
    assert np.where(disc_pts == 8)[0].size > 0
    return


def test_find_first_pair():
    """
    Testing the function to find the first pair of elements in two arrays.
    """

    a = np.array((5, 6, 8, 1, 2, 3))
    b = np.array((43, 727, 6, 8, 9, 10))

    assert find_first_pair(a, b) == 6
    assert find_first_pair(b, a) == 6
    return


def test_find_object_in_catalog():
    """
    Testing finding an object in a catalog.
    """
    cat_coord_1 = np.arange(20)
    cat_coord_2 = np.roll(np.arange(20), 10)

    cat_3 = np.arange(20) * 20
    coord_1 = 5.1
    coord_2 = 14.9

    assert find_object_in_catalog(coord_1, coord_2, cat_coord_1, cat_coord_2) == 5
    assert find_object_in_catalog(coord_1, coord_2, cat_coord_1, cat_coord_2, False, cat_3) == 100

    return
