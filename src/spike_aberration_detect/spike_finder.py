import numpy as np
import scipy.interpolate as sp_itp


# def line_coords(angle, bound, center=np.array((0, 0))):
def line_coords(angle, bound, center=(0, 0)):
    """
    Gets a pixelated line at a specified origin, angle, and length.

    Parameters
    ----------
    angle : float or float array
        The angle(s) of the line(s), clockwise from the positive x-axis.
    bound : int
        The length of the line.
    center: array of int
        The point at which the line begins.

    Returns
    -------
    lines : ndarray of int
        If one angle was provided this is a 2xbound array of ints that represent the
        line in data coordinates. If N angles were provided this array has shape (N, 2, bound).
    """
    # since (0, 0) in data coords is top left instead of bottom left we must use clockwise rotations to get
    # a visually ccl rotation. However we will take a transpose later anyways so the rotation matrix definition is ccl, as expected. How quaint
    # In data coordinates, however, the rotation is clockwise. The unit test keeps this in mind.
    angle = np.deg2rad(angle)
    line = np.zeros((2, bound))
    line[0] = np.arange(bound)

    rotation = np.array(
        (
            (np.cos(angle), -np.sin(angle)),
            (np.sin(angle), np.cos(angle)),
        )
    )

    if len(rotation.shape) > 2:
        line = line.reshape((1, 2, bound))

    return np.int32(rotation.T @ line) + np.int32(np.reshape(center, (2, 1)))


def find_spikes(image, step, bound=1024, center=(0, 0), borders=(0.0, 1.0), threshold=0.5, verbose=False):
    """
    Given an image of a PSF, find the diffraction spikes within.

    Parameters
    ----------
    image : 2D array of int or float
        The image that supposedly contains diffraction spikes.
    step : float
        The increment angle between successive lines, in degrees.
    bound : int, optional
        The length of lines to be drawn.
    center : tuple or array of int, optional
        The point at which each line begins.
    borders : tuple or array of float, optional
        The part of the line that is to be used for fitting. borders[0] must be less than
        borders[1], and both values must be <=1. A value of 1 corresponds to the end of
        the line and 0 to the beginning of the line. For example, a borders value of
        (0.2, 0.9) only sums pixels on the line from 0.2 of its length to 0.9 of its length.
    threshold : float, optional
        Where the cutoff is that determines which sums are part of spikes and which are not.
        This value must be between 0 and 1. A value of 0 corresponds to min(S(theta)), and a
        value of 1 corresponds to max(S(theta)). The default value is 0.5, corresponding to
        (min(S(theta)) + max(S(theta)))/2.
    verbose : boolean, optional
        Whether to return more information than an array of spike angles. If set to true,
        this function returns S(theta), the cutoff value, all angles part of detected spikes,
        and the spike angles.

    Returns
    -------
    sums : array of float, optional
        Only returned if verbose is True. This array is the sum over each line as a function of line
        angle.
    spike_angles : tuple of array of float, optional
        All angles that are part of detected spikes. Each element of the tuple is corresponds to one of
        the spikes.
    cutoff : float, optional
        Sums below this value are not part of spikes, and values above it are part of spikes.
    spike_list : array of float
        Array of found spike angles. Each group of spikes has a minimum and maximum angle, found by linearly
        interpolating S(theta) and intersecting it with the cutoff value. The midpoint between the minimum
        and maximum is the measured angle of the spike.
    """
    angles = np.linspace(0, 360, num=np.int32(360.0 / step), endpoint=False)

    lines = line_coords(angles, bound, center)
    image_analyze = (
        image.T
    )  # imshow will display the image as row,col with 0,0 in top left. For our analysis we want (col,row).

    line_vals = image_analyze[lines[:, 0], lines[:, 1]]
    sums = np.sum(line_vals, axis=-1)

    # 0 to 1 for these. Should I add a way to make it force 0 <= borders[0] < borders[1] <= 1?
    range_low = np.int_(borders[0] * bound)
    range_high = np.int_(borders[1] * bound)

    sums = np.sum(line_vals[:, range_low:range_high], axis=-1)
    """
    This formula is slightly different from the midpoint formula from before. The formula defaults to 50%, using 50% recovers the original behavior. Other
    ways to do this include rejecting the lowest 10% of values or so. That one might be worth pursuing in the future. The goal is to keep this method as simple as possible--
    I do NOT want to start staring at noise to find spikes.

    There is another method that may be worth pursuing that involves a boxcar average that goes around the circle and might be able to correct for groups of spikes that are much
    higher or lower than the naive cutoff assumes.
    """
    diff = np.max(sums) - np.min(sums)
    cutoff = np.min(sums) + (diff * threshold)
    spike_indices = np.where(sums > cutoff)[0]
    spike_angles = angles[spike_indices]
    # print( np.diff( spike_angles ) )

    # try 3 degrees to tell differing spikes apart for now
    spike_angle_discriminator = 3.0
    borders = np.where(np.diff(spike_angles) > spike_angle_discriminator)[0]
    spike_groups = np.split(spike_angles, borders + 1)
    spike_group_indices = np.split(spike_indices, borders + 1)

    # print(f"Cutoff of {cutoff:.3f}")
    # unfortunately this for loop is necessary unless theres a convenient package for jagged arrays
    spike_list = np.zeros(len(spike_groups))
    for i in np.arange(len(spike_groups)):
        idx_min_r = spike_group_indices[i][0]
        idx_max_l = spike_group_indices[i][-1]

        idx_min_l = idx_min_r - 1
        idx_max_r = idx_max_l + 1

        slope_left = (sums[idx_min_r] - sums[idx_min_l]) / step
        slope_right = (sums[idx_max_r] - sums[idx_max_l]) / step

        th_min = ((cutoff - sums[idx_min_l]) / slope_left) + angles[idx_min_l]
        th_max = ((cutoff - sums[idx_max_l]) / slope_right) + angles[idx_max_l]

        # print( f'{sums[idx_min_l]:.3f}, {sums[idx_min_r]:.3f} and {sums[idx_max_l]:.3f}, {sums[idx_max_r]:.3f}' )
        # print(
        #     f"{angles[idx_min_l]:.2f}, {angles[idx_min_r]:.2f} and {angles[idx_max_l]:.2f}, {angles[idx_max_r]:.2f}. interpolated as {th_min:.2f}, {th_max:.2f}"
        # )

        # spike_list[i] = np.median( spike_groups[i] )
        spike_list[i] = (th_min + th_max) / 2.0

    # print( spike_list )
    if verbose:
        return sums, spike_angles, cutoff, spike_list
    else:
        return spike_list


def draw_ray(ax, angle, bound, center=(0, 0), borders=(0.0, 1.0), **kwargs):
    """
    Draws a line segment on an image at a specified origin, and length.

    Parameters
    ----------
    ax : axis
        Matplotlib axes object on which the line is drawn.
    angle : float
        The angle of the line, clockwise from the positive x-axis.
    bound : int
        The length of the line.
    center : tuple or array of int
        The point at which the line begins in data coordinates.
    borders : tuple or array of float
        The part of the line that is to be drawn. borders[0] must be less than borders[1], and
        both values must be <=1. A value of 1 corresponds to the end of the line and 0 to the
        beginning of the line. For example, a borders value of (0.2, 0.9) draws the line from
        0.2 of its length to 0.9 of its length.
    **kwargs : tuple
        Extra kwargs to pass to the matplotlib plot function, such as line style or color.
    """
    line = line_coords(angle, bound, center)
    range_low = np.int_(borders[0] * bound)
    range_high = np.int_(borders[1] * bound)
    ax.plot(line[0, range_low:range_high], line[1, range_low:range_high], **kwargs)
    return


def downsample_2d_image(image, pixel_size=8):  # ONLY WORKS ON 2D IMAGES
    """
    Downsaples a 2D image by taking the mean of subpixels of a specified size.

    Parameters
    ----------
    image : 2D array of int or float
        The image to downsample.
    pixel_size : int
        The side length, in data coordinates, of the regions over which the
        mean is to be taken. This value must divide both values in image.shape.

    Returns
    -------
    downsampled_image : 2D array of float
        The downsampled image, of shape image.shape / pixel_size.
    """

    if np.any(np.array(image.shape) % pixel_size) != 0:
        raise ValueError(
            f"The side lengths of image must be divisble by pixel_size. Got remainders {np.array(image.shape) % pixel_size}."
        )

    subpixel_side = np.arange(pixel_size)
    subpixel_cols, subpixel_rows = np.meshgrid(subpixel_side, subpixel_side)

    dsamp_size = image.shape[0] // pixel_size
    pixels_side = np.arange(dsamp_size) * pixel_size
    pixels_cols, pixels_rows = np.meshgrid(pixels_side, pixels_side)

    pixels_rows = pixels_rows[:, :, np.newaxis]  # promote to 3D
    pixels_cols = pixels_cols[:, :, np.newaxis]

    idx_arr_subpixel_rows = np.full((dsamp_size, dsamp_size, pixel_size**2), subpixel_rows.flatten())
    idx_arr_subpixel_cols = np.full((dsamp_size, dsamp_size, pixel_size**2), subpixel_cols.flatten())

    idx_arr_rows = idx_arr_subpixel_rows + pixels_rows
    idx_arr_cols = idx_arr_subpixel_cols + pixels_cols

    pixelated_image = image[idx_arr_rows, idx_arr_cols]
    downsampled_image = np.mean(pixelated_image, axis=-1)
    return downsampled_image


def poisson_resample_image(img, flux_factor, seed):
    """
    Resamples an image using a per-pixel poisson distribution.

    Parameters
    ----------
    img : 2D array of int or float
        The image to resample.
    flux_factor : float
        A value the image is multiplied by. A typical PSF is normalized to sum to 1, so this
        value can serve as the total magnitude of the image.
    seed : int
        RNG seed for np.random.

    Returns
    -------
    resampled_image : 2D array of int
        The resampled image. Since a poisson distribution is drawn from the values are guaranteed
        to be integers.
    """
    # resamples an image where each pixel gets its own poisson distribution
    # this will change the scales around so that the resulting image has large positive values!!!! be aware
    rng = np.random.default_rng(seed)
    poisson_means = img * flux_factor

    resampled_img = rng.poisson(poisson_means, size=poisson_means.shape)
    return resampled_img


def interpolate_image(img, dense_size):  # 2D ONLY
    """
    Linearly interpolates a 2D image and returns another, denser image.

    Parameters
    ----------
    img : 2D array of int or float
        The image to interpolate.
    dense_size : int
        The new side length of the interpolated image.

    Returns
    -------
    interpolated_image : 2D array of float
        A linearly interpolated version of the input image. This image is still
        discrete so that it can be passed to all the other functions that use discrete
        numerical methods. This image is of shape (dense_size, dense_size).
    """
    x_pts = np.arange(img.shape[0])
    y_pts = np.arange(img.shape[1])

    new_grid = sp_itp.RegularGridInterpolator((x_pts, y_pts), img, method="linear")
    dense_x_1d = np.linspace(0, img.shape[0] - 1, dense_size)
    dense_y_1d = np.linspace(0, img.shape[1] - 1, dense_size)
    dense_x, dense_y = np.meshgrid(dense_x_1d, dense_y_1d, indexing="ij")

    return new_grid((dense_x, dense_y))
