import numpy as np
import scipy.interpolate as sp_itp


def line_coords(angle, bound, center=np.array((0, 0))):
    # since (0, 0) in data coords is top left instead of bottom left we must use clockwise rotations to get
    # a visually ccl rotation. However we will take a transpose later anyways so the rotation matrix def is ccl, as expected. How quaint
    angle = np.deg2rad(angle)
    # print( angle )
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

    # print( line.shape )
    # print( rotation.T.shape )

    return np.int32(rotation.T @ line) + np.int32(np.reshape(center, (2, 1)))


def find_spikes(
    image, step, bound=1024, center=np.array((0, 0)), borders=(0.0, 1.0), threshold=0.5, verbose=False
):
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

    print(f"Cutoff of {cutoff:.3f}")
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
        print(
            f"{angles[idx_min_l]:.2f}, {angles[idx_min_r]:.2f} and {angles[idx_max_l]:.2f}, {angles[idx_max_r]:.2f}. interpolated as {th_min:.2f}, {th_max:.2f}"
        )

        # spike_list[i] = np.median( spike_groups[i] )
        spike_list[i] = (th_min + th_max) / 2.0

    # print( spike_list )
    if verbose:
        return sums, spike_angles, cutoff, spike_list
    else:
        return spike_list


def draw_ray(ax, angle, bound, center=np.array((0, 0)), borders=(0.0, 1.0), **kwargs):
    line = line_coords(angle, bound, center)
    range_low = np.int_(borders[0] * bound)
    range_high = np.int_(borders[1] * bound)
    ax.plot(line[0, range_low:range_high], line[1, range_low:range_high], **kwargs)
    return


def downsample_2d_image(image, pixel_size=8):  # ONLY WORKS ON 2D IMAGES
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


def poisson_resample_image(img, flux_factor):
    # resamples an image where each pixel gets its own poisson distribution
    # this will change the scales around so that the resulting image has large positive values!!!! be aware
    rng = np.random.default_rng()
    poisson_means = img * flux_factor

    resampled_img = rng.poisson(poisson_means, size=poisson_means.shape)
    return resampled_img


def interpolate_image(img, dense_size):  # 2D ONLY
    x_pts = np.arange(img.shape[0])
    y_pts = np.arange(img.shape[1])

    new_grid = sp_itp.RegularGridInterpolator((x_pts, y_pts), img, method="linear")
    dense_x_1d = np.linspace(0, img.shape[0] - 1, dense_size)
    dense_y_1d = np.linspace(0, img.shape[1] - 1, dense_size)
    dense_x, dense_y = np.meshgrid(dense_x_1d, dense_y_1d, indexing="ij")

    return new_grid((dense_x, dense_y))
