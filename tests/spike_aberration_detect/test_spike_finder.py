import numpy as np
from spike_aberration_detect.spike_finder import line_coords


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
