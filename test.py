import numpy as np
from numba import njit
from skimage import data
from skimage.color import rgb2gray

from dither import dither as c_dither_inplace

image = (rgb2gray(data.coffee()) * 255.0).astype('u1')


@njit
def py_dither(img):
    result = np.empty(img.shape, dtype=np.uint8)
    y_size = img.shape[0]
    x_size = img.shape[1]
    staging_current = np.zeros(x_size + 2, np.int16)
    staging_current[1:-1] = img[0]
    staging_next = np.zeros(x_size + 2, np.int16)

    for y in range(y_size):
        right_pixel_error = 0
        downleft_prev_error = 0
        downleft_prevprev_error = 0
        for x in range(x_size):
            old_value = staging_current[x + 1] + right_pixel_error
            new_value = 0 if old_value < 128 else 255
            result[y, x] = new_value
            error = old_value - new_value
            right_pixel_error = error * 7 // 16
            staging_next[x] = (
                img[y + 1, x - 1] + downleft_prev_error + error * 3 // 16
            )
            downleft_prev_error = downleft_prevprev_error + error * 5 // 16
            downleft_prevprev_error = error // 16

        staging_next[x_size] = img[y + 1, x_size - 1] + downleft_prev_error

        staging_current, staging_next = staging_next, staging_current

    return result


def c_dither(img):
    h, w = img.shape
    result = img.copy()
    c_dither_inplace((w, h), result)
    return result


def test_py_dither(benchmark):
    benchmark(py_dither, image)


def test_c_dither(benchmark):
    benchmark(c_dither, image)
