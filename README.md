# optimized-floyd-steinberg-dithering

I recently stumbled upon a [really nice article](https://pythonspeed.com/articles/optimizing-dithering/) implementing an optimized version of the [Floyd–Steinberg dithering](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering).
In this article, the author employed numpy, numba jit, and clever mathematical tricks to achieve maximal performance.

However, I believe there are two potential areas of improvement:

1. Multiplication is a computationally intensive operation. It could be swapped with bitwise operations for a more efficient approach.
2. Instead of using numba jit, it might be more performant to just write C, as it could surpass the speed of the jit compiled code.

Unfortunately, I wasn't able to locate the original image that the author worked with,
so I opted for a random sample image from the skimage library.
I picked `skimage.data.coffee()` because I like coffee a lot.

Below is the original `dither6()` function from the article for reference:

```py
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
```

Subsequently, I've ported this function to a conventional Python C extension and replaced the multiplication with bitwise operations:

```c
static PyObject * meth_dither(PyObject * self, PyObject * args, PyObject * kwargs) {
    const char * keywords[] = {"size", "data", NULL};

    int width;
    int height;
    Py_buffer view;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)y*", (char **)keywords, &width, &height, &view)) {
        return NULL;
    }

    unsigned char * pixels = (unsigned char *)view.buf;
    short * current_error = error + 16;
    short * next_error = error + (width & ~0xf) + 32;

    next_error[0] = 0;
    current_error[0] = 0;
    next_error[width + 1] = 0;
    current_error[width + 1] = 0;
    for (int x = width; x; --x) {
        next_error[x] = 0;
        current_error[x] = pixels[x - 1];
    }

    for (int y = 0; y < height; ++y) {
        short right_pixel_error = 0;
        short downleft_prev_error = 0;
        short downleft_prevprev_error = 0;
        const int yw = y * width;
        for (int x = 0; x < width; ++x) {
            const short old_value = current_error[x + 1] + right_pixel_error;
            const unsigned char new_value = old_value < 128 ? 0 : 255;
            const short error = old_value - new_value;
            pixels[yw + x] = new_value;
            right_pixel_error = ((error << 3) - error) >> 4;
            next_error[x] = pixels[yw + width + x - 1] + downleft_prev_error + (((error << 1) + error) >> 4);
            downleft_prev_error = downleft_prevprev_error + (((error << 2) + error) >> 4);
            downleft_prevprev_error = error >> 4;
        }
        next_error[width] = pixels[yw + (width << 1) - 1] + downleft_prev_error;
        short * tmp = current_error;
        current_error = next_error;
        next_error = tmp;
    }

    PyBuffer_Release(&view);
    Py_RETURN_NONE;
}
```

I wanted to implement an in-place operation because allocations frequently come at a cost.
To match the original implementation, I wrapped my function:

```py
def c_dither(img):
    h, w = img.shape
    result = img.copy()
    c_dither_inplace((w, h), result)
    return result
```

To run the benchmarks first build the dither extension then invoke pytest-benchmark as follows:

```bash
python setup.py build_ext --inplace
```

```bash
python -W ignore -m pytest test.py \
    --benchmark-disable-gc \
    --benchmark-columns=mean,stddev,median,ops \
    --benchmark-sort=mean
```
