#include <Python.h>
#include <structmember.h>

short error[32768];

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

static PyMethodDef module_methods[] = {
    {"dither", (PyCFunction)meth_dither, METH_VARARGS | METH_KEYWORDS},
    {0},
};

static PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "dither", NULL, -1, module_methods};

extern PyObject * PyInit_dither() {
    PyObject * module = PyModule_Create(&module_def);
    return module;
}
