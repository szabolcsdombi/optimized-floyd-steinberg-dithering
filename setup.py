from setuptools import Extension, setup

ext = Extension(
    name='dither',
    sources=['./dither.c'],
    define_macros=[('PY_SSIZE_T_CLEAN', None)],
)

setup(
    name='mydither',
    version='0.1.0',
    ext_modules=[ext],
)
