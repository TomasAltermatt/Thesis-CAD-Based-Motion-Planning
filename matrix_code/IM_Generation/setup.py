from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# A completely flat extension definition since we are running it from inside the folder
extensions = [
    Extension("narrow_phase_c", ["narrow_phase_c.pyx"])
]

setup(
    name="Narrow Phase C-Extension",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()] 
)