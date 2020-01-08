from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext = Extension("ep_fast",
	sources=["ep_fast.pyx"],
    include_dirs = [numpy.get_include()],
	extra_compile_args=['-O3'],
	language="c"
	)
                
setup(
	  name = "ep_fast",
	  #ext_modules=cythonize('*.pyx'),
	  #sources=["/homes/omerw/experiments/pred_code/boost_invcdf.cpp"]	  
	  ext_modules=[ext],
      cmdclass = {'build_ext': build_ext})

