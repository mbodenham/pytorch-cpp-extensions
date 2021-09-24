from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='linear_cpp',
      ext_modules=[cpp_extension.CppExtension('linear_cpp', ['linear.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
