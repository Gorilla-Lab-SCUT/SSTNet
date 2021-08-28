# Copyright (c) Gorilla-Lab. All rights reserved.
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
# from torch.utils import cpp_extension

setup(
    name="htree",
    ext_modules=[
        Pybind11Extension("htree", [
            "src/tree.cpp",
            "src/api.cpp",
        ])
    ],
    cmdclass={"build_ext": build_ext}
)