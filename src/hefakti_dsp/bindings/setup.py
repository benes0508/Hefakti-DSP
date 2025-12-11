"""
Build script for Audacity C++ bindings.

Usage:
    pip install pybind11
    python setup.py build_ext --inplace

Or with CMake:
    mkdir build && cd build
    cmake .. -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
    make
"""

import os
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# pybind11 is required
try:
    import pybind11
except ImportError:
    print("pybind11 is required. Install with: pip install pybind11")
    sys.exit(1)


class CMakeExtension(Extension):
    """CMake-based extension."""
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Build extension using CMake."""

    def build_extension(self, ext):
        import subprocess

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # CMake configuration
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        ]

        # Build configuration
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]

        # Make build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Run CMake
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=self.build_temp,
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=self.build_temp,
        )


# Get the directory containing this file
HERE = Path(__file__).parent.absolute()
AUDACITY_SRC = HERE.parent.parent / "src-audacity"


def get_pybind11_include():
    """Get pybind11 include path."""
    return pybind11.get_include()


# Common include directories
include_dirs = [
    get_pybind11_include(),
    str(AUDACITY_SRC),
    str(AUDACITY_SRC / "mir"),
    str(AUDACITY_SRC / "soundtouch"),
    str(AUDACITY_SRC / "fft"),
    str(AUDACITY_SRC / "effects"),
    str(AUDACITY_SRC / "scorealign"),
    str(AUDACITY_SRC / "vamp"),
    str(AUDACITY_SRC / "vamp" / "vamp-sdk"),
]

# Compiler flags
extra_compile_args = ["-std=c++17"]
if sys.platform == "darwin":
    extra_compile_args += ["-stdlib=libc++"]


# Define extensions (fallback if CMake is not available)
extensions = [
    Extension(
        "soundtouch_bindings",
        sources=[
            str(HERE / "soundtouch_bindings.cpp"),
            str(AUDACITY_SRC / "soundtouch" / "BPMDetect.cpp"),
            str(AUDACITY_SRC / "soundtouch" / "PeakFinder.cpp"),
            str(AUDACITY_SRC / "soundtouch" / "FIFOSampleBuffer.cpp"),
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]


setup(
    name="audacity_bindings",
    version="0.1.0",
    author="Hefakti R&D",
    description="pybind11 bindings for Audacity audio analysis",
    ext_modules=extensions,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=["pybind11>=2.10"],
)
