"""
Setup script for hefakti-dsp package.
Audacity DSP algorithms with pybind11 bindings.
"""

import sys
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

try:
    import pybind11
except ImportError:
    print("pybind11 is required. Install with: pip install pybind11")
    sys.exit(1)


class BuildExt(build_ext):
    """Custom build_ext command with C++17 support."""

    def build_extensions(self):
        for ext in self.extensions:
            if sys.platform == "darwin":
                ext.extra_compile_args = ["-std=c++17", "-stdlib=libc++"]
                ext.extra_link_args = ["-stdlib=libc++"]
            elif sys.platform == "win32":
                ext.extra_compile_args = ["/std:c++17"]
            else:
                ext.extra_compile_args = ["-std=c++17"]
        build_ext.build_extensions(self)


include_dirs = [
    pybind11.get_include(),
    "src-audacity",
    "src-audacity/soundtouch",
    "src-audacity/mir",
    "src-audacity/fft",
    "src-audacity/effects",
    "src-audacity/scorealign",
    "src-audacity/vamp",
]

soundtouch_ext = Extension(
    "hefakti_dsp.bindings.soundtouch_bindings",
    sources=[
        "src/hefakti_dsp/bindings/soundtouch_bindings.cpp",
        "src-audacity/soundtouch/BPMDetect.cpp",
        "src-audacity/soundtouch/PeakFinder.cpp",
        "src-audacity/soundtouch/FIFOSampleBuffer.cpp",
    ],
    include_dirs=include_dirs,
    language="c++",
)

setup(
    name="hefakti-dsp",
    version="0.1.0",
    description="Audacity DSP algorithms with Python bindings",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[soundtouch_ext],
    cmdclass={"build_ext": BuildExt},
    python_requires=">=3.9",
    install_requires=["numpy>=1.20"],
)
