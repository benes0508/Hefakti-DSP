"""
Audacity C++ Bindings
=====================

pybind11 bindings for Audacity's C++ audio analysis algorithms.

These bindings wrap the C++ code in src-audacity/ to provide
Python access to Audacity's DSP implementations.

Modules:
    soundtouch_bindings: Classic BPM detection (autocorrelation)
    mir_bindings: BPM/Meter detection (Tatum Quantization Fit)
    vamp_bindings: Amplitude follower, ZCR, Spectral centroid, Percussion onset
    scorealign_bindings: Chroma features
    effects_bindings: RMS, Peak, Clipping, Silence detection
"""

# Note: The actual binding modules are compiled C++ extensions.
# They are imported dynamically when available.

import importlib
from typing import Optional


def _try_import(module_name: str) -> Optional[object]:
    """Try to import a binding module, return None if not available."""
    try:
        return importlib.import_module(f".{module_name}", __name__)
    except ImportError:
        return None


def get_available_bindings() -> dict[str, bool]:
    """Get dict of binding module names and their availability status."""
    modules = [
        "soundtouch_bindings",
        "mir_bindings",
        "vamp_bindings",
        "scorealign_bindings",
        "effects_bindings",
    ]
    return {name: _try_import(name) is not None for name in modules}


def is_binding_available(name: str) -> bool:
    """Check if a specific binding module is available."""
    return _try_import(name) is not None


# Try to import bindings (will fail if not compiled yet)
soundtouch_bindings = _try_import("soundtouch_bindings")
mir_bindings = _try_import("mir_bindings")
vamp_bindings = _try_import("vamp_bindings")
scorealign_bindings = _try_import("scorealign_bindings")
effects_bindings = _try_import("effects_bindings")

__all__ = [
    "get_available_bindings",
    "is_binding_available",
    "soundtouch_bindings",
    "mir_bindings",
    "vamp_bindings",
    "scorealign_bindings",
    "effects_bindings",
]
