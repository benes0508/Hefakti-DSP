cmake_minimum_required(VERSION 3.15)
project(audacity_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Path to Audacity source code
set(AUDACITY_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../src-audacity")

# Common include directories
include_directories(
    ${AUDACITY_SRC}
    ${AUDACITY_SRC}/mir
    ${AUDACITY_SRC}/soundtouch
    ${AUDACITY_SRC}/fft
    ${AUDACITY_SRC}/effects
    ${AUDACITY_SRC}/scorealign
    ${AUDACITY_SRC}/vamp
    ${AUDACITY_SRC}/vamp/vamp-sdk
)

# =============================================================================
# MIR Bindings (BPM, Onset, Meter detection)
# =============================================================================
pybind11_add_module(mir_bindings
    mir_bindings.cpp
    ${AUDACITY_SRC}/mir/MusicInformationRetrieval.cpp
    ${AUDACITY_SRC}/mir/GetMeterUsingTatumQuantizationFit.cpp
    ${AUDACITY_SRC}/mir/MirDsp.cpp
    ${AUDACITY_SRC}/mir/MirUtils.cpp
    ${AUDACITY_SRC}/mir/StftFrameProvider.cpp
    ${AUDACITY_SRC}/mir/DecimatingMirAudioReader.cpp
)

# =============================================================================
# SoundTouch Bindings (Classic BPM detection)
# =============================================================================
pybind11_add_module(soundtouch_bindings
    soundtouch_bindings.cpp
    ${AUDACITY_SRC}/soundtouch/BPMDetect.cpp
    ${AUDACITY_SRC}/soundtouch/PeakFinder.cpp
    ${AUDACITY_SRC}/soundtouch/FIFOSampleBuffer.cpp
)

# =============================================================================
# FFT Bindings (Spectrum analysis)
# =============================================================================
pybind11_add_module(fft_bindings
    fft_bindings.cpp
    ${AUDACITY_SRC}/fft/SpectrumAnalyst.cpp
    ${AUDACITY_SRC}/fft/FFT.cpp
    ${AUDACITY_SRC}/fft/RealFFTf.cpp
    ${AUDACITY_SRC}/fft/PowerSpectrumGetter.cpp
    ${AUDACITY_SRC}/fft/Spectrum.cpp
    ${AUDACITY_SRC}/fft/SpectrumTransformer.cpp
)

# =============================================================================
# Effects Bindings (Levels, Clipping, Silence)
# =============================================================================
pybind11_add_module(effects_bindings
    effects_bindings.cpp
    ${AUDACITY_SRC}/effects/WaveChannelUtilities.cpp
    ${AUDACITY_SRC}/effects/FindClippingBase.cpp
    ${AUDACITY_SRC}/effects/TruncSilenceBase.cpp
    ${AUDACITY_SRC}/effects/ContrastBase.cpp
)

# =============================================================================
# ScoreAlign Bindings (Chroma features)
# =============================================================================
pybind11_add_module(scorealign_bindings
    scorealign_bindings.cpp
    ${AUDACITY_SRC}/scorealign/gen_chroma.cpp
    ${AUDACITY_SRC}/scorealign/comp_chroma.cpp
    ${AUDACITY_SRC}/scorealign/scorealign.cpp
    ${AUDACITY_SRC}/scorealign/audioreader.cpp
    ${AUDACITY_SRC}/scorealign/curvefit.cpp
    ${AUDACITY_SRC}/scorealign/hillclimb.cpp
    ${AUDACITY_SRC}/scorealign/fft3/FFT3.cpp
)

# =============================================================================
# Vamp Bindings (Transient analysis)
# =============================================================================
pybind11_add_module(vamp_bindings
    vamp_bindings.cpp
    ${AUDACITY_SRC}/vamp/AmplitudeFollower.cpp
    ${AUDACITY_SRC}/vamp/ZeroCrossing.cpp
    ${AUDACITY_SRC}/vamp/PercussionOnsetDetector.cpp
    ${AUDACITY_SRC}/vamp/SpectralCentroid.cpp
    ${AUDACITY_SRC}/vamp/PowerSpectrum.cpp
    ${AUDACITY_SRC}/vamp/FixedTempoEstimator.cpp
)

# Install targets
install(TARGETS
    mir_bindings
    soundtouch_bindings
    fft_bindings
    effects_bindings
    scorealign_bindings
    vamp_bindings
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)
