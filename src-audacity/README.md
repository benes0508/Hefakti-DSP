# Audacity Audio Analysis Functions

This folder contains extracted DSP and analysis code from Audacity source.
These are C++ implementations that can be used as reference or ported to Python.

## Directory Structure

```
src-audacity/
├── mir/           # Music Information Retrieval (BPM, Onset, Meter)
├── soundtouch/    # SoundTouch BPM Detection
├── fft/           # FFT and Spectrum Analysis
├── effects/       # Level Analysis, Clipping, Silence Detection
├── scorealign/    # Chroma Features and Score Alignment
└── vamp/          # Transient/Envelope Analysis (Vamp plugins)
```

## Functionality Overview

### 1. BPM Detection (`mir/` + `soundtouch/`)

**Tatum Quantization Fit Method** (`mir/GetMeterUsingTatumQuantizationFit.cpp`)
- State-of-the-art tempo detection for loop classification
- Returns BPM with confidence score
- Based on quantizing ODF peaks to tatums (smallest rhythmic units)

**SoundTouch BPMDetect** (`soundtouch/BPMDetect.cpp`)
- Classic autocorrelation-based tempo detection
- Decimates to ~500 Hz, envelope detection, autocorrelation
- BPM range: 29-200 BPM

Key files:
- `mir/MusicInformationRetrieval.h` - High-level API
- `mir/MirDsp.cpp` - DSP primitives (ODF, normalization)
- `soundtouch/BPMDetect.cpp` - SoundTouch algorithm
- `soundtouch/PeakFinder.cpp` - Peak detection for BPM

### 2. Onset Detection (`mir/`)

**Spectral Flux ODF** (`mir/MirDsp.cpp`)
- Onset Detection Function using spectral flux
- STFT-based onset strength calculation

Key files:
- `mir/StftFrameProvider.cpp` - STFT frame generation
- `mir/MirDsp.cpp` - `GetOnsetDetectionFunction()`

### 3. Meter/Time Signature (`mir/`)

**Tatum-based Meter Detection** (`mir/GetMeterUsingTatumQuantizationFit.cpp`)
- Detects time signature (4/4, 3/4, etc.)
- Uses autocorrelation comb filtering for bar division

Key types in `mir/MirTypes.h`:
- `MusicalMeter` - BPM, time signature, confidence

### 4. Spectrum Analysis (`fft/`)

**SpectrumAnalyst** (`fft/SpectrumAnalyst.cpp`)
5 analysis modes:
- Spectrum (magnitude)
- Autocorrelation
- Cepstrum
- Standard Autocorrelation
- Cue Beat

**Power Spectrum** (`fft/PowerSpectrumGetter.cpp`)
- Power spectrum calculation from audio

Key files:
- `fft/FFT.cpp` - FFT implementation
- `fft/RealFFTf.cpp` - Real-valued FFT
- `fft/SpectrumTransformer.cpp` - STFT processing

### 5. Level Analysis (`effects/`)

**RMS / Peak / Min-Max** (`effects/WaveChannelUtilities.cpp`)
- `GetRMS()` - Root mean square level
- `GetMinMax()` - Peak levels (min/max)

**Contrast Analysis** (`effects/ContrastBase.cpp`)
- Foreground/background contrast measurement
- Useful for speech/music discrimination

Key files:
- `effects/WaveChannelUtilities.h` - RMS, GetMinMax APIs
- `effects/ContrastBase.cpp` - Contrast ratio calculation

### 6. Quality Detection (`effects/`)

**Clipping Detection** (`effects/FindClippingBase.cpp`)
- Detects clipped samples in audio
- Configurable start/stop thresholds

**Silence Detection** (`effects/TruncSilenceBase.cpp`)
- Detects silence regions
- Configurable threshold and minimum duration

### 7. Chroma Features (`scorealign/`)

**Chroma Generation** (`scorealign/gen_chroma.cpp`)
- 12-bin chroma feature extraction
- Uses FFT3 for spectral analysis
- Pitch class energy distribution

**Chroma Comparison** (`scorealign/comp_chroma.cpp`)
- Compare chroma vectors between audio files
- Dynamic time warping for alignment

Key files:
- `scorealign/gen_chroma.cpp` - Main chroma generation
- `scorealign/fft3/FFT3.cpp` - FFT for chroma
- `scorealign/scorealign.cpp` - Score alignment algorithm

### 8. Transient/Envelope Analysis (`vamp/`)

**Amplitude Follower** (`vamp/AmplitudeFollower.cpp`)
- Envelope extraction
- Attack/release time constants

**Zero Crossing Rate** (`vamp/ZeroCrossing.cpp`)
- Sign changes per time window
- Useful for speech/music discrimination

**Percussion Onset Detector** (`vamp/PercussionOnsetDetector.cpp`)
- Specialized onset detection for drums
- High-frequency content ratio method

**Fixed Tempo Estimator** (`vamp/FixedTempoEstimator.cpp`)
- Alternative tempo estimation
- Autocorrelation with peak picking

**Spectral Centroid** (`vamp/SpectralCentroid.cpp`)
- "Brightness" measure
- Frequency-weighted mean

## Licenses

- **SoundTouch**: LGPL 2.1+ (Olli Parviainen)
- **Audacity MIR**: GPL 2.0+ (Audacity Team)
- **libscorealign**: MIT-style
- **libvamp examples**: BSD-style

## Usage Notes

These are C++ source files intended as reference implementations.
For Python usage, consider:
1. Using the algorithms as reference to implement in Python/NumPy
2. Creating pybind11 bindings for the C++ code
3. Using librosa equivalents where available

## Related Python Implementations

See `src/dsp/` for Python ports:
- `tempo_audacity.py` - Python port of tempo detection
- `tempo_beatthis.py` - Beat This! integration (ISMIR 2024)
