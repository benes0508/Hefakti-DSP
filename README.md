# Hefakti DSP

Audacity DSP algorithms with Python bindings (pybind11).

## Features

- **BPM Detection** - SoundTouch algorithm
- **FFT & Spectrum Analysis**
- **Music Information Retrieval** (MIR)
- **Onset Detection** (VAMP plugins)
- **Clipping & Silence Detection**

## Installation

```bash
# As submodule
git submodule add https://github.com/YOUR_USERNAME/Hefakti-DSP.git

# Build locally
cd Hefakti-DSP
pip install -e .
```

## Usage

```python
import numpy as np
import soundfile as sf
from hefakti_dsp.bindings import soundtouch_bindings

# Load audio
audio, sr = sf.read("music.wav")
audio = audio.astype(np.float32)

# Detect BPM
bpm = soundtouch_bindings.detect_bpm(audio, sr)
print(f"BPM: {bpm}")
```

## License

LGPL-2.1 (SoundTouch/Audacity license)
