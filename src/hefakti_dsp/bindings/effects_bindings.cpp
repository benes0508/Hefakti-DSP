/**
 * Effects Bindings
 * ================
 *
 * pybind11 bindings for audio analysis utilities.
 * Provides RMS, peak detection, and clipping detection.
 *
 * Note: The Audacity effects code requires WaveChannel infrastructure.
 * These bindings provide standalone implementations of the same algorithms.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <vector>
#include <algorithm>

namespace py = pybind11;

/**
 * Compute RMS (Root Mean Square) level of audio.
 *
 * @param samples Audio samples as numpy array (float32)
 * @return RMS level (0.0 to 1.0 for normalized audio)
 */
float compute_rms(py::array_t<float> samples) {
    auto buf = samples.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array of samples");
    }

    float* data = static_cast<float*>(buf.ptr);
    size_t num_samples = buf.size;

    if (num_samples == 0) return 0.0f;

    double sum_squares = 0.0;
    for (size_t i = 0; i < num_samples; i++) {
        double sample = data[i];
        sum_squares += sample * sample;
    }

    return static_cast<float>(std::sqrt(sum_squares / num_samples));
}

/**
 * Find min and max sample values.
 *
 * @param samples Audio samples as numpy array (float32)
 * @return Tuple of (min, max)
 */
std::pair<float, float> get_min_max(py::array_t<float> samples) {
    auto buf = samples.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array of samples");
    }

    float* data = static_cast<float*>(buf.ptr);
    size_t num_samples = buf.size;

    if (num_samples == 0) return {0.0f, 0.0f};

    float min_val = data[0];
    float max_val = data[0];

    for (size_t i = 1; i < num_samples; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    return {min_val, max_val};
}

/**
 * Get peak amplitude (maximum absolute value).
 *
 * @param samples Audio samples as numpy array (float32)
 * @return Peak amplitude
 */
float get_peak(py::array_t<float> samples) {
    auto [min_val, max_val] = get_min_max(samples);
    return std::max(std::abs(min_val), std::abs(max_val));
}

/**
 * Detect clipping in audio.
 *
 * Finds regions where consecutive samples are at or near the maximum level.
 *
 * @param samples Audio samples as numpy array (float32)
 * @param threshold Clipping threshold (default 1.0)
 * @param min_consecutive Minimum consecutive clipped samples (default 3)
 * @return Vector of (start_sample, end_sample) pairs for clipped regions
 */
std::vector<std::pair<size_t, size_t>> find_clipping(
    py::array_t<float> samples,
    float threshold = 1.0f,
    int min_consecutive = 3
) {
    auto buf = samples.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array of samples");
    }

    float* data = static_cast<float*>(buf.ptr);
    size_t num_samples = buf.size;

    std::vector<std::pair<size_t, size_t>> clips;

    size_t clip_start = 0;
    int clip_count = 0;
    bool in_clip = false;

    for (size_t i = 0; i < num_samples; i++) {
        bool is_clipped = std::abs(data[i]) >= threshold;

        if (is_clipped) {
            if (!in_clip) {
                clip_start = i;
                in_clip = true;
            }
            clip_count++;
        } else {
            if (in_clip) {
                if (clip_count >= min_consecutive) {
                    clips.push_back({clip_start, i - 1});
                }
                in_clip = false;
                clip_count = 0;
            }
        }
    }

    // Handle clip at end of file
    if (in_clip && clip_count >= min_consecutive) {
        clips.push_back({clip_start, num_samples - 1});
    }

    return clips;
}

/**
 * Detect silence in audio.
 *
 * @param samples Audio samples as numpy array (float32)
 * @param threshold Silence threshold (default 0.001)
 * @param min_duration_samples Minimum silence duration in samples (default 1000)
 * @return Vector of (start_sample, end_sample) pairs for silent regions
 */
std::vector<std::pair<size_t, size_t>> find_silence(
    py::array_t<float> samples,
    float threshold = 0.001f,
    size_t min_duration_samples = 1000
) {
    auto buf = samples.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array of samples");
    }

    float* data = static_cast<float*>(buf.ptr);
    size_t num_samples = buf.size;

    std::vector<std::pair<size_t, size_t>> silences;

    size_t silence_start = 0;
    bool in_silence = false;

    for (size_t i = 0; i < num_samples; i++) {
        bool is_silent = std::abs(data[i]) < threshold;

        if (is_silent) {
            if (!in_silence) {
                silence_start = i;
                in_silence = true;
            }
        } else {
            if (in_silence) {
                size_t duration = i - silence_start;
                if (duration >= min_duration_samples) {
                    silences.push_back({silence_start, i - 1});
                }
                in_silence = false;
            }
        }
    }

    // Handle silence at end of file
    if (in_silence) {
        size_t duration = num_samples - silence_start;
        if (duration >= min_duration_samples) {
            silences.push_back({silence_start, num_samples - 1});
        }
    }

    return silences;
}

/**
 * Compute RMS envelope over time.
 *
 * @param samples Audio samples as numpy array (float32)
 * @param window_size Window size in samples
 * @param hop_size Hop size in samples (default = window_size)
 * @return Array of RMS values
 */
py::array_t<float> rms_envelope(
    py::array_t<float> samples,
    size_t window_size,
    size_t hop_size = 0
) {
    auto buf = samples.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array of samples");
    }

    float* data = static_cast<float*>(buf.ptr);
    size_t num_samples = buf.size;

    if (hop_size == 0) hop_size = window_size;

    size_t num_frames = (num_samples >= window_size)
                        ? (num_samples - window_size) / hop_size + 1
                        : 0;

    py::array_t<float> output(num_frames);
    auto out_buf = output.request();
    float* out_ptr = static_cast<float*>(out_buf.ptr);

    for (size_t frame = 0; frame < num_frames; frame++) {
        size_t start = frame * hop_size;
        double sum_squares = 0.0;

        for (size_t i = 0; i < window_size; i++) {
            double sample = data[start + i];
            sum_squares += sample * sample;
        }

        out_ptr[frame] = static_cast<float>(std::sqrt(sum_squares / window_size));
    }

    return output;
}


PYBIND11_MODULE(effects_bindings, m) {
    m.doc() = R"pbdoc(
        Audio Analysis Utilities
        ========================

        Python bindings for audio level analysis.

        Functions:
            compute_rms: Calculate RMS level
            get_min_max: Find min/max sample values
            get_peak: Get peak amplitude
            find_clipping: Detect clipped regions
            find_silence: Detect silent regions
            rms_envelope: Compute RMS over time

        Usage:
            import numpy as np
            import soundfile as sf
            from hefakti_dsp.bindings import effects_bindings

            audio, sr = sf.read("music.wav")
            audio = audio.astype(np.float32)

            # Get overall levels
            rms = effects_bindings.compute_rms(audio)
            peak = effects_bindings.get_peak(audio)
            print(f"RMS: {rms:.4f}, Peak: {peak:.4f}")

            # Find clipping
            clips = effects_bindings.find_clipping(audio)
            print(f"Found {len(clips)} clipped regions")

            # Get RMS envelope
            envelope = effects_bindings.rms_envelope(audio, window_size=1024)
    )pbdoc";

    m.def("compute_rms", &compute_rms, R"pbdoc(
        Compute RMS (Root Mean Square) level of audio.

        Args:
            samples: Audio samples as numpy array (float32)

        Returns:
            RMS level (0.0 to 1.0 for normalized audio)
    )pbdoc",
        py::arg("samples")
    );

    m.def("get_min_max", &get_min_max, R"pbdoc(
        Find minimum and maximum sample values.

        Args:
            samples: Audio samples as numpy array (float32)

        Returns:
            Tuple of (min_value, max_value)
    )pbdoc",
        py::arg("samples")
    );

    m.def("get_peak", &get_peak, R"pbdoc(
        Get peak amplitude (maximum absolute value).

        Args:
            samples: Audio samples as numpy array (float32)

        Returns:
            Peak amplitude
    )pbdoc",
        py::arg("samples")
    );

    m.def("find_clipping", &find_clipping, R"pbdoc(
        Detect clipping in audio.

        Finds regions where consecutive samples are at or near the maximum level.

        Args:
            samples: Audio samples as numpy array (float32)
            threshold: Clipping threshold (default 1.0)
            min_consecutive: Minimum consecutive clipped samples (default 3)

        Returns:
            List of (start_sample, end_sample) tuples for clipped regions
    )pbdoc",
        py::arg("samples"),
        py::arg("threshold") = 1.0f,
        py::arg("min_consecutive") = 3
    );

    m.def("find_silence", &find_silence, R"pbdoc(
        Detect silence in audio.

        Args:
            samples: Audio samples as numpy array (float32)
            threshold: Silence threshold (default 0.001)
            min_duration_samples: Minimum silence duration in samples (default 1000)

        Returns:
            List of (start_sample, end_sample) tuples for silent regions
    )pbdoc",
        py::arg("samples"),
        py::arg("threshold") = 0.001f,
        py::arg("min_duration_samples") = 1000
    );

    m.def("rms_envelope", &rms_envelope, R"pbdoc(
        Compute RMS envelope over time.

        Args:
            samples: Audio samples as numpy array (float32)
            window_size: Window size in samples
            hop_size: Hop size in samples (default = window_size)

        Returns:
            Array of RMS values, one per frame
    )pbdoc",
        py::arg("samples"),
        py::arg("window_size"),
        py::arg("hop_size") = 0
    );

    m.attr("__version__") = "0.1.0";
}
