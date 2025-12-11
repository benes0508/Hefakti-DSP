/**
 * Vamp Plugin Bindings
 * ====================
 *
 * pybind11 bindings for Vamp audio analysis plugins.
 * Provides: AmplitudeFollower, ZeroCrossing, SpectralCentroid, PercussionOnsetDetector
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "vamp/AmplitudeFollower.h"
#include "vamp/ZeroCrossing.h"
#include "vamp/SpectralCentroid.h"
#include "vamp/PercussionOnsetDetector.h"
#include "vamp/FixedTempoEstimator.h"

#include <vector>
#include <cmath>

namespace py = pybind11;

/**
 * Helper to compute power spectrum for frequency domain plugins.
 */
void compute_power_spectrum(const float* time_domain, float* freq_domain, size_t block_size) {
    // Simple DFT for now - could be optimized with FFT library
    size_t num_bins = block_size / 2 + 1;
    for (size_t k = 0; k < num_bins; ++k) {
        float real = 0.0f, imag = 0.0f;
        for (size_t n = 0; n < block_size; ++n) {
            float angle = -2.0f * M_PI * k * n / block_size;
            real += time_domain[n] * cos(angle);
            imag += time_domain[n] * sin(angle);
        }
        // Store as interleaved real/imag pairs
        freq_domain[k * 2] = real;
        freq_domain[k * 2 + 1] = imag;
    }
}

/**
 * Compute amplitude envelope using AmplitudeFollower plugin.
 *
 * @param samples Audio samples as numpy array (float32, mono)
 * @param sample_rate Sample rate in Hz
 * @param attack Attack time coefficient (0-1, default 0.9)
 * @param release Release time coefficient (0-1, default 0.9)
 * @return Amplitude envelope values
 */
py::array_t<float> amplitude_follower(
    py::array_t<float> samples,
    float sample_rate,
    float attack = 0.9f,
    float release = 0.9f
) {
    auto buf = samples.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array of samples");
    }

    float* data = static_cast<float*>(buf.ptr);
    size_t num_samples = buf.size;

    AmplitudeFollower plugin(sample_rate);

    // Set parameters
    plugin.setParameter("attack", attack);
    plugin.setParameter("release", release);

    // Use default block size
    size_t block_size = 1024;
    size_t step_size = block_size;

    if (!plugin.initialise(1, step_size, block_size)) {
        throw std::runtime_error("Failed to initialize AmplitudeFollower plugin");
    }

    std::vector<float> result;

    // Process in blocks
    for (size_t i = 0; i + block_size <= num_samples; i += step_size) {
        const float* block = data + i;
        const float* const* input_buffers = &block;

        Vamp::RealTime timestamp = Vamp::RealTime::fromSeconds(static_cast<double>(i) / sample_rate);
        auto features = plugin.process(input_buffers, timestamp);

        // Output 0 is the amplitude
        if (features.find(0) != features.end()) {
            for (const auto& feature : features[0]) {
                if (!feature.values.empty()) {
                    result.push_back(feature.values[0]);
                }
            }
        }
    }

    // Get remaining features
    auto remaining = plugin.getRemainingFeatures();
    if (remaining.find(0) != remaining.end()) {
        for (const auto& feature : remaining[0]) {
            if (!feature.values.empty()) {
                result.push_back(feature.values[0]);
            }
        }
    }

    // Create output array
    py::array_t<float> output(result.size());
    auto out_buf = output.request();
    float* out_ptr = static_cast<float*>(out_buf.ptr);
    std::copy(result.begin(), result.end(), out_ptr);

    return output;
}

/**
 * Count zero crossings in audio signal.
 *
 * @param samples Audio samples as numpy array (float32, mono)
 * @param sample_rate Sample rate in Hz
 * @return Zero crossing counts per block
 */
py::array_t<float> zero_crossing_rate(
    py::array_t<float> samples,
    float sample_rate
) {
    auto buf = samples.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array of samples");
    }

    float* data = static_cast<float*>(buf.ptr);
    size_t num_samples = buf.size;

    ZeroCrossing plugin(sample_rate);

    size_t block_size = 1024;
    size_t step_size = block_size;

    if (!plugin.initialise(1, step_size, block_size)) {
        throw std::runtime_error("Failed to initialize ZeroCrossing plugin");
    }

    std::vector<float> result;

    for (size_t i = 0; i + block_size <= num_samples; i += step_size) {
        const float* block = data + i;
        const float* const* input_buffers = &block;

        Vamp::RealTime timestamp = Vamp::RealTime::fromSeconds(static_cast<double>(i) / sample_rate);
        auto features = plugin.process(input_buffers, timestamp);

        // Output 0 is zero crossing counts
        if (features.find(0) != features.end()) {
            for (const auto& feature : features[0]) {
                if (!feature.values.empty()) {
                    // Normalize to rate (crossings per second)
                    float rate = feature.values[0] * sample_rate / block_size;
                    result.push_back(rate);
                }
            }
        }
    }

    auto remaining = plugin.getRemainingFeatures();
    if (remaining.find(0) != remaining.end()) {
        for (const auto& feature : remaining[0]) {
            if (!feature.values.empty()) {
                float rate = feature.values[0] * sample_rate / block_size;
                result.push_back(rate);
            }
        }
    }

    py::array_t<float> output(result.size());
    auto out_buf = output.request();
    float* out_ptr = static_cast<float*>(out_buf.ptr);
    std::copy(result.begin(), result.end(), out_ptr);

    return output;
}

/**
 * Compute spectral centroid (brightness) of audio.
 *
 * @param samples Audio samples as numpy array (float32, mono)
 * @param sample_rate Sample rate in Hz
 * @return Spectral centroid values in Hz
 */
py::array_t<float> spectral_centroid(
    py::array_t<float> samples,
    float sample_rate
) {
    auto buf = samples.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array of samples");
    }

    float* data = static_cast<float*>(buf.ptr);
    size_t num_samples = buf.size;

    SpectralCentroid plugin(sample_rate);

    size_t block_size = 2048;
    size_t step_size = 1024;

    if (!plugin.initialise(1, step_size, block_size)) {
        throw std::runtime_error("Failed to initialize SpectralCentroid plugin");
    }

    std::vector<float> result;
    std::vector<float> freq_buffer(block_size + 2);  // For FFT output

    for (size_t i = 0; i + block_size <= num_samples; i += step_size) {
        // Compute FFT (SpectralCentroid uses FrequencyDomain)
        compute_power_spectrum(data + i, freq_buffer.data(), block_size);

        const float* freq_ptr = freq_buffer.data();
        const float* const* input_buffers = &freq_ptr;

        Vamp::RealTime timestamp = Vamp::RealTime::fromSeconds(
            (static_cast<double>(i) + block_size / 2) / sample_rate
        );
        auto features = plugin.process(input_buffers, timestamp);

        if (features.find(0) != features.end()) {
            for (const auto& feature : features[0]) {
                if (!feature.values.empty()) {
                    result.push_back(feature.values[0]);
                }
            }
        }
    }

    auto remaining = plugin.getRemainingFeatures();
    if (remaining.find(0) != remaining.end()) {
        for (const auto& feature : remaining[0]) {
            if (!feature.values.empty()) {
                result.push_back(feature.values[0]);
            }
        }
    }

    py::array_t<float> output(result.size());
    auto out_buf = output.request();
    float* out_ptr = static_cast<float*>(out_buf.ptr);
    std::copy(result.begin(), result.end(), out_ptr);

    return output;
}

/**
 * Detect percussion onsets in audio.
 *
 * @param samples Audio samples as numpy array (float32, mono)
 * @param sample_rate Sample rate in Hz
 * @param threshold Detection threshold (default 3.0)
 * @param sensitivity Detection sensitivity (default 40.0)
 * @return Onset times in seconds
 */
py::array_t<double> percussion_onsets(
    py::array_t<float> samples,
    float sample_rate,
    float threshold = 3.0f,
    float sensitivity = 40.0f
) {
    auto buf = samples.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array of samples");
    }

    float* data = static_cast<float*>(buf.ptr);
    size_t num_samples = buf.size;

    PercussionOnsetDetector plugin(sample_rate);

    plugin.setParameter("threshold", threshold);
    plugin.setParameter("sensitivity", sensitivity);

    size_t block_size = plugin.getPreferredBlockSize();
    if (block_size == 0) block_size = 2048;
    size_t step_size = plugin.getPreferredStepSize();
    if (step_size == 0) step_size = 1024;

    if (!plugin.initialise(1, step_size, block_size)) {
        throw std::runtime_error("Failed to initialize PercussionOnsetDetector plugin");
    }

    std::vector<double> onset_times;
    std::vector<float> freq_buffer(block_size + 2);

    for (size_t i = 0; i + block_size <= num_samples; i += step_size) {
        compute_power_spectrum(data + i, freq_buffer.data(), block_size);

        const float* freq_ptr = freq_buffer.data();
        const float* const* input_buffers = &freq_ptr;

        Vamp::RealTime timestamp = Vamp::RealTime::fromSeconds(
            (static_cast<double>(i) + block_size / 2) / sample_rate
        );
        auto features = plugin.process(input_buffers, timestamp);

        // Output 0 is onset detection function, output 1 might be peaks
        for (auto& [output_idx, feature_list] : features) {
            for (const auto& feature : feature_list) {
                if (feature.hasTimestamp) {
                    double time = feature.timestamp.sec + feature.timestamp.nsec / 1e9;
                    onset_times.push_back(time);
                }
            }
        }
    }

    auto remaining = plugin.getRemainingFeatures();
    for (auto& [output_idx, feature_list] : remaining) {
        for (const auto& feature : feature_list) {
            if (feature.hasTimestamp) {
                double time = feature.timestamp.sec + feature.timestamp.nsec / 1e9;
                onset_times.push_back(time);
            }
        }
    }

    py::array_t<double> output(onset_times.size());
    auto out_buf = output.request();
    double* out_ptr = static_cast<double*>(out_buf.ptr);
    std::copy(onset_times.begin(), onset_times.end(), out_ptr);

    return output;
}

/**
 * Estimate tempo of a fixed-tempo audio sample.
 *
 * @param samples Audio samples as numpy array (float32, mono)
 * @param sample_rate Sample rate in Hz
 * @param min_bpm Minimum BPM to consider (default 50)
 * @param max_bpm Maximum BPM to consider (default 190)
 * @return Estimated BPM, or 0 if detection failed
 */
double estimate_tempo(
    py::array_t<float> samples,
    float sample_rate,
    float min_bpm = 50.0f,
    float max_bpm = 190.0f
) {
    auto buf = samples.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array of samples");
    }

    float* data = static_cast<float*>(buf.ptr);
    size_t num_samples = buf.size;

    FixedTempoEstimator plugin(sample_rate);

    plugin.setParameter("minbpm", min_bpm);
    plugin.setParameter("maxbpm", max_bpm);

    size_t block_size = plugin.getPreferredBlockSize();
    if (block_size == 0) block_size = 2048;
    size_t step_size = plugin.getPreferredStepSize();
    if (step_size == 0) step_size = 1024;

    if (!plugin.initialise(1, step_size, block_size)) {
        throw std::runtime_error("Failed to initialize FixedTempoEstimator plugin");
    }

    std::vector<float> freq_buffer(block_size + 2);

    for (size_t i = 0; i + block_size <= num_samples; i += step_size) {
        compute_power_spectrum(data + i, freq_buffer.data(), block_size);

        const float* freq_ptr = freq_buffer.data();
        const float* const* input_buffers = &freq_ptr;

        Vamp::RealTime timestamp = Vamp::RealTime::fromSeconds(
            (static_cast<double>(i) + block_size / 2) / sample_rate
        );
        plugin.process(input_buffers, timestamp);
    }

    // Get the tempo from remaining features
    auto remaining = plugin.getRemainingFeatures();

    // Output 0 is typically the tempo
    if (remaining.find(0) != remaining.end() && !remaining[0].empty()) {
        for (const auto& feature : remaining[0]) {
            if (!feature.values.empty()) {
                return static_cast<double>(feature.values[0]);
            }
        }
    }

    return 0.0;
}


PYBIND11_MODULE(vamp_bindings, m) {
    m.doc() = R"pbdoc(
        Vamp Audio Analysis Plugins
        ===========================

        Python bindings for Vamp audio feature extraction plugins.

        Functions:
            amplitude_follower: Track amplitude envelope
            zero_crossing_rate: Compute zero crossing rate
            spectral_centroid: Compute spectral centroid (brightness)
            percussion_onsets: Detect percussive events
            estimate_tempo: Estimate BPM of fixed-tempo audio

        Usage:
            import numpy as np
            import soundfile as sf
            from hefakti_dsp.bindings import vamp_bindings

            audio, sr = sf.read("music.wav")
            audio = audio.astype(np.float32)

            # Track amplitude envelope
            envelope = vamp_bindings.amplitude_follower(audio, sr)

            # Get zero crossing rate
            zcr = vamp_bindings.zero_crossing_rate(audio, sr)

            # Compute spectral centroid (brightness)
            centroid = vamp_bindings.spectral_centroid(audio, sr)

            # Detect percussion onsets
            onsets = vamp_bindings.percussion_onsets(audio, sr)
    )pbdoc";

    m.def("amplitude_follower", &amplitude_follower, R"pbdoc(
        Track amplitude envelope of audio signal.

        Uses an envelope follower with configurable attack and release times.

        Args:
            samples: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz
            attack: Attack time coefficient (0-1, default 0.9)
            release: Release time coefficient (0-1, default 0.9)

        Returns:
            Amplitude envelope values as numpy array
    )pbdoc",
        py::arg("samples"),
        py::arg("sample_rate"),
        py::arg("attack") = 0.9f,
        py::arg("release") = 0.9f
    );

    m.def("zero_crossing_rate", &zero_crossing_rate, R"pbdoc(
        Compute zero crossing rate of audio signal.

        Zero crossing rate is useful for distinguishing speech from music,
        and for onset detection.

        Args:
            samples: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz

        Returns:
            Zero crossing rate per block (crossings per second)
    )pbdoc",
        py::arg("samples"),
        py::arg("sample_rate")
    );

    m.def("spectral_centroid", &spectral_centroid, R"pbdoc(
        Compute spectral centroid of audio signal.

        The spectral centroid indicates the "brightness" of a sound.
        Higher values indicate more high-frequency content.

        Args:
            samples: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz

        Returns:
            Spectral centroid values in Hz
    )pbdoc",
        py::arg("samples"),
        py::arg("sample_rate")
    );

    m.def("percussion_onsets", &percussion_onsets, R"pbdoc(
        Detect percussion onsets in audio.

        Uses high-frequency content ratio method to detect percussive events.

        Args:
            samples: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz
            threshold: Detection threshold (default 3.0)
            sensitivity: Detection sensitivity (default 40.0)

        Returns:
            Onset times in seconds as numpy array
    )pbdoc",
        py::arg("samples"),
        py::arg("sample_rate"),
        py::arg("threshold") = 3.0f,
        py::arg("sensitivity") = 40.0f
    );

    m.def("estimate_tempo", &estimate_tempo, R"pbdoc(
        Estimate tempo of a fixed-tempo audio sample.

        Uses autocorrelation-based tempo estimation. Works best with
        short samples that have a consistent tempo throughout.

        Args:
            samples: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz
            min_bpm: Minimum BPM to consider (default 50)
            max_bpm: Maximum BPM to consider (default 190)

        Returns:
            Estimated BPM, or 0.0 if detection failed
    )pbdoc",
        py::arg("samples"),
        py::arg("sample_rate"),
        py::arg("min_bpm") = 50.0f,
        py::arg("max_bpm") = 190.0f
    );

    m.attr("__version__") = "0.1.0";
}
