/**
 * MIR (Music Information Retrieval) Bindings
 * ==========================================
 *
 * pybind11 bindings for Audacity's MIR algorithms.
 * Provides BPM and time signature detection using Tatum Quantization Fit.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "mir/MirTypes.h"
#include "mir/MusicInformationRetrieval.h"
#include "mir/GetMeterUsingTatumQuantizationFit.h"

namespace py = pybind11;

/**
 * Numpy array wrapper implementing MirAudioReader interface.
 */
class NumpyAudioReader : public MIR::MirAudioReader {
public:
    NumpyAudioReader(py::array_t<float> samples, double sample_rate)
        : samples_(samples), sample_rate_(sample_rate) {
        auto buf = samples_.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Expected 1D array of samples");
        }
        num_samples_ = buf.size;
        data_ = static_cast<float*>(buf.ptr);
    }

    double GetSampleRate() const override {
        return sample_rate_;
    }

    long long GetNumSamples() const override {
        return num_samples_;
    }

    void ReadFloats(float* buffer, long long where, size_t numFrames) const override {
        // Bounds checking
        long long end = where + static_cast<long long>(numFrames);
        if (where < 0) where = 0;
        if (end > num_samples_) end = num_samples_;

        size_t actual_frames = static_cast<size_t>(end - where);
        if (actual_frames > 0 && where >= 0 && where < num_samples_) {
            std::copy(data_ + where, data_ + where + actual_frames, buffer);
        }

        // Zero-pad if we read past the end
        if (actual_frames < numFrames) {
            std::fill(buffer + actual_frames, buffer + numFrames, 0.0f);
        }
    }

private:
    py::array_t<float> samples_;
    double sample_rate_;
    long long num_samples_;
    float* data_;
};

/**
 * Result structure for BPM/meter detection.
 */
struct MeterResult {
    double bpm;
    int time_signature_numerator;
    int time_signature_denominator;
    bool has_time_signature;
};

/**
 * Detect BPM and time signature from audio using Tatum Quantization Fit.
 *
 * This is Audacity's state-of-the-art tempo detection algorithm, designed
 * specifically for loop detection with low false positive rates.
 *
 * @param samples Audio samples as numpy array (float32, mono)
 * @param sample_rate Sample rate in Hz
 * @param strict Use strict tolerance (fewer false positives, may miss some loops)
 * @param progress_callback Optional callback for progress updates (0.0 to 1.0)
 * @return MeterResult with BPM and optional time signature, or nullopt if not detected
 */
std::optional<MeterResult> detect_meter(
    py::array_t<float> samples,
    double sample_rate,
    bool strict = true,
    std::function<void(double)> progress_callback = nullptr
) {
    NumpyAudioReader reader(samples, sample_rate);

    auto tolerance = strict ? MIR::FalsePositiveTolerance::Strict
                            : MIR::FalsePositiveTolerance::Lenient;

    std::function<void(double)> callback = progress_callback ? progress_callback
                                                              : [](double) {};

    auto result = MIR::GetMusicalMeterFromSignal(reader, tolerance, callback, nullptr);

    if (!result.has_value()) {
        return std::nullopt;
    }

    MeterResult meter;
    meter.bpm = result->bpm;

    if (result->timeSignature.has_value()) {
        meter.has_time_signature = true;
        meter.time_signature_numerator = MIR::GetNumerator(*result->timeSignature);
        meter.time_signature_denominator = MIR::GetDenominator(*result->timeSignature);
    } else {
        meter.has_time_signature = false;
        meter.time_signature_numerator = 4;
        meter.time_signature_denominator = 4;
    }

    return meter;
}

/**
 * Simple BPM detection - returns just the BPM value.
 *
 * @param samples Audio samples as numpy array (float32, mono)
 * @param sample_rate Sample rate in Hz
 * @param strict Use strict tolerance
 * @return Detected BPM, or 0 if detection failed
 */
double detect_bpm(
    py::array_t<float> samples,
    double sample_rate,
    bool strict = true
) {
    auto result = detect_meter(samples, sample_rate, strict, nullptr);
    return result.has_value() ? result->bpm : 0.0;
}

/**
 * Extract BPM from filename if present.
 *
 * Looks for patterns like "120bpm", "120-bpm", "120_bpm" in the filename.
 *
 * @param filename The filename to parse
 * @return Detected BPM, or 0 if not found
 */
double bpm_from_filename(const std::string& filename) {
    auto result = MIR::GetBpmFromFilename(filename);
    return result.has_value() ? *result : 0.0;
}


PYBIND11_MODULE(mir_bindings, m) {
    m.doc() = R"pbdoc(
        Music Information Retrieval
        ===========================

        Python bindings for Audacity's MIR algorithms.

        Features:
            - BPM detection using Tatum Quantization Fit
            - Time signature detection (4/4, 3/4, 6/8, 2/2)
            - BPM extraction from filename

        The detection algorithm is optimized for loops and short audio samples
        (< 60 seconds). It has been tuned for low false positive rates, making
        it suitable for automatic tempo detection in music production workflows.

        Usage:
            import numpy as np
            import soundfile as sf
            from hefakti_dsp.bindings import mir_bindings

            audio, sr = sf.read("loop.wav")
            audio = audio.astype(np.float32)

            # Simple BPM detection
            bpm = mir_bindings.detect_bpm(audio, sr)
            print(f"BPM: {bpm}")

            # Full meter detection with time signature
            meter = mir_bindings.detect_meter(audio, sr)
            if meter is not None:
                print(f"BPM: {meter.bpm}")
                if meter.has_time_signature:
                    print(f"Time signature: {meter.time_signature_numerator}/{meter.time_signature_denominator}")

            # Extract BPM from filename
            bpm = mir_bindings.bpm_from_filename("drums_120bpm.wav")
    )pbdoc";

    // MeterResult class
    py::class_<MeterResult>(m, "MeterResult", R"pbdoc(
        Result of meter detection.

        Attributes:
            bpm: Detected tempo in beats per minute
            time_signature_numerator: Numerator of time signature (e.g., 4 in 4/4)
            time_signature_denominator: Denominator of time signature (e.g., 4 in 4/4)
            has_time_signature: Whether a time signature was detected
    )pbdoc")
        .def_readonly("bpm", &MeterResult::bpm)
        .def_readonly("time_signature_numerator", &MeterResult::time_signature_numerator)
        .def_readonly("time_signature_denominator", &MeterResult::time_signature_denominator)
        .def_readonly("has_time_signature", &MeterResult::has_time_signature)
        .def("__repr__", [](const MeterResult& r) {
            std::string ts = r.has_time_signature
                ? std::to_string(r.time_signature_numerator) + "/" + std::to_string(r.time_signature_denominator)
                : "unknown";
            return "<MeterResult bpm=" + std::to_string(r.bpm) + " time_signature=" + ts + ">";
        });

    // Functions
    m.def("detect_meter", &detect_meter, R"pbdoc(
        Detect BPM and time signature from audio.

        Uses Audacity's Tatum Quantization Fit algorithm, which is optimized
        for loop detection with low false positive rates.

        Note: Works best with loops under 60 seconds. Longer files will return None.

        Args:
            samples: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz
            strict: Use strict tolerance (default True). Strict mode has fewer
                    false positives but may miss some loops. Use False for more
                    lenient detection.
            progress_callback: Optional callback function(progress: float) for
                              progress updates (0.0 to 1.0)

        Returns:
            MeterResult with BPM and time signature, or None if not detected
    )pbdoc",
        py::arg("samples"),
        py::arg("sample_rate"),
        py::arg("strict") = true,
        py::arg("progress_callback") = nullptr
    );

    m.def("detect_bpm", &detect_bpm, R"pbdoc(
        Simple BPM detection from audio.

        Convenience function that returns just the BPM value.

        Args:
            samples: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz
            strict: Use strict tolerance (default True)

        Returns:
            Detected BPM, or 0.0 if detection failed
    )pbdoc",
        py::arg("samples"),
        py::arg("sample_rate"),
        py::arg("strict") = true
    );

    m.def("bpm_from_filename", &bpm_from_filename, R"pbdoc(
        Extract BPM from filename.

        Looks for patterns like "120bpm", "120-bpm", "120_bpm", "120.bpm" in
        the filename. Case insensitive. Only returns values between 30 and 300.

        Args:
            filename: The filename to parse

        Returns:
            Detected BPM, or 0.0 if not found
    )pbdoc",
        py::arg("filename")
    );

    m.attr("__version__") = "0.1.0";
}
