/**
 * SoundTouch BPM Detection Bindings
 * ==================================
 *
 * pybind11 bindings for SoundTouch's BPMDetect algorithm.
 * Uses decimation + envelope detection + autocorrelation.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "soundtouch/BPMDetect.h"

namespace py = pybind11;

/**
 * Wrapper class for BPMDetect that accepts numpy arrays.
 */
class PyBPMDetect {
public:
    PyBPMDetect(int sample_rate, int channels = 1)
        : detector(channels, sample_rate), sample_rate_(sample_rate), channels_(channels) {}

    /**
     * Process audio samples and detect BPM.
     *
     * @param samples Audio samples as numpy array (float32)
     * @return Detected BPM (0 if detection failed)
     */
    float detect(py::array_t<float> samples) {
        auto buf = samples.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Expected 1D array of samples");
        }

        float* data = static_cast<float*>(buf.ptr);
        size_t num_samples = buf.size;

        // Feed samples in chunks
        const size_t chunk_size = 4096;
        for (size_t i = 0; i < num_samples; i += chunk_size) {
            size_t this_chunk = std::min(chunk_size, num_samples - i);
            detector.inputSamples(data + i, static_cast<int>(this_chunk));
        }

        return detector.getBpm();
    }

    /**
     * Reset the detector for a new analysis.
     */
    void reset() {
        detector = soundtouch::BPMDetect(channels_, sample_rate_);
    }

    int get_sample_rate() const { return sample_rate_; }
    int get_channels() const { return channels_; }

private:
    soundtouch::BPMDetect detector;
    int sample_rate_;
    int channels_;
};


/**
 * Simple function to detect BPM from audio samples.
 *
 * @param samples Audio samples as numpy array (float32, mono)
 * @param sample_rate Sample rate in Hz
 * @return Detected BPM (0 if detection failed)
 */
float detect_bpm(py::array_t<float> samples, int sample_rate) {
    PyBPMDetect detector(sample_rate);
    return detector.detect(samples);
}


PYBIND11_MODULE(soundtouch_bindings, m) {
    m.doc() = R"pbdoc(
        SoundTouch BPM Detection
        ========================

        Python bindings for SoundTouch's BPMDetect algorithm.

        Usage:
            import numpy as np
            import soundfile as sf
            from bindings import soundtouch_bindings

            # Load audio
            audio, sr = sf.read("music.wav")
            audio = audio.astype(np.float32)

            # Simple detection
            bpm = soundtouch_bindings.detect_bpm(audio, sr)
            print(f"BPM: {bpm}")

            # Or use the detector class for streaming
            detector = soundtouch_bindings.BPMDetect(sr)
            bpm = detector.detect(audio)
    )pbdoc";

    // Simple function
    m.def("detect_bpm", &detect_bpm, R"pbdoc(
        Detect BPM from audio samples.

        Args:
            samples: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz

        Returns:
            Detected BPM (0 if detection failed)
    )pbdoc",
        py::arg("samples"),
        py::arg("sample_rate")
    );

    // Detector class
    py::class_<PyBPMDetect>(m, "BPMDetect", R"pbdoc(
        BPM detector using SoundTouch algorithm.

        The algorithm works by:
        1. Decimating input to ~500 Hz (bass frequencies determine beat)
        2. Computing amplitude envelope
        3. Autocorrelation to find beat period
        4. Converting to BPM (29-200 range)
    )pbdoc")
        .def(py::init<int, int>(),
            py::arg("sample_rate"),
            py::arg("channels") = 1,
            "Create a BPM detector.")
        .def("detect", &PyBPMDetect::detect,
            py::arg("samples"),
            "Process samples and return detected BPM.")
        .def("reset", &PyBPMDetect::reset,
            "Reset detector for new analysis.")
        .def_property_readonly("sample_rate", &PyBPMDetect::get_sample_rate)
        .def_property_readonly("channels", &PyBPMDetect::get_channels);

    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("MIN_BPM") = 29;
    m.attr("MAX_BPM") = 200;
}
