/**
 * ScoreAlign Bindings
 * ===================
 *
 * pybind11 bindings for Audacity's ScoreAlign chroma feature extraction.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "scorealign/audioreader.h"
#include "scorealign/scorealign.h"
#include "scorealign/gen_chroma.h"

namespace py = pybind11;

/**
 * Numpy array wrapper implementing Audio_reader interface.
 */
class NumpyAudioReader : public Audio_reader {
public:
    NumpyAudioReader(py::array_t<float> samples, double sample_rate)
        : samples_(samples), sample_rate_(sample_rate), position_(0) {
        auto buf = samples_.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Expected 1D array of samples");
        }
        num_samples_ = buf.size;
        data_ = static_cast<float*>(buf.ptr);
    }

    void print_info() override {
        // No-op for Python bindings
    }

    long read(float* data, long n) override {
        long available = num_samples_ - position_;
        long to_read = std::min(n, available);
        if (to_read > 0) {
            std::copy(data_ + position_, data_ + position_ + to_read, data);
            position_ += to_read;
        }
        return to_read;
    }

    double get_sample_rate() override {
        return sample_rate_;
    }

    long get_frames() override {
        return num_samples_;
    }

    void reset() {
        position_ = 0;
    }

private:
    py::array_t<float> samples_;
    double sample_rate_;
    long num_samples_;
    float* data_;
    long position_;
};

/**
 * Extract chroma features from audio.
 *
 * @param samples Audio samples as numpy array (float32, mono)
 * @param sample_rate Sample rate in Hz
 * @param frame_period Frame period in seconds (default 0.2)
 * @param window_size Window size in seconds (default 0.2)
 * @param low_cutoff Low frequency cutoff in Hz (default 40)
 * @param high_cutoff High frequency cutoff in Hz (default 4000)
 * @return 2D numpy array of chroma vectors (num_frames x 12)
 */
py::array_t<float> extract_chroma(
    py::array_t<float> samples,
    double sample_rate,
    double frame_period = 0.2,
    double window_size = 0.2,
    int low_cutoff = 40,
    int high_cutoff = 4000
) {
    NumpyAudioReader reader(samples, sample_rate);

    Scorealign sa;
    sa.frame_period = frame_period;
    sa.window_size = window_size;
    sa.verbose = false;
    sa.progress = nullptr;

    // Calculate parameters for the reader
    reader.calculate_parameters(sa, false);

    float* chrom_energy = nullptr;
    double actual_frame_period = 0;

    int num_frames = sa.gen_chroma_audio(
        reader, high_cutoff, low_cutoff,
        &chrom_energy, &actual_frame_period, 0
    );

    if (num_frames <= 0 || chrom_energy == nullptr) {
        throw std::runtime_error("Chroma extraction failed");
    }

    // Create output array (num_frames x 12)
    py::array_t<float> output({num_frames, CHROMA_BIN_COUNT});
    auto out_buf = output.request();
    float* out_ptr = static_cast<float*>(out_buf.ptr);

    // Copy chroma vectors (each row has 12 bins + 1 silence flag, we skip the flag)
    for (int i = 0; i < num_frames; i++) {
        float* cv = AREF1(chrom_energy, i);
        for (int j = 0; j < CHROMA_BIN_COUNT; j++) {
            out_ptr[i * CHROMA_BIN_COUNT + j] = cv[j];
        }
    }

    free(chrom_energy);
    return output;
}


PYBIND11_MODULE(scorealign_bindings, m) {
    m.doc() = R"pbdoc(
        ScoreAlign Chroma Features
        ==========================

        Python bindings for Audacity's chroma feature extraction.

        Chroma features represent the distribution of energy across the 12
        pitch classes (C, C#, D, ..., B), making them useful for:
        - Music similarity comparison
        - Key detection
        - Chord recognition
        - Audio alignment

        Usage:
            import numpy as np
            import soundfile as sf
            from hefakti_dsp.bindings import scorealign_bindings

            audio, sr = sf.read("music.wav")
            audio = audio.astype(np.float32)

            # Extract chroma features
            chroma = scorealign_bindings.extract_chroma(audio, sr)
            print(f"Chroma shape: {chroma.shape}")  # (num_frames, 12)
    )pbdoc";

    m.def("extract_chroma", &extract_chroma, R"pbdoc(
        Extract chroma features from audio.

        Computes 12-bin chroma vectors representing pitch class energy
        distribution over time.

        Args:
            samples: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz
            frame_period: Frame period in seconds (default 0.2)
            window_size: Window size in seconds (default 0.2)
            low_cutoff: Low frequency cutoff in Hz (default 40)
            high_cutoff: High frequency cutoff in Hz (default 4000)

        Returns:
            2D numpy array of shape (num_frames, 12) containing normalized
            chroma vectors. Each row sums to approximately 1.
    )pbdoc",
        py::arg("samples"),
        py::arg("sample_rate"),
        py::arg("frame_period") = 0.2,
        py::arg("window_size") = 0.2,
        py::arg("low_cutoff") = 40,
        py::arg("high_cutoff") = 4000
    );

    m.attr("__version__") = "0.1.0";
    m.attr("CHROMA_BIN_COUNT") = CHROMA_BIN_COUNT;
}
