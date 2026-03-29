# ---------------------------------------------------------
# Program: FFT Signal Analysis
# Description:
# This program demonstrates Fast Fourier Transform (FFT)
# using scipy.fft to analyze signal frequency components.
# ---------------------------------------------------------

import numpy as np
from scipy.fft import fft, fftfreq

# Generate sample signal
sample_count = 100
time_step = 1.0 / 100
x = np.linspace(0.0, 1.0, sample_count, endpoint=False)
signal = np.sin(2.0 * np.pi * 5.0 * x) + 0.5 * np.sin(2.0 * np.pi * 10.0 * x)

# Compute FFT
signal_fft = fft(signal)
frequencies = fftfreq(sample_count, time_step)

print("Frequencies:")
print(frequencies[:sample_count // 2])

print("\nFFT magnitudes:")
print(np.abs(signal_fft[:sample_count // 2]))
