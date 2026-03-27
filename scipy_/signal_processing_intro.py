# ---------------------------------------------------------
# Program: Signal Processing Intro
# Description:
# This program demonstrates basic signal smoothing
# using scipy.signal.
# ---------------------------------------------------------

import numpy as np
from scipy.signal import savgol_filter

# Sample noisy signal
signal = np.array([1, 3, 2, 5, 7, 8, 6, 5, 7, 9, 8])

# Apply Savitzky-Golay filter
smoothed_signal = savgol_filter(signal, window_length=5, polyorder=2)

print("Original signal:")
print(signal)

print("\nSmoothed signal:")
print(smoothed_signal)
