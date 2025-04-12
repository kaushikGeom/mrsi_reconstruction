import numpy as np
import matplotlib.pyplot as plt

# Simulating an MRSI frequency spectrum (random peaks)
freq_points = 96
x = np.linspace(-50, 50, freq_points)  # Frequency axis (arbitrary units)
spectrum = np.exp(-((x - 10) ** 2) / 50) + np.exp(-((x + 20) ** 2) / 30)  # Example spectrum
spectrum = np.fft.fft(spectrum)  # Simulating a spectrum in frequency domain

# Without fftshift
spectrum_unshifted = np.abs(spectrum)

# With fftshift
spectrum_shifted = np.abs(np.fft.fftshift(spectrum))

# Plot the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(freq_points), spectrum_unshifted)
plt.title("Without fftshift")
plt.xlabel("Frequency Index")
plt.ylabel("Magnitude")

plt.subplot(1, 2, 2)
plt.plot(np.fft.fftshift(range(freq_points)), spectrum_shifted)  # Shift x-axis for alignment
plt.title("With fftshift")
plt.xlabel("Frequency Index")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()
