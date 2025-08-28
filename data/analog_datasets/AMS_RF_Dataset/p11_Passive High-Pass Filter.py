import math
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
circuit = Circuit('Passive High-Pass Filter')
# Input voltage source (DC for operating point)
circuit.V('in', 'Vin', circuit.gnd, 1.0) # 1V DC
# Capacitor in series with input
circuit.C('1', 'Vin', 'Vout', 10@u_nF)
# Resistor from output to ground
circuit.R('1', 'Vout', circuit.gnd, 10@u_kΩ)
simulator = circuit.simulator()
has_vin = False
for element in circuit.elements:
    if "vin" in element.name.lower():
        element.dc_value = "dc 2.5 ac 1"
        has_vin = True
        break

if not has_vin:
    circuit.V('in', 'Vin', circuit.gnd, dc_value=0, ac_value=1)

import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    # Only AC analysis
    ac_analysis = simulator.ac(start_frequency=1@u_Hz, stop_frequency=1@u_GHz, 
                              number_of_points=1000, variation='dec')
except:
    print("Analysis failed.")
    sys.exit(2)

# Get frequency response data
frequencies = np.array(ac_analysis.frequency)

node = 'Vout'

has_node = False
# find any node with "vout"
for element in circuit.elements:
    # get pins
    for pin in element.pins:
        if "vout" == str(pin.node).lower():
            has_node = True
            break

if has_node == False:
    for element in circuit.elements:
        for pin in element.pins:
            if "vout" in str(pin.node).lower():
                node = str(pin.node)
                break

vout_ac = np.array(ac_analysis[node])
gain_db = 20 * np.log10(np.abs(vout_ac))
phase = np.angle(vout_ac, deg=True)

# Create frequency domain plot
plt.figure(figsize=(10, 6))
plt.semilogx(frequencies, gain_db)
plt.title('Frequency Response of High-Pass Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain [dB]')
plt.grid(True)

plt.axhline(y=-3, color='g', linestyle='--', label='-3dB Point')
plt.legend()

plt.tight_layout()
plt.savefig('p11_figure.png')

# Basic High-Pass Filter Verification - Including Monotonicity Check
# 1. Check High-Frequency Gain
high_freq_gain = gain_db[-1]  # Gain at highest frequency
print(f"Gain at highest frequency ({frequencies[-1]:.2f} Hz): {high_freq_gain:.2f} dB")

# 2. Check low frequency attenuation
low_freq_gain = gain_db[0]  # Gain at lowest frequency
print(f"Gain at lowest frequency ({frequencies[0]:.2f} Hz): {low_freq_gain:.2f} dB")
low_freq_attenuation = high_freq_gain - low_freq_gain
print(f"Low frequency attenuation: {low_freq_attenuation:.2f} dB")

# 3. Find the approximate -3dB point
idx_3db = np.argmin(np.abs(gain_db - (high_freq_gain-3)))
cutoff_freq = frequencies[idx_3db]
print(f"Approximate -3dB cutoff frequency: {cutoff_freq:.2f} Hz")

# 4. Check monotonicity
# Use smoothing to reduce measurement noise
window_size = min(11, len(gain_db) // 20)  #  Use window smoothing
if window_size % 2 == 0:  # Ensure window size is odd
    window_size += 1
    
if window_size > 2:  # If there are enough points to smooth
    from scipy.signal import savgol_filter
    smoothed_gain = savgol_filter(gain_db, window_size, 1)  # Use 1st order polynomial smoothing
else:
    smoothed_gain = gain_db
    
# Calculate the difference of the smoothed gain - note that a high-pass filter should increase with frequency
diff_gain = np.diff(smoothed_gain)
non_monotonic_points = np.sum(diff_gain < -0.5)  # Allow a small decrease of 0.5dB

if non_monotonic_points > 0:
    monotonic_percentage = 100 * (1 - non_monotonic_points / len(diff_gain))
    print(f"Warning: Gain is not strictly monotonically increasing.")
    print(f"Monotonicity: {monotonic_percentage:.1f}% of frequency points")
    if monotonic_percentage < 90:  # if non-monotonic points exceed 10%
        print("This may not be a well-behaved high-pass filter.")
else:
    print("Filter response is monotonically increasing with frequency, as expected.")

# 5. Determine if it meets high-pass characteristics
if low_freq_attenuation > 2 and (non_monotonic_points == 0 or monotonic_percentage >= 90):
    print("The circuit exhibits proper high-pass filter characteristics.")
    sys.exit(0)
else:
    print("The circuit does not show expected high-pass filter characteristics.")
    sys.exit(2)