from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib.pyplot as plt
import numpy as np

# Create the circuit
circuit = Circuit('Sample and Hold Circuit')

# Define power supply
circuit.V('dd', 'Vdd', circuit.gnd, 5@u_V)  # 5V power supply

# Define input signal (sinusoidal)
circuit.SinusoidalVoltageSource('input', 'Vin', circuit.gnd, 
                               amplitude=2.5@u_V,  # 2.5V amplitude
                               frequency=1@u_kHz)  # 1kHz frequency

# Define control signal (pulse for sampling)
circuit.PulseVoltageSource('control', 'Ctrl', circuit.gnd,
                          initial_value=0@u_V,      # Start at 0V
                          pulsed_value=5@u_V,       # Pulse to 5V
                          pulse_width=20@u_us,      # 20μs pulse width
                          period=100@u_us,          # 100μs period (10kHz)
                          rise_time=1@u_ns,         # Fast rise
                          fall_time=1@u_ns)         # Fast fall

# Define MOSFET as switch (NMOS)
circuit.MOSFET('M1', 'node1', 'Ctrl', 'Vin', circuit.gnd, model='NMOS')

# Define hold capacitor with initial condition
circuit.C('hold', 'node1', circuit.gnd, 10@u_nF, ic=0@u_V)  # 10nF capacitor with 0V initial condition

# Define buffer (source follower) to prevent loading of capacitor
circuit.MOSFET('M2', 'Vdd', 'node1', 'Vout', circuit.gnd, model='NMOS')
circuit.I('bias', 'Vout', circuit.gnd, 100@u_uA)  # 100μA current source bias

# MOSFET models
circuit.model('NMOS', 'nmos', 
              level=1,
              kp=120e-6,    # Transconductance parameter
              vto=0.7,      # Threshold voltage
              lambda_=0.02, # Channel-length modulation
              w=50e-6,      # Width
              l=1e-6)       # Length

# Setup simulation
simulator = circuit.simulator(temperature=25, nominal_temperature=25)

# Perform transient analysis
analysis = simulator.transient(
    step_time=0.1@u_us,    # 100ns step time
    end_time=3000@u_us,     # 500μs simulation time
)

# Plot results
plt.figure(figsize=(12, 8))

# Plot input signal
plt.subplot(3, 1, 1)
plt.plot(analysis.time*1e6, analysis['Vin'])  # Time in μs
plt.title('Input Signal')
plt.ylabel('Voltage (V)')
plt.grid(True)

# Plot control signal
plt.subplot(3, 1, 2)
plt.plot(analysis.time*1e6, analysis['Ctrl'])  # Time in μs
plt.title('Control Signal')
plt.ylabel('Voltage (V)')
plt.grid(True)

# Plot output signal
plt.subplot(3, 1, 3)
plt.plot(analysis.time*1e6, analysis['Vout'])  # Time in μs
plt.title('Output Signal (Sampled & Held)')
plt.xlabel('Time (μs)')
plt.ylabel('Voltage (V)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Optional: Print some key measurements
print("Simulation completed successfully!")
print(f"Input signal frequency: 1 kHz")
print(f"Sampling frequency: 10 kHz")
print(f"Hold capacitor: 10 nF")